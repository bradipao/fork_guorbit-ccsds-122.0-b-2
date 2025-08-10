from __future__ import annotations
import argparse
import sys
import numpy as np

from src.bpe_ac_stage23 import decode_stage2_plane, decode_stage3_plane, encode_stage2_plane, encode_stage3_plane
from src.bpe_state import SigRefState
from src.bpe_ac_stage4 import decode_stage4_plane, encode_stage4_plane
from src.io_segmentation import read_bmp, write_bmp, iter_segments, segment_view, split_rgb_to_bands
from src.dwt import dwt97m_forward_2d
from src.idwt import dwt97m_inverse_2d
from src.utils import apply_subband_weights, subband_rects, undo_subband_weights
from src.bpe_dc import encode_dc, decode_dc
from src.bpe_ac_depth import encode_ac_bitdepths, decode_ac_bitdepths
from src.bpe_ac_stage0 import encode_stage0_plane, decode_stage0_plane
from src.bpe_dc import encode_dc, DCEncodeInfo, _bitdepth_ac


def _process_band_roundtrip(band_img: np.ndarray, seg_h: int, seg_w: int, levels: int) -> tuple[np.ndarray, bool]:
    """
    Per-segment pipeline:
      forward 9/7M DWT -> (weights)
      DC encode/decode check
      AC bit-depth encode/decode check
      Stage 0/2/3/4 (raw scaffolds) encode/decode check
      inverse weights -> inverse DWT
    Returns (reconstructed_band, all_ok)
    """
    # If you enable official subband weights, set to 3 (Table 3-4). Keep 0 for identity weights.
    BITSHIFT_LL3 = 3

    H, W = band_img.shape
    recon = np.empty_like(band_img)
    all_ok = True

    for seg in iter_segments(band_img, seg_h=seg_h, seg_w=seg_w, band_index=0):
        sv = segment_view(band_img, seg)
        seg_src = sv.copy()

        # ---- Forward DWT (signed coeffs) ----
        coeffs = dwt97m_forward_2d(seg_src, levels=levels).astype(np.int32)

        # ---- Apply subband weights (identity unless you've enabled real weights) ----
        coeffs_w = apply_subband_weights(coeffs, levels=levels)

        # ===== DC (quantized) encode/decode sanity check =====
        payload_dc, dcinfo = encode_dc(
            coeffs_w, levels=levels, bitshift_ll3=BITSHIFT_LL3, code_sel="opt"
        )
        Hc, Wc = coeffs_w.shape
        rec_cprime = decode_dc((Hc, Wc), levels=levels, payload=payload_dc, info=dcinfo)

        # Quantized LL(top) must match decoder result
        LL = subband_rects(Hc, Wc, levels)[f"LL{levels}"]
        y, x, hh, ww = LL.y, LL.x, LL.h, LL.w
        assert np.array_equal(
            rec_cprime[y:y+hh, x:x+ww],
            (coeffs_w[y:y+hh, x:x+ww] >> dcinfo.q)
        ), "DC encode/decode mismatch on LL(top)"

        # ===== AC bit-depths encode/decode check (§4.4) =====
        ac_payload, acinfo, depths = encode_ac_bitdepths(coeffs_w, levels=levels, code_sel="opt")
        depths_back = decode_ac_bitdepths((Hc, Wc), levels=levels, payload=ac_payload, info=acinfo)
        assert np.array_equal(depths_back, depths), "AC bit-depths encode/decode mismatch"

        # ===== Stages 0/2/3/4 (raw scaffolds) per-plane encode/decode =====
        plane_indices = list(range(acinfo.BitDepthAC - 1, -1, -1))

        # Encoder/decoder state trackers
        enc_state = SigRefState((Hc, Wc))
        dec_state = SigRefState((Hc, Wc))

        # Per-plane payload buffers (keep order!)
        pl0_bytes_list: list[bytes] = []
        pl2_bytes_list: list[bytes] = []
        pl3_bytes_list: list[bytes] = []
        pl4_bytes_list: list[bytes] = []

        # ---- Encode sweep (MSB -> LSB) ----
        for b in plane_indices:
            # Stage 0: significance (+ remaining DC plane if b < q)
            pl0, rep0 = encode_stage0_plane(
                coeffs=coeffs_w,
                levels=levels,
                bitdepth_ac_per_block=depths,
                b=b,
                q=dcinfo.q,
                significance_map=enc_state.sig_map,
            )
            pl0_bytes_list.append(pl0)
            enc_state.note_new(b, rep0.newly_sig_coords)

            # Stage 2: refinement for newly significant at this plane
            pl2 = encode_stage2_plane(coeffs_w, rep0.newly_sig_coords, b,
                          levels=levels, shape_hint=(Hc, Wc))
            pl2_bytes_list.append(pl2)

            # Stage 3: refinement for previously significant (from higher planes)
            prev_coords_enc = enc_state.compute_prev_for_plane(b)
            pl3 = encode_stage3_plane(coeffs_w, prev_coords_enc, b,
                          levels=levels, shape_hint=(Hc, Wc))
            pl3_bytes_list.append(pl3)

            # Stage 4: sign bits for newly significant
            pl4 = encode_stage4_plane(coeffs_w, rep0.newly_sig_coords)
            pl4_bytes_list.append(pl4)

        # ---- Decode sweep (MSB -> LSB), consuming exactly the same payloads ----
        for idx, b in enumerate(plane_indices):
            # Stage 0
            dc_plane_dec, newly_dec = decode_stage0_plane(
                shape=(Hc, Wc),
                levels=levels,
                bitdepth_ac_per_block=depths,
                b=b,
                q=dcinfo.q,
                significance_map=dec_state.sig_map,
                payload=pl0_bytes_list[idx],
            )
            dec_state.note_new(b, newly_dec)

            # Stage 2
            _ = decode_stage2_plane((Hc, Wc), newly_dec, b, pl2_bytes_list[idx], levels=levels)

            # Stage 3
            prev_coords_dec = dec_state.compute_prev_for_plane(b)
            _ = decode_stage3_plane((Hc, Wc), prev_coords_dec, b, pl3_bytes_list[idx], levels=levels)

            # Stage 4
            _signs = decode_stage4_plane((Hc, Wc), newly_dec, pl4_bytes_list[idx])
            # (optional) accumulate a sign map for additional checks

        # Encoder/decoder significance maps must match after Stage 0–4 scaffolds
        assert np.array_equal(enc_state.sig_map, dec_state.sig_map), "Sig map mismatch after Stage 0–4"

        # ---- Inverse weights -> inverse DWT (lossless image round-trip) ----
        coeffs_u = undo_subband_weights(coeffs_w, levels=levels)
        seg_rec = dwt97m_inverse_2d(coeffs_u, levels=levels, out_dtype=seg_src.dtype)

        # Write back reconstructed segment & check identity
        recon[seg.y:seg.y+seg.h, seg.x:seg.x+seg.w] = seg_rec
        all_ok &= np.array_equal(seg_src, seg_rec)

    return recon, all_ok





def main():
    ap = argparse.ArgumentParser(description="CCSDS-122 (lossless path): I/O + segmentation + 9/7M DWT round-trip")
    ap.add_argument("input_bmp", help="8-bit/16-bit grayscale BMP or 8-bit RGB BMP")
    ap.add_argument("--seg", default="128x128", help="Segment size HxW (default: 128x128)")
    ap.add_argument("--levels", type=int, default=3, help="DWT levels (default: 3)")
    ap.add_argument("--rgb", choices=["auto", "band0", "band1", "band2", "all"], default="auto",
                    help="RGB handling: auto (detect & use band0), band{0,1,2}, or all (process all bands)")
    ap.add_argument("--save-recon", default=None, help="Optional path to save reconstructed BMP")
    args = ap.parse_args()

    seg_h, seg_w = map(int, args.seg.lower().split("x"))
    img = read_bmp(args.input_bmp)

    print(f"[Input] shape={img.shape}, dtype={img.dtype}")
    if img.ndim == 2:
        # Grayscale
        recon, ok = _process_band_roundtrip(img, seg_h, seg_w, args.levels)
        img_out = recon
        print(f"[Round-trip] grayscale: {ok}  (global_equal={np.array_equal(img, recon)})")

    else:
        # RGB: decide which bands to process
        bands = split_rgb_to_bands(img)  # views [R,G,B]
        out = img.copy()

        targets = []
        if args.rgb == "auto":
            targets = [0]
        elif args.rgb == "all":
            targets = [0, 1, 2]
        elif args.rgb.startswith("band"):
            targets = [int(args.rgb[-1])]
        else:
            print(f"Unexpected --rgb value: {args.rgb}", file=sys.stderr)
            sys.exit(2)

        global_ok = True
        for b in targets:
            print(f"[Processing] RGB band {b}")
            recon_b, ok = _process_band_roundtrip(bands[b], seg_h, seg_w, args.levels)
            out[..., b] = recon_b
            global_ok &= ok
            print(f"  band{b} per-segment round-trip: {ok}  (band_equal={np.array_equal(bands[b], recon_b)})")

        img_out = out
        print(f"[Round-trip] RGB (processed {targets}): {global_ok}  (global_equal={np.array_equal(img, out)})")

    if args.save_recon:
        write_bmp(args.save_recon, img_out)
        print(f"[Saved] reconstructed image -> {args.save_recon}")


if __name__ == "__main__":
    main()
