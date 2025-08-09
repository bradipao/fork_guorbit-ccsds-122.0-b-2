from __future__ import annotations
import argparse
import sys
import numpy as np

from src.io_segmentation import (
    read_bmp, write_bmp,
    iter_segments, segment_view,
    split_rgb_to_bands
)
from src.dwt import (
    dwt97m_forward_2d
)
from src.idwt import dwt97m_inverse_2d


def _process_band_roundtrip(band_img: np.ndarray, seg_h: int, seg_w: int, levels: int) -> tuple[np.ndarray, bool]:
    """
    Runs per-segment forward 9/7M DWT then inverse and writes results into a reconstructed band.
    Returns (reconstructed_band, all_ok)
    """
    H, W = band_img.shape
    recon = np.empty_like(band_img)
    all_ok = True

    for seg in iter_segments(band_img, seg_h=seg_h, seg_w=seg_w, band_index=0):
        sv = segment_view(band_img, seg)            
        seg_src = sv.copy()                          

        # Forward -> Inverse on this segment
        coeffs = dwt97m_forward_2d(seg_src, levels=levels)
        seg_rec = dwt97m_inverse_2d(coeffs, levels=levels)

        # Write back
        recon[seg.y:seg.y+seg.h, seg.x:seg.x+seg.w] = seg_rec

        # Verify per-segment identity
        ok = np.array_equal(seg_src, seg_rec)
        all_ok &= ok

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
