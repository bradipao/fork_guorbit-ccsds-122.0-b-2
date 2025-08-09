from __future__ import annotations
import argparse
import os
import numpy as np

from src.io_segmentation import (
    read_bmp, write_bmp,
    iter_segments, segment_view,
    iter_blocks, block_view,
    split_rgb_to_bands
)

def roundtrip_test(input_path: str, out_path: str) -> None:
    img = read_bmp(input_path)
    write_bmp(out_path, img)
    back = read_bmp(out_path)

    same_shape = back.shape == img.shape
    same_dtype = back.dtype == img.dtype
    same_pixels = np.array_equal(back, img)

    print("[Roundtrip]")
    print(f"  shape: {img.shape} -> {back.shape}  (ok={same_shape})")
    print(f"  dtype: {img.dtype} -> {back.dtype}  (ok={same_dtype})")
    print(f"  pixels equal: {same_pixels}")

def segmentation_demo(input_path: str, seg_h: int, seg_w: int, blk_h: int | None, blk_w: int | None, band: int) -> None:
    img = read_bmp(input_path)
    print(f"[Input] shape={img.shape}, dtype={img.dtype}")

    # If RGB, show quick split example (not required to process)
    if img.ndim == 3 and img.shape[2] == 3:
        bands = split_rgb_to_bands(img)
        print("  RGB detected, processing band index:", band)
        assert 0 <= band < 3
    else:
        band = 0

    total_pixels = 0
    seg_count = 0
    blk_count = 0

    for seg in iter_segments(img, seg_h=seg_h, seg_w=seg_w, band_index=band):
        seg_count += 1
        sv = segment_view(img, seg)  # (h,w)
        total_pixels += sv.size

        if blk_h and blk_w:
            for blk in iter_blocks(seg.h, seg.w, blk_h=blk_h, blk_w=blk_w):
                blk_count += 1
                bv = block_view(sv, blk)
                assert bv.shape == (blk.h, blk.w)

    H, W = img.shape[:2]
    print("[Segmentation]")
    print(f"  segments: {seg_count}")
    if blk_h and blk_w:
        print(f"  blocks:   {blk_count} (inside all segments)")
    print(f"  coverage: {total_pixels} of {H*W} pixels  (ok={total_pixels == H*W})")

def main():
    ap = argparse.ArgumentParser(description="CCSDS 122: I/O + Segmentation smoke tests")
    ap.add_argument("input_bmp", help="Path to an 8-bit or 16-bit grayscale BMP, or 8-bit RGB BMP")
    ap.add_argument("--out-bmp", default="roundtrip.bmp", help="Where to write the roundtrip BMP")
    ap.add_argument("--seg", default="128x128", help="Segment size HxW (e.g., 128x128)")
    ap.add_argument("--blk", default=None, help="Optional block size HxW (e.g., 32x32)")
    ap.add_argument("--band", type=int, default=0, help="Band index for RGB (0=R,1=G,2=B)")
    args = ap.parse_args()

    seg_h, seg_w = map(int, args.seg.lower().split("x"))
    if args.blk:
        blk_h, blk_w = map(int, args.blk.lower().split("x"))
    else:
        blk_h = blk_w = None

    print("== Running I/O roundtrip test ==")
    roundtrip_test(args.input_bmp, args.out_bmp)

    print("\n== Running segmentation demo ==")
    segmentation_demo(args.input_bmp, seg_h, seg_w, blk_h, blk_w, args.band)

    if os.path.exists(args.out_bmp):
        print(f"\nSaved roundtrip image to: {args.out_bmp}")

if __name__ == "__main__":
    main()
