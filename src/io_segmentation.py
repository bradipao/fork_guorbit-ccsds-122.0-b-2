from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List
import numpy as np
from PIL import Image

# ---------------------------
# Image I/O (BMP <-> ndarray)
# ---------------------------

def read_bmp(path: str) -> np.ndarray:
    """
    Read a BMP into a numpy array with shape:
      - Gray8: (H, W) uint8
      - Gray16: (H, W) uint16
      - RGB8: (H, W, 3) uint8
    """
    im = Image.open(path)
    if im.mode in ("L", "P"):
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.uint8)
    elif im.mode == "I;16":
        arr = np.asarray(im, dtype=np.uint16)
    elif im.mode == "RGB":
        arr = np.asarray(im, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported BMP mode for lossless path: {im.mode}")
    return arr

def write_bmp(path: str, arr: np.ndarray) -> None:
    """
    Write numpy array to BMP:
      - (H,W) uint8  -> 'L'
      - (H,W) uint16 -> 'I;16'
      - (H,W,3) uint8 -> 'RGB'
    """
    if arr.ndim == 2 and arr.dtype == np.uint8:
        Image.fromarray(arr, mode="L").save(path, format="BMP")
    elif arr.ndim == 2 and arr.dtype == np.uint16:
        Image.fromarray(arr, mode="I;16").save(path, format="BMP")
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
        Image.fromarray(arr, mode="RGB").save(path, format="BMP")
    else:
        raise ValueError(f"Unsupported shape/dtype for BMP: {arr.shape}, {arr.dtype}")

# -------------------------------------------------
# Deterministic segmentation (segments & sub-blocks)
# -------------------------------------------------

@dataclass(frozen=True)
class Block:
    y: int; x: int; h: int; w: int  # segment-local coords

@dataclass(frozen=True)
class Segment:
    y: int; x: int; h: int; w: int  # image-global coords
    band_index: int                 # 0 for gray; 0/1/2 for R/G/B

def iter_segments(img: np.ndarray, seg_h: int, seg_w: int, band_index: int = 0) -> Iterator[Segment]:
    """
    Row-major segmentation without padding. Edge segments may be smaller.
    Works for (H,W) or (H,W,3); band_index selects band for multi-band.
    """
    if img.ndim == 2:
        H, W = img.shape
    elif img.ndim == 3:
        H, W, C = img.shape
        if not (0 <= band_index < C):
            raise IndexError("band_index out of range")
    else:
        raise ValueError("img must be (H,W) or (H,W,C)")
    if seg_h <= 0 or seg_w <= 0:
        raise ValueError("seg_h and seg_w must be positive")

    for y in range(0, H, seg_h):
        h = min(seg_h, H - y)
        for x in range(0, W, seg_w):
            w = min(seg_w, W - x)
            yield Segment(y=y, x=x, h=h, w=w, band_index=band_index)

def segment_view(img: np.ndarray, seg: Segment) -> np.ndarray:
    """Zero-copy view of the segment for the chosen band: (seg.h, seg.w)."""
    if img.ndim == 2:
        return img[seg.y:seg.y+seg.h, seg.x:seg.x+seg.w]
    return img[seg.y:seg.y+seg.h, seg.x:seg.x+seg.w, seg.band_index]

def iter_blocks(h: int, w: int, blk_h: int, blk_w: int) -> Iterator[Block]:
    """Row-major blocks within a segment, no padding (edge blocks may be smaller)."""
    if blk_h <= 0 or blk_w <= 0:
        raise ValueError("blk_h and blk_w must be positive")
    for y in range(0, h, blk_h):
        bh = min(blk_h, h - y)
        for x in range(0, w, blk_w):
            bw = min(blk_w, w - x)
            yield Block(y=y, x=x, h=bh, w=bw)

def block_view(seg_view_arr: np.ndarray, blk: Block) -> np.ndarray:
    """Zero-copy view of a block within a segment view."""
    return seg_view_arr[blk.y:blk.y+blk.h, blk.x:blk.x+blk.w]

# -----------------------
# Convenience for bands
# -----------------------

def split_rgb_to_bands(img_rgb: np.ndarray) -> List[np.ndarray]:
    """Split (H,W,3) uint8 into [R,G,B] views."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3 or img_rgb.dtype != np.uint8:
        raise ValueError("Expected (H,W,3) uint8 RGB image")
    return [img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]]
