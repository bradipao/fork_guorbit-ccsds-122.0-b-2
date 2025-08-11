#Subband Indexing Helper
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable
import numpy as np

from src.bpe_ac_scan import iter_block_families
from src.io_segmentation import BitReader
from src.shared import subband_rects, BandRect
from src.weights import ShiftTable

def apply_subband_weights(coeffs: np.ndarray, levels: int, table: ShiftTable) -> np.ndarray:
    """Left-shift (multiply by 2^Γ) each subband."""
    H, W = coeffs.shape
    R = subband_rects(H, W, levels)
    out = coeffs.copy()

    # Top-level LL
    sLL = table.get((levels, "LL"), 0)
    if sLL:
        LL = R[f"LL{levels}"]
        out[LL.y:LL.y+LL.h, LL.x:LL.x+LL.w] = np.left_shift(out[LL.y:LL.y+LL.h, LL.x:LL.x+LL.w], sLL)

    # Detail bands at each level
    for L in range(levels, 0, -1):
        for sb in ("HL", "LH", "HH"):
            s = table.get((L, sb), 0)
            if not s:
                continue
            rect = R[f"{sb}{L}"]
            y, x, h, w = rect.y, rect.x, rect.h, rect.w
            out[y:y+h, x:x+w] = np.left_shift(out[y:y+h, x:x+w], s)
    return out

def undo_subband_weights(coeffs_w: np.ndarray, levels: int, table: ShiftTable) -> np.ndarray:
    """Right-shift (divide by 2^Γ) each subband. Uses arithmetic shift on signed ints."""
    H, W = coeffs_w.shape
    R = subband_rects(H, W, levels)
    out = coeffs_w.copy()

    sLL = table.get((levels, "LL"), 0)
    if sLL:
        LL = R[f"LL{levels}"]
        out[LL.y:LL.y+LL.h, LL.x:LL.x+LL.w] = np.right_shift(out[LL.y:LL.y+LL.h, LL.x:LL.x+LL.w], sLL)

    for L in range(levels, 0, -1):
        for sb in ("HL", "LH", "HH"):
            s = table.get((L, sb), 0)
            if not s:
                continue
            rect = R[f"{sb}{L}"]
            y, x, h, w = rect.y, rect.x, rect.h, rect.w
            out[y:y+h, x:x+w] = np.right_shift(out[y:y+h, x:x+w], s)
    return out
def tb_of(v: int, b: int, bitshift_gamma: int) -> int:
    """
    tb(x) per §4.5.2:
      0 if |x| < 2^b
      1 if 2^b ≤ |x| < 2^(b+1)
      2 if 2^(b+1) ≤ |x|
     -1 if b < BitShift(Γ)   (forced zero at this plane due to subband scaling)
    """
    if b < bitshift_gamma:
        return -1
    a = abs(int(v))
    if a < (1 << b):            return 0
    if a < (1 << (b+1)):        return 1
    return 2

def bit_b_of(v: int, b: int) -> int:
    """The b-th magnitude bit (MSB=high planes). Used for typesb[...] when tb ∈ {0,1}."""
    return (abs(int(v)) >> b) & 1

def subband_bitshift_lookup(H: int, W: int, levels: int, shifts: dict[str,int]):
    """
    Returns a function (y,x) -> BitShift(Γ) using your shift table (e.g., Table 3-4).
    Pass in your actual shifts dict like {'LL3':3,'HL3':3,'LH3':3,'HH3':3,'HL2':2,...}.
    """
    R = subband_rects(H, W, levels)
    def lookup(y: int, x: int) -> int:
        for name, rect in R.items():
            if rect.y <= y < rect.y+rect.h and rect.x <= x < rect.x+rect.w:
                return shifts[name]
        raise RuntimeError("coord not in any subband")
    return lookup

def tmax(values: Iterable[int]) -> int:
    """max over tb values; empty -> -1 (null)."""
    m = -1
    for v in values:
        if v > m: m = v
    return m

def _read_option_id(br: BitReader, N: int) -> int:
    """Read per-gaggle option ID for pattern size N (Table 4-18)."""
    if N == 2:
        b = br.read_bits(1)
        return 1 if b == 1 else 0
    bits = (br.read_bits(1) << 1) | br.read_bits(1)
    if N == 3:
        return 0 if bits == 0b00 else 1 if bits == 0b01 else 3
    else:
        return bits
    
def compute_bitdepth_per_block(coeffs_w: np.ndarray, levels: int) -> np.ndarray:
    """
    For each 4×4 block (family root LL node), return the number of bitplanes required.
    Uses the max abs() coefficient in the block's descendants.
    """
    block_depths = []

    for bf in iter_block_families(coeffs_w, levels):
        # All coords in the families of this block (parents + children + grandchildren)
        coords = []
        for fam in bf.families:
            coords.extend(fam.coords)
        # Max abs coeff in this block
        max_abs_val = 0
        for (y, x) in coords:
            v = abs(int(coeffs_w[y, x]))
            if v > max_abs_val:
                max_abs_val = v
        if max_abs_val == 0:
            bitplanes = 0
        else:
            bitplanes = max_abs_val.bit_length()  # integer MSB position
        block_depths.append(bitplanes)

    return np.array(block_depths, dtype=np.int32)