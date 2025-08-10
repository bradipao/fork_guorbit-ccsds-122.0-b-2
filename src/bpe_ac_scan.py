from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable
import numpy as np
from .utils import subband_rects

@dataclass(frozen=True)
class FamilyCoords:
    """Holds the (y,x) coords of all coefficients in a family for one block."""
    block_index: int
    family_id: int      # 0=F0 (HL*), 1=F1 (LH*), 2=F2 (HH*)
    coords: List[Tuple[int,int]]

@dataclass(frozen=True)
class BlockFamilies:
    """All families for a block, in scan order F0, F1, F2."""
    block_index: int
    families: Tuple[FamilyCoords, FamilyCoords, FamilyCoords]

def _safe_add(rect, y: int, x: int) -> Tuple[int,int] | None:
    """Return (Y,X) inside rect or None if out of bounds."""
    Y = rect.y + y
    X = rect.x + x
    if 0 <= y < rect.h and 0 <= x < rect.w:
        return (Y, X)
    return None

def _coords_parents(rect, r, c) -> List[Tuple[int,int]]:
    out = []
    p = _safe_add(rect, r, c)
    if p: out.append(p)
    return out

def _coords_children(rect, r, c) -> List[Tuple[int,int]]:
    out = []
    for dr in (0,1):
        for dc in (0,1):
            p = _safe_add(rect, 2*r + dr, 2*c + dc)
            if p: out.append(p)
    return out

def _coords_grandchildren(rect, r, c) -> List[Tuple[int,int]]:
    out = []
    # four 2x2 groups around (4r,4c)
    for yoff in (0,2):
        for xoff in (0,2):
            for dr in (0,1):
                for dc in (0,1):
                    p = _safe_add(rect, 4*r + yoff + dr, 4*c + xoff + dc)
                    if p: out.append(p)
    return out

def iter_block_families(coeffs: np.ndarray, levels: int) -> Iterable[BlockFamilies]:
    """
    Yields BlockFamilies for each DC position in LL(top) in raster order.
    Families are F0 (HL*), F1 (LH*), F2 (HH*), each containing:
      parents at level 3, children at level 2 (2x2), grandchildren at level 1 (4x 2x2).
    Coords are absolute indices in 'coeffs'.
    """
    H, W = coeffs.shape
    R = subband_rects(H, W, levels)
    LL3 = R[f"LL{levels}"]

    # family base rects at each level
    HL3, LH3, HH3 = R["HL3"], R["LH3"], R["HH3"]
    HL2, LH2, HH2 = R["HL2"], R["LH2"], R["HH2"]
    HL1, LH1, HH1 = R["HL1"], R["LH1"], R["HH1"]

    bidx = 0
    for r in range(LL3.h):
        for c in range(LL3.w):
            # F0 : HL3 → HL2 → HL1
            f0 = []
            f0 += _coords_parents(HL3, r, c)
            f0 += _coords_children(HL2, r, c)
            f0 += _coords_grandchildren(HL1, r, c)

            # F1 : LH3 → LH2 → LH1
            f1 = []
            f1 += _coords_parents(LH3, r, c)
            f1 += _coords_children(LH2, r, c)
            f1 += _coords_grandchildren(LH1, r, c)

            # F2 : HH3 → HH2 → HH1
            f2 = []
            f2 += _coords_parents(HH3, r, c)
            f2 += _coords_children(HH2, r, c)
            f2 += _coords_grandchildren(HH1, r, c)

            yield BlockFamilies(
                block_index=bidx,
                families=(
                    FamilyCoords(bidx, 0, f0),
                    FamilyCoords(bidx, 1, f1),
                    FamilyCoords(bidx, 2, f2),
                ),
            )
            bidx += 1
