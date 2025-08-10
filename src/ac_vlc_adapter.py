# ccsds122/ac_vlc_adapter.py
from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple, Optional

def split_into_gaggles(seq_len: int) -> List[int]:
    """
    Plane-stream splitter: 16,16,... (no per-plane reference for AC planes).
    """
    out = []
    rem = seq_len
    while rem > 0:
        take = min(16, rem)
        out.append(take)
        rem -= take
    return out

# ---- Stage-specific symbol mappings (family-aware signatures) ----
# For now, these are simple 0/1 passthroughs with N=1. Later, replace bodies with
# your spec mappings (Tables 4-14..4-18) and return the correct N (2/3/4) per family.

# Stage 0 (significance)
def map_stage0_significance(bits: Iterable[int], *, family_id: Optional[int] = None) -> tuple[np.ndarray, int]:
    vals = np.fromiter((int(b) & 1 for b in bits), dtype=np.int64)
    return vals, 1

def unmap_stage0_significance(vals: np.ndarray, N: int, *, family_id: Optional[int] = None) -> np.ndarray:
    assert N == 1 or N in (2,3,4)  # when you upgrade, adjust logic here
    return (vals & 1).astype(np.uint8) if N == 1 else (vals & 1).astype(np.uint8)

# Stage 2 (refinement for newly significant)
def map_stage2_refinement(bits: Iterable[int], *, family_id: Optional[int] = None) -> tuple[np.ndarray, int]:
    vals = np.fromiter((int(b) & 1 for b in bits), dtype=np.int64)
    return vals, 1

def unmap_stage2_refinement(vals: np.ndarray, N: int, *, family_id: Optional[int] = None) -> np.ndarray:
    return (vals & 1).astype(np.uint8)

# Stage 3 (refinement for previously significant)
def map_stage3_refinement(bits: Iterable[int], *, family_id: Optional[int] = None) -> tuple[np.ndarray, int]:
    vals = np.fromiter((int(b) & 1 for b in bits), dtype=np.int64)
    return vals, 1

def unmap_stage3_refinement(vals: np.ndarray, N: int, *, family_id: Optional[int] = None) -> np.ndarray:
    return (vals & 1).astype(np.uint8)

# Optional helper if you prefer to query N without building vals:
def stageN_for(family_id: int, stage: int) -> int:
    """
    Return the mapped pattern size N (2/3/4 or 1 for passthrough) for a given family and stage.
    For now everything is 1; update when you implement the real tables.
    """
    return 1
