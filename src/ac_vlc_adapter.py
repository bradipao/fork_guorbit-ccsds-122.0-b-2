from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple

def split_into_gaggles(seq_len: int) -> List[int]:
    """
    Plane-stream splitter: 16,16,... (no per-plane reference sample for AC planes).
    """
    out = []
    rem = seq_len
    while rem > 0:
        take = min(16, rem)
        out.append(take)
        rem -= take
    return out

# ---- Stage-specific symbol mappings (temporary: direct 0/1, N=1) ----
# Swap these for Tables 4-14..4-18 when you’re ready.

def map_stage0_significance(bits: Iterable[int]) -> tuple[np.ndarray, int]:
    vals = np.fromiter((int(b) & 1 for b in bits), dtype=np.int64)
    return vals, 1

def unmap_stage0_significance(vals: np.ndarray, N: int) -> np.ndarray:
    assert N == 1
    return (vals & 1).astype(np.uint8)

def map_stage2_refinement(bits: Iterable[int]) -> tuple[np.ndarray, int]:
    vals = np.fromiter((int(b) & 1 for b in bits), dtype=np.int64)
    return vals, 1

def unmap_stage2_refinement(vals: np.ndarray, N: int) -> np.ndarray:
    assert N == 1
    return (vals & 1).astype(np.uint8)

def map_stage3_refinement(bits: Iterable[int]) -> tuple[np.ndarray, int]:
    vals = np.fromiter((int(b) & 1 for b in bits), dtype=np.int64)
    return vals, 1

def unmap_stage3_refinement(vals: np.ndarray, N: int) -> np.ndarray:
    assert N == 1
    return (vals & 1).astype(np.uint8)
