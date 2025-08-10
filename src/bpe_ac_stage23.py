from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .io_segmentation import BitWriter, BitReader

# ---- Stage 2: refinement for NEWLY significant at this plane ----
def encode_stage2_plane(coeffs: np.ndarray,
                        newly_sig_coords: List[Tuple[int,int]],
                        b: int) -> bytes:
    """
    For each coord that became significant at plane b, output this plane's refinement bit:
      bit = (|coeff| >> b) & 1
    Raw (no VLC) scaffold.
    """
    if not newly_sig_coords:
        return b""
    bw = BitWriter()
    for (y, x) in newly_sig_coords:
        mag_bit = (abs(int(coeffs[y, x])) >> b) & 1
        bw.write_bits(mag_bit, 1)
    return bw.to_bytes()

def decode_stage2_plane(shape: Tuple[int,int],
                        newly_sig_coords: List[Tuple[int,int]],
                        b: int,
                        payload: bytes) -> np.ndarray:
    """
    Returns an int8 mask with +1 where refinement bit==1, 0 otherwise.
    (You likely just need to consume bits and sanity-check counts for now.)
    """
    out = np.zeros(shape, dtype=np.int8)
    if not newly_sig_coords:
        return out
    br = BitReader(payload)
    for (y, x) in newly_sig_coords:
        bit = br.read_bits(1)
        if bit == 1:
            out[y, x] = 1
    return out

# ---- Stage 3: refinement for PREVIOUSLY significant ----
def encode_stage3_plane(coeffs: np.ndarray,
                        prev_sig_coords: List[Tuple[int,int]],
                        b: int) -> bytes:
    """
    For each coord that was already significant before plane b, output refinement bit:
      bit = (|coeff| >> b) & 1
    Raw (no VLC) scaffold.
    """
    if not prev_sig_coords:
        return b""
    bw = BitWriter()
    for (y, x) in prev_sig_coords:
        mag_bit = (abs(int(coeffs[y, x])) >> b) & 1
        bw.write_bits(mag_bit, 1)
    return bw.to_bytes()

def decode_stage3_plane(shape: Tuple[int,int],
                        prev_sig_coords: List[Tuple[int,int]],
                        b: int,
                        payload: bytes) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int8)
    if not prev_sig_coords:
        return out
    br = BitReader(payload)
    for (y, x) in prev_sig_coords:
        bit = br.read_bits(1)
        if bit == 1:
            out[y, x] = 1
    return out
