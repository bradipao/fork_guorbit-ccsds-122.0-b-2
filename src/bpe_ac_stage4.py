# ccsds122/bpe_ac_stage4.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .io_segmentation import BitWriter, BitReader

def encode_stage4_plane(coeffs: np.ndarray,
                        newly_sig_coords: List[Tuple[int,int]]) -> bytes:
    """
    Stage 4 for one plane: emit one raw sign bit for each coefficient that
    became significant at this plane (in the exact order they were reported).
    Convention: 0 => non-negative, 1 => negative.
    """
    bw = BitWriter()
    for (y, x) in newly_sig_coords:
        sgn = 1 if int(coeffs[y, x]) < 0 else 0
        bw.write_bits(sgn, 1)
    return bw.to_bytes()

def decode_stage4_plane(shape: Tuple[int,int],
                        newly_sig_coords: List[Tuple[int,int]],
                        payload: bytes) -> np.ndarray:
    """
    Mirror of encode_stage4_plane. Returns an array with only the sign bits set to:
      +1 for non-negative, -1 for negative, 0 elsewhere (so you can merge later).
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.int8)
    if not newly_sig_coords:
        return out
    br = BitReader(payload)
    for (y, x) in newly_sig_coords:
        bit = br.read_bits(1)
        out[y, x] = -1 if bit == 1 else +1
    return out
