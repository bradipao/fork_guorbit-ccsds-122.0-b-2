from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .io_segmentation import BitWriter, BitReader
from .bpe_dc import CodeSel
from .gaggle_vlc import encode_gaggle, decode_gaggle
from .ac_vlc_adapter import split_into_gaggles

def encode_stage4_plane(coeffs: np.ndarray,
                        newly_sig_coords: List[Tuple[int,int]],
                        code_sel: CodeSel = "opt") -> bytes:
    """
    Sign bits for coefficients that became significant at this plane.
    Convention: 0 => non-negative, 1 => negative.
    Encoded as gaggles; N=1 so it behaves like raw bits (no ID).
    """
    if not newly_sig_coords:
        return b""
    vals = np.fromiter((1 if int(coeffs[y, x]) < 0 else 0 for (y, x) in newly_sig_coords), dtype=np.int64)
    sizes = split_into_gaggles(vals.size)
    off = 0
    bw = BitWriter()
    for J in sizes:
        gag = vals[off:off+J]
        by = encode_gaggle(gag, N=1, first_gaggle=False, code_sel=code_sel)
        for ch in by: bw.buf.append(ch)
        off += J
    return bw.to_bytes()

def decode_stage4_plane(shape: Tuple[int,int],
                        newly_sig_coords: List[Tuple[int,int]],
                        payload: bytes,
                        code_sel: CodeSel = "opt") -> np.ndarray:
    """
    Returns an int8 plane with +1 for non-negative, -1 for negative at 'newly' coords.
    """
    out = np.zeros(shape, dtype=np.int8)
    if not newly_sig_coords:
        return out
    br = BitReader(payload)
    vals_all: list[int] = []
    remaining = len(newly_sig_coords)
    while remaining > 0:
        J = min(16, remaining)
        arr = decode_gaggle(br, count=J, N=1, first_gaggle=False)  # N=1 here
        vals_all.extend(list(arr))
        remaining -= J
    for (y, x), v in zip(newly_sig_coords, vals_all):
        out[y, x] = -1 if (v & 1) == 1 else +1
    return out
