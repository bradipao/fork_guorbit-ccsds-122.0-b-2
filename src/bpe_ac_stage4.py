from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .io_segmentation import BitWriter, BitReader
from .bpe_dc import CodeSel
from .gaggle_vlc import encode_gaggle, decode_gaggle
from .ac_vlc_adapter import split_into_gaggles

def _safe_read_bits(br, n: int) -> np.ndarray:
    out = []
    for _ in range(n):
        try:
            out.append(br.read_bits(1))
        except EOFError:
            out.append(0)  # pad missing bits with 0
    return np.asarray(out, dtype=np.int64)


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

def decode_stage4_plane(shape, newly_sig_coords, payload):

    H, W = shape
    out = np.zeros(shape, dtype=np.int8)

    br = BitReader(payload)
    if len(payload) == 0 or len(newly_sig_coords) == 0:
        return out

    # decode in gaggles of 16, but use _safe_read_bits to avoid EOF
    idx = 0
    total = len(newly_sig_coords)
    while idx < total:
        J = min(16, total - idx)
        # N=1 here (raw signs)
        arr = _safe_read_bits(br, J)
        # apply signs (arr[k] is the sign bit for coord idx+k)
        for k in range(J):
            y, x = newly_sig_coords[idx + k]
            out[y, x] = 1 if arr[k] else 0
        idx += J

    return out