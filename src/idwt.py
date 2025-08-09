#Inverse 9/7 Integer DWT

from __future__ import annotations
from src.dwt import _sym_idx, dwt97m_forward_1d
import numpy as np



def dwt97m_inverse_1d(s: np.ndarray, d: np.ndarray, out_dtype=None) -> np.ndarray:
    """
    Inverse 1-D CCSDS 9/7M (integer) DWT.

    Equations (Green Book 120.1‑G‑3, Annex D, Fig. D‑8 / page D‑8):
      s_l^(0) = s_l^(1) + floor( (-1/4)(d_{l-1}^(1)+d_l^(1)) + 1/2 )
      d_l^(0) = d_l^(1) + floor( (-1/16)(s_{l-1}^(0)+s_{l+2}^(0)) + (9/16)(s_l^(0)+s_{l+1}^(0)) + 1/2 )
      x_{2l} = s_l^(0),  x_{2l+1} = d_l^(0). :contentReference[oaicite:1]{index=1}
    """
    s1 = s.astype(np.int64, copy=True)
    d1 = d.astype(np.int64, copy=True)
    ns, nd = len(s1), len(d1)

    # inverse update -> s^(0)
    s0 = s1.copy()
    for l in range(ns):
        dm1 = d1[_sym_idx(l - 1, nd)]
        d0  = d1[_sym_idx(l,     nd)]
        upd = (2 - (dm1 + d0)) // 4   # same rounding as forward
        s0[l] = s1[l] + upd

    # inverse prediction -> d^(0)
    d0_arr = d1.copy()
    for l in range(nd):
        sm1 = s0[_sym_idx(l - 1, ns)]
        s00 = s0[_sym_idx(l,     ns)]
        s01 = s0[_sym_idx(l + 1, ns)]
        s02 = s0[_sym_idx(l + 2, ns)]
        num = -(sm1 + s02) + 9 * (s00 + s01)
        pred = (num + 8) // 16
        d0_arr[l] = d1[l] + pred

    # interleave back
    n = ns + nd
    out = np.empty(n, dtype=np.int64)
    out[0::2] = s0
    out[1::2] = d0_arr
    if out_dtype is None:
        out_dtype = s.dtype
    return out.astype(out_dtype, copy=False)

def dwt97m_inverse_2d(coeffs: np.ndarray, levels: int = 3, out_dtype=None) -> np.ndarray:
    """
    Inverse of dwt97m_forward_2d. Expects subbands packed the same way.
    """
    assert coeffs.ndim == 2
    H, W = coeffs.shape
    out = coeffs.astype(np.int64, copy=True)

    sizes = []
    h, w = H, W
    for _ in range(levels):
        sizes.append((h, w))
        h = (h + 1) // 2
        w = (w + 1) // 2

    for lvl in reversed(range(levels)):
        h, w = sizes[lvl]
        # inverse cols
        for c in range(w):
            ns = (h + 1) // 2
            s_col = out[:ns, c].astype(np.int32, copy=False)
            d_col = out[ns:h, c].astype(np.int32, copy=False)
            out[:h, c] = dwt97m_inverse_1d(s_col, d_col, out_dtype=np.int64)
        # inverse rows
        for r in range(h):
            ns = (w + 1) // 2
            s_row = out[r, :ns].astype(np.int32, copy=False)
            d_row = out[r, ns:w].astype(np.int32, copy=False)
            out[r, :w] = dwt97m_inverse_1d(s_row, d_row, out_dtype=np.int64)

    if out_dtype is None:
        out_dtype = coeffs.dtype
    return out.astype(out_dtype, copy=False)