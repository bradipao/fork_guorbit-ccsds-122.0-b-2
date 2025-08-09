#9/7 Integer DWT

from __future__ import annotations
import numpy as np

# Symmetric (mirror) edge access

def _sym_idx(i: int, n: int) -> int:
    # mirror with duplication at the border (… 1 0 | 0 1 2 …  n-2 n-1 | n-1 n-2 …)
    if i < 0:
        return -i - 1
    if i >= n:
        return 2 * n - 1 - i
    return i

# 1-D forward / inverse 9/7M lifting
def dwt97m_forward_1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward 1-D CCSDS 9/7M (integer) DWT (lifting).
    Returns (s, d) = (low-pass, high-pass) as int arrays, same dtype as x.

    Equations (Green Book 120.1‑G‑3, Annex D, Fig. D‑7):
      s_l^(0) = x_{2l},  d_l^(0) = x_{2l+1}
      d_l^(1) = d_l^(0) - floor( (-1/16)(s_{l-1}^(0)+s_{l+2}^(0)) + (9/16)(s_l^(0)+s_{l+1}^(0)) + 1/2 )
      s_l^(1) = s_l^(0) - floor( (-1/4) (d_{l-1}^(1)+d_{l}^(1)) + 1/2 )
    with symmetric-copy boundary handling. :contentReference[oaicite:0]{index=0}
    """
    assert x.ndim == 1
    n = x.shape[0]
    # work in int64 but return int32 coeffs
    s = x[0::2].astype(np.int64, copy=True)
    d = x[1::2].astype(np.int64, copy=True)
    ns, nd = len(s), len(d)

    # prediction
    d1 = d.copy()
    for l in range(nd):
        sm1 = s[_sym_idx(l - 1, ns)]
        s0  = s[_sym_idx(l,     ns)]
        s1  = s[_sym_idx(l + 1, ns)]
        s2  = s[_sym_idx(l + 2, ns)]
        num = -(sm1 + s2) + 9 * (s0 + s1)
        pred = (num + 8) // 16
        d1[l] = d[l] - pred

    # update
    s1a = s.copy()
    for l in range(ns):
        dm1 = d1[_sym_idx(l - 1, nd)]
        d0  = d1[_sym_idx(l,     nd)]
        upd = (2 - (dm1 + d0)) // 4
        s1a[l] = s[l] - upd

    # return signed coeffs with enough headroom
    return s1a.astype(np.int32, copy=False), d1.astype(np.int32, copy=False)

# 2-D (rows then cols), 3-level DWT
def dwt97m_forward_2d(img: np.ndarray, levels: int = 3) -> np.ndarray:
    """
    In-place-ish 2-D forward DWT with row-then-column lifting per level.
    Returns an array with subbands packed in-place (LL in top-left).
    """
    assert img.ndim == 2
    H, W = img.shape
    # work buffer (signed)
    out = img.astype(np.int64, copy=True)

    h, w = H, W
    for _ in range(levels):
        # rows
        for r in range(h):
            s, d = dwt97m_forward_1d(out[r, :w].astype(np.int32, copy=False))
            out[r, :w] = np.concatenate([s, d], axis=0)
        # cols
        for c in range(w):
            s, d = dwt97m_forward_1d(out[:h, c].astype(np.int32, copy=False))
            out[:h, c] = np.concatenate([s, d], axis=0)
        h = (h + 1) // 2
        w = (w + 1) // 2

    # return signed coeffs (int32)
    return out.astype(np.int32, copy=False)