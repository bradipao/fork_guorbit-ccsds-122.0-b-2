#Subband Indexing Helper
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

@dataclass(frozen=True)
class BandRect:
    y: int; x: int; h: int; w: int  # top-left + size in the packed coeffs

def subband_rects(H: int, W: int, levels: int) -> Dict[str, BandRect]:
    """
    Returns rects for LL{l}, HL{l}, LH{l}, HH{l} for l=levels..1 in the packed layout
    produced by dwt97m_forward_2d (split rows, then cols per level).
    """
    rects: Dict[str, BandRect] = {}
    h, w = H, W
    for l in range(1, levels+1):
        # current LL size at start of level l
        ll_h, ll_w = (h + 1)//2, (w + 1)//2

        # Rects inside the top-left (h,w) window:
        # [ s_row | d_row ] then per-column [ s_col ; d_col ]
        # s_row region => width = ll_w, d_row region => width = w - ll_w
        # after column split, s_col => height = ll_h, d_col => height = h - ll_h
        # LL_l  = s_col(s_row)  => (0,0,ll_h,ll_w)
        # HL_l  = s_col(d_row)  => (0,ll_w,ll_h,w-ll_w)
        # LH_l  = d_col(s_row)  => (ll_h,0,h-ll_h,ll_w)
        # HH_l  = d_col(d_row)  => (ll_h,ll_w,h-ll_h,w-ll_w)
        rects[f"LL{l}"] = BandRect(0,       0,      ll_h, ll_w)
        rects[f"HL{l}"] = BandRect(0,       ll_w,   ll_h, w - ll_w)
        rects[f"LH{l}"] = BandRect(ll_h,    0,      h - ll_h, ll_w)
        rects[f"HH{l}"] = BandRect(ll_h,    ll_w,   h - ll_h, w - ll_w)

        # next level operates on the LL area only
        h, w = ll_h, ll_w

    return rects

def apply_subband_weights(coeffs: np.ndarray, levels: int,
                          num: dict[str,int] | None=None,
                          den: dict[str,int] | None=None) -> np.ndarray:
    """
    Multiply each subband by num[name]/den[name].
    Defaults to identity if maps are None (or missing keys).
    """
    out = coeffs.copy()
    rects = subband_rects(coeffs.shape[0], coeffs.shape[1], levels)
    for name, rect in rects.items():
        y, x, h, w = rect.y, rect.x, rect.h, rect.w
        n = 1 if num is None else num.get(name, 1)
        d = 1 if den is None else den.get(name, 1)
        if n == 1 and d == 1:
            continue
        # exact integer scaling when possible
        if d == 1:
            out[y:y+h, x:x+w] = out[y:y+h, x:x+w] * n
        else:
            # keep integer arith: round toward zero (spec uses exact rules; we’ll adjust later)
            out[y:y+h, x:x+w] = (out[y:y+h, x:x+w] * n) // d
    return out

def undo_subband_weights(coeffs_w: np.ndarray, levels: int,
                         num: dict[str,int] | None=None,
                         den: dict[str,int] | None=None) -> np.ndarray:
    """
    Inverse of apply_subband_weights: multiply by den/name over num/name.
    """
    return apply_subband_weights(coeffs_w, levels, num=den, den=num)