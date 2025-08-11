
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class BandRect:
    y: int
    x: int
    h: int
    w: int

def subband_rects(H: int, W: int, levels: int) -> Dict[str, BandRect]:
    """
    Returns rects for LL{l}, HL{l}, LH{l}, HH{l} for l=levels..1 in the packed layout
    produced by dwt97m_forward_2d (split rows, then cols per level).
    """
    rects: Dict[str, BandRect] = {}
    h, w = H, W
    for l in range(1, levels+1):
        ll_h, ll_w = (h + 1)//2, (w + 1)//2
        rects[f"LL{l}"] = BandRect(0,       0,      ll_h, ll_w)
        rects[f"HL{l}"] = BandRect(0,       ll_w,   ll_h, w - ll_w)
        rects[f"LH{l}"] = BandRect(ll_h,    0,      h - ll_h, ll_w)
        rects[f"HH{l}"] = BandRect(ll_h,    ll_w,   h - ll_h, w - ll_w)
        h, w = ll_h, ll_w
    return rects
