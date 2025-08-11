from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from .shared import subband_rects

# Shift table type: (level, subband) -> shift
# subband keys: "LL", "HL", "LH", "HH"
ShiftTable = Dict[Tuple[int, str], int]

# PResets
def preset_none(levels: int) -> ShiftTable:
    # all zeros (useful for A/B testing)
    d: ShiftTable = {}
    for L in range(1, levels + 1):
        for sb in ("HL", "LH", "HH"):
            d[(L, sb)] = 0
    d[(levels, "LL")] = 0
    return d

def preset_ccsds122(levels: int) -> ShiftTable:
    """
    Shifts from CCSDS 122.0-B-2 Table 3-4 for 9/7 integer DWT.
    Γ is the number of bits to shift (multiply by 2^Γ before coding).
    """
    t: ShiftTable = {}
    if levels >= 1:
        t[(1, "HH")] = 0
        t[(1, "HL")] = 1
        t[(1, "LH")] = 1
    if levels >= 2:
        t[(2, "HH")] = 1
        t[(2, "HL")] = 2
        t[(2, "LH")] = 2
    if levels >= 3:
        t[(3, "HH")] = 2
        t[(3, "HL")] = 3
        t[(3, "LH")] = 3
        # LL at the top level uses Γ=3
        t[(levels, "LL")] = 3
    else:
        # If levels<3 and you still want an LL shift, set it to 0 (spec table only lists LL3)
        t[(levels, "LL")] = 0
    return t

def dc_ll_top_shift(levels: int, table: ShiftTable) -> int:
    """Γ for LL at the top decomposition level, used to adjust DC plane q."""
    return table.get((levels, "LL"), 0)
