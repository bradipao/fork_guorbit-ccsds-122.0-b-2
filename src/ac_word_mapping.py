from __future__ import annotations
from typing import Literal, Tuple

# Word kinds we will encounter in stages 1–3 (4.5.3.1)
WordKind = Literal[
    "typesP",      # typesb[P]     (parents)        -> 2 or 3 bits
    "typesCi",     # typesb[Ci]    (children)       -> up to 4 bits
    "typesHij",    # typesb[Hij]   (grandchildren)  -> up to 4 bits (0000 impossible)
    "tranD",       # tranD         -> 3 bits (000 impossible)
    "tranG",       # tranG         -> 3 bits
    "tranHi",      # tranHi        -> 4 bits (0000 impossible)
]

def _ensure_bits(s: str, n: int) -> str:
    s = s.strip()
    if len(s) != n or any(c not in "01" for c in s):
        raise ValueError(f"need {n} bits, got '{s}'")
    return s

# ---------- Tables 4-12..4-14 (bit-pattern -> symbol) ----------

# Table 4-12: 2-bit words (all kinds that are length==2)
_T12 = {"00": 0, "01": 2, "10": 1, "11": 3}  # CCSDS 122.0-B-2, Table 4-12

# Table 4-13: 3-bit words; left column for typesP/typesCi/typesHij/tranG/tranHi;
# right column for tranD (special)
_T13_other = {  # (typesb[P], typesb[Ci], typesb[Hij], tranG, tranHi)
    "000": 1, "001": 4, "010": 0, "011": 5, "100": 2, "101": 6, "110": 3, "111": 7
}
_T13_tranD = {  # (tranD)
    "001": 3, "010": 0, "011": 4, "100": 1, "101": 5, "110": 2, "111": 6
    # "000" impossible per Table 4-11
}

# Table 4-14: 4-bit words; two columns
_T14_typesCi = {  # Symbol (typesb[Ci])
    "0000": 10, "0001": 1, "0010": 3, "0011": 6,
    "0100": 2,  "0101": 5, "0110": 9, "0111": 12,
    "1000": 0,  "1001": 8, "1010": 7, "1011": 13,
    "1100": 4,  "1101": 14, "1110": 11, "1111": 15
}
_T14_typesHij_tranHi = {  # Symbol (typesb[Hij], tranHi)
    # "0000" impossible per Table 4-11
    "0001": 1,  "0010": 3,  "0011": 6,
    "0100": 2,  "0101": 5,  "0110": 9,  "0111": 11,
    "1000": 0,  "1001": 8,  "1010": 7,  "1011": 12,
    "1100": 4,  "1101": 13, "1110": 10, "1111": 14
}

# Also build inverse maps (symbol -> bit-pattern), used by decoder.
_INV_T12 = {v: k for k, v in _T12.items()}
_INV_T13_other = {v: k for k, v in _T13_other.items()}
_INV_T13_tranD  = {v: k for k, v in _T13_tranD.items()}
_INV_T14_typesCi = {v: k for k, v in _T14_typesCi.items()}
_INV_T14_typesHij_tranHi = {v: k for k, v in _T14_typesHij_tranHi.items()}

def word_to_symbol(bits: str, kind: WordKind) -> Tuple[int, int]:
    """
    Map a variable-length word (2/3/4 bits) to an integer symbol per Tables 4-12..4-14.
    Returns (symbol, N) where N is the 'mapped pattern size' (2,3, or 4).
    """
    n = len(bits)
    if n == 2:
        return _T12[_ensure_bits(bits, 2)], 2

    if n == 3:
        b = _ensure_bits(bits, 3)
        if kind == "tranD":
            if b == "000":
                raise ValueError("tranD cannot be 000 (Table 4-11)")
            return _T13_tranD[b], 3
        # other 3-bit words
        return _T13_other[b], 3

    if n == 4:
        b = _ensure_bits(bits, 4)
        if kind in ("typesHij", "tranHi"):
            if b == "0000":
                raise ValueError(f"{kind} cannot be 0000 (Table 4-11)")
            return _T14_typesHij_tranHi[b], 4
        # typesCi 4-bit
        return _T14_typesCi[b], 4

    raise ValueError("word length must be 2, 3, or 4")

def symbol_to_word(symbol: int, N: int, kind: WordKind) -> str:
    """
    Inverse of word_to_symbol: map a symbol (and N and kind) back to the original bit word.
    """
    if N == 2:
        return _INV_T12[symbol]
    if N == 3:
        if kind == "tranD":
            return _INV_T13_tranD[symbol]
        return _INV_T13_other[symbol]
    if N == 4:
        if kind in ("typesHij", "tranHi"):
            return _INV_T14_typesHij_tranHi[symbol]
        return _INV_T14_typesCi[symbol]
    raise ValueError("N must be 2, 3, or 4")
