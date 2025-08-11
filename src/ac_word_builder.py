from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from .bpe_ac_scan import iter_block_families
from .utils import tb_of, bit_b_of, tmax, subband_bitshift_lookup
from .shared import subband_rects

# Each 'word' is (bits:str, kind:str, N:int). Kinds per Table 4-11 column names.
Word = Tuple[str, str, int, int]

def _types_word_for(coords: List[Tuple[int,int]],
                    coeffs: np.ndarray, b: int, bs_lookup) -> str:
    """typesb[Ψ]: b-th bit for members whose tb ∈ {0,1}, in scan order (§4.5.3.1.4/.1.6)."""
    bits = []
    for (y, x) in coords:
        tb = tb_of(int(coeffs[y, x]), b, bs_lookup(y, x))
        if tb in (0, 1):
            bits.append(str(bit_b_of(int(coeffs[y, x]), b)))
    return "".join(bits)  # can be "" (null)

def _tword_from_type_values(type_values: List[int]) -> str:
    """tword[...] includes entries that are 0 or 1 only (§4.5.3.1.4)."""
    out = []
    for t in type_values:
        if t in (0, 1):
            out.append("1" if t == 1 else "0")
    return "".join(out)

def _Nlen(s: str) -> int:
    return len(s) if s else 0

def build_stage_words_for_block(coeffs: np.ndarray,
                                levels: int,
                                b: int,
                                block_families,
                                bs_lookup) -> List[Word]:
    """
    Returns the Stage-1/2/3 'words' for ONE block, in exact order (§4.5.3.1.8):
      Stage 1: typesb[P], signsb[P] (signs not VLC-coded)
      Stage 2: tranB; tranD (if needed); typesb[Ci] & signsb[Ci] where required
      Stage 3: if needed: tranG; tranHi (per i); typesb[Hij] & signsb[Hij] where required
    Only non-null words with length >=2 are VLC-coded (Tables 4-11..4-14); signs are raw.
    """
    words: List[Word] = []

    # Build P, Ci, Gi, Hij lists of coords for this block
    F = block_families.families  # tuple(F0, F1, F2)
    # Parents per spec P={p0,p1,p2}: one parent from each family (coords[0] of each family)
    P_coords = [F[0].coords[0], F[1].coords[0], F[2].coords[0]]

    # For each family i: children Ci are the next 4 coords; grandchildren are the rest 16, grouped 4x4
    Ci_coords: Dict[int, List[Tuple[int,int]]] = {0: F[0].coords[1:5], 1: F[1].coords[1:5], 2: F[2].coords[1:5]}
    Hij_coords: Dict[int, List[List[Tuple[int,int]]]] = {}
    for i in (0,1,2):
        g = F[i].coords[5:]      # 16
        Hij_coords[i] = [ g[0:4], g[4:8], g[8:12], g[12:16] ]  # Hi0..Hi3

    # Compute tb for everything we may need 
    tbP  = [tb_of(int(coeffs[y,x]), b, bs_lookup(y,x)) for (y,x) in P_coords]
    tbCi = {i: [tb_of(int(coeffs[y,x]), b, bs_lookup(y,x)) for (y,x) in Ci_coords[i]] for i in (0,1,2)}
    tbHij = {i: [[tb_of(int(coeffs[y,x]), b, bs_lookup(y,x)) for (y,x) in Hij_coords[i][j]] for j in range(4)] for i in (0,1,2)}

    # Di = {Ci, Gi}; Gi is union of all Hij groups
    tmax_D = {}
    tmax_G = {}
    for i in (0,1,2):
        tmax_G[i] = tmax(v for grp in tbHij[i] for v in grp)
        tmax_D[i] = max(tmax(tbCi[i]), tmax_G[i])

    # B = {D0, D1, D2}
    tmax_B = tmax(tmax_D[i] for i in (0,1,2))

    # Stage 1: parents 
    typesP = _types_word_for(P_coords, coeffs, b, bs_lookup)
    if _Nlen(typesP) >= 2:
        # 2 or 3 bits max per Table 4-11, kind "typesP"
        words.append((typesP, "typesP", len(typesP), -1))
    # signsb[P] are raw bits 

    # Stage 2: children
    # 1) tranB
    # Definition (§4.5.3.1.7): tranB = null if it was 1 at any more significant plane;
    # else tword[{tmax(B)}]. Here we don't track history yet; at a single plane it is one bit if tmax(B) in {0,1}, else null.
    tranB = _tword_from_type_values([tmax_B])   # -> "" if tmax_B in {-1,2}, "0" or "1" otherwise
    if _Nlen(tranB) == 1:
        words.append((tranB, "tranB", 1, -1))   # note: never entropy-coded (max 1 bit), per §4.5.3.1.9

    # 2) tranD, if tranB ≠ 0 and tmax(B) ≠ -1
    # (When tranB == "0" it signals all descendants are Type 0, so no deeper coding.)
    if (tranB != "0") and (tmax_B != -1):
        # tword of the three Di that haven't been permanently 1 at earlier planes.
        # At a single plane, use all three families’ tmax(Di) values.
        tranD_bits = _tword_from_type_values([tmax_D[i] for i in (0,1,2)])
        # Important: Table 4-11 forbids "000" for tranD (implied by tranB==0) — we gate above.
        if _Nlen(tranD_bits) >= 2:  # 2 or 3 → entropy-coded
            words.append((tranD_bits, "tranD", len(tranD_bits), -1))

        # 3) typesb[Ci] (+ signs) *only* for families with tmax(Di) > 0 now or earlier
        for i in (0,1,2):
            if tmax_D[i] > 0:
                w = _types_word_for(Ci_coords[i], coeffs, b, bs_lookup)
                if _Nlen(w) >= 2:
                    words.append((w, "typesCi", len(w), i))
                # signsb[Ci] are raw; emit elsewhere from same coords when tb==1

    # Stage 3: grandchildren
    # If tranB == 0 or tmax(B) == -1 → omit Stage 3 entirely
    if not (tranB == "0" or tmax_B == -1):
        # 1) tranG = tword[{tmax(Gi) : i with tmax(Di) > 0 now or earlier}]
        tvals_G = [tmax_G[i] for i in (0,1,2) if tmax_D[i] > 0]
        tranG_bits = _tword_from_type_values(tvals_G)
        if _Nlen(tranG_bits) >= 2:
            words.append((tranG_bits, "tranG", len(tranG_bits), -1))

        # 2) For each i with tmax(Gi) != 0, -1: tranHi for i
        for i in (0,1,2):
            if tmax_G[i] not in (0, -1):
                tvals_Hi = [ tmax(grp) for grp in tbHij[i] ]  # four groups
                tranHi_bits = _tword_from_type_values(tvals_Hi)
                if _Nlen(tranHi_bits) >= 2:
                    words.append((tranHi_bits, "tranHi", len(tranHi_bits), i))

                # 3) For each j with tmax(Hij) ≠ 0, -1: typesb[Hij]
                for j in range(4):
                    if tmax(tbHij[i][j]) not in (0, -1):
                        w = _types_word_for(Hij_coords[i][j], coeffs, b, bs_lookup)
                        if _Nlen(w) >= 2:
                            words.append((w, "typesHij", len(w), i))
                        # signsb[Hij] are raw when tb==1
    return words
