from typing import List, Tuple, Dict, cast
import numpy as np

from .ac_vlc_tables import best_option_for_symbols, decode_symbol_with_option, write_id_bits, encode_symbol_with_option
from .bpe_ac_scan import iter_block_families
from .ac_word_mapping import symbol_to_word, word_to_symbol, WordKind, infer_N_for_length
from .io_segmentation import BitReader, BitWriter
from .shared import subband_rects
from .utils import _read_option_id


def _candidates_in(coords: List[Tuple[int,int]], sig_map: np.ndarray) -> List[Tuple[int,int]]:
    return [(y, x) for (y, x) in coords if sig_map[y, x] == 0]


def _emit_word(bw: BitWriter, bits: str, kind: WordKind):
    """Per-word emit (ID per word). You can later batch per gaggle+N."""
    if len(bits) <= 1:
        if len(bits) == 1:
            bw.write_bits(1 if bits == "1" else 0, 1)
        return
    sym, N = word_to_symbol(bits, kind)
    option = best_option_for_symbols([int(sym)], N)
    write_id_bits(bw, N, option)
    encode_symbol_with_option(bw, int(sym), N, option)


def encode_stage0_combined(coeffs: np.ndarray,
                           levels: int,
                           b: int,
                           bitdepth_ac_per_block: np.ndarray,
                           sig_map: np.ndarray) -> Tuple[bytes, List[Tuple[int,int]]]:
    H, W = coeffs.shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    bw = BitWriter()
    newly: List[Tuple[int,int]] = []

    def bit_at(y: int, x: int) -> int:
        return (abs(int(coeffs[y, x])) >> b) & 1

    def _emit_tran(kind: WordKind, bits: str):
        """
        Emit a transition word:
          - len==0: nothing
          - len==1: raw bit
          - len>=2: mapped (ID + one codeword)
        """
        L = len(bits)
        if L == 0:
            return
        if L == 1:
            bw.write_bits(1 if bits == "1" else 0, 1)
            return
        sym, N = word_to_symbol(bits, kind)
        opt = best_option_for_symbols([int(sym)], N)
        write_id_bits(bw, N, opt)
        encode_symbol_with_option(bw, int(sym), N, opt)

    def _emit_word_mapped(bits: str, kind: WordKind):
        # 1-bit -> raw; >=2 -> mapped then VLC (ID per word for now)
        L = len(bits)
        if L == 0:
            return
        if L == 1:
            bw.write_bits(1 if bits == "1" else 0, 1)
            return
        sym, N = word_to_symbol(bits, kind)
        opt = best_option_for_symbols([int(sym)], N)
        write_id_bits(bw, N, opt)
        encode_symbol_with_option(bw, int(sym), N, opt)

    block = 0
    for bf in iter_block_families(coeffs, levels):
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue

        # ---- Parents
        parents = [fam.coords[0] for fam in bf.families]
        P = _candidates_in(parents, sig_map)
        if len(P) > 0:
            bitsP = "".join("1" if bit_at(y, x) else "0" for (y, x) in P)
            _emit_word_mapped(bitsP, cast(WordKind, "typesP"))
            for (y, x), ch in zip(P, bitsP):
                if ch == "1":
                    sig_map[y, x] = 1
                    newly.append((y, x))

        # ---- Children + Grandchildren precompute per family
        Ci: Dict[int, List[Tuple[int,int]]] = {}
        Ci_bits: Dict[int, str] = {}
        tmax_C: Dict[int, int] = {}
        groups_cands: Dict[int, List[List[Tuple[int,int]]]] = {}
        groups_bits: Dict[int, List[str]] = {}
        tmax_G: Dict[int, int] = {}
        tmax_Hij: Dict[int, List[int]] = {}

        for i in (0, 1, 2):
            # Children (4)
            Ci[i] = [(y, x) for (y, x) in bf.families[i].coords[1:5] if sig_map[y, x] == 0]
            if len(Ci[i]) > 0:
                bits = "".join("1" if bit_at(y, x) else "0" for (y, x) in Ci[i])
            else:
                bits = ""
            Ci_bits[i] = bits
            tmax_C[i] = (-1 if len(Ci[i]) == 0 else (1 if ("1" in bits) else 0))

            # Grandchildren groups (4 × ≤4)
            G16 = bf.families[i].coords[5:]
            groups = [G16[0:4], G16[4:8], G16[8:12], G16[12:16]]
            g_cands = [[(y, x) for (y, x) in g if sig_map[y, x] == 0] for g in groups]
            groups_cands[i] = g_cands
            grp_bits: List[str] = []
            hij_t: List[int] = []
            for g_c in g_cands:
                if len(g_c) == 0:
                    grp_bits.append("")
                    hij_t.append(-1)
                else:
                    bitsg = "".join("1" if bit_at(y, x) else "0" for (y, x) in g_c)
                    grp_bits.append(bitsg)
                    hij_t.append(1 if ("1" in bitsg) else 0)
            groups_bits[i] = grp_bits
            # Gi type: -1 (no groups), 0 (all zero), 1 (some 1)
            if all(t == -1 for t in hij_t):
                tmax_G[i] = -1
            else:
                tmax_G[i] = 1 if any(t == 1 for t in hij_t) else 0
            tmax_Hij[i] = hij_t

        # D_i = Ci ∪ Gi
        tmax_D = {
            i: (max(tmax_C[i], tmax_G[i]) if not (tmax_C[i] == -1 and tmax_G[i] == -1) else -1)
            for i in (0, 1, 2)
        }

        # B = {D0, D1, D2}
        tvals_D = [tmax_D[i] for i in (0, 1, 2) if tmax_D[i] != -1]
        if len(tvals_D) == 0:
            # no descendants at all this plane
            block += 1
            continue

        # tranB: raw 1 bit (1 if any Di has a '1')
        tmax_B = 1 if any(t == 1 for t in tvals_D) else 0
        bw.write_bits(1 if tmax_B == 1 else 0, 1)
        if tmax_B == 0:
            # all descendants are Type-0 on this plane; no further words
            block += 1
            continue

        # ---- tranD over families that have any descendant candidates (Ci or any Gi group)
        D_indices = [i for i in (0, 1, 2) if (len(Ci[i]) > 0 or any(len(g) > 0 for g in groups_cands[i]))]
        tranD_bits = "".join("1" if (tmax_D[i] == 1) else "0" for i in D_indices)
        _emit_tran(cast(WordKind, "tranD"), tranD_bits)

        # typesCi for families with tranD=1 (and child candidates)
        for i in D_indices:
            if tmax_D[i] == 1 and len(Ci[i]) > 0:
                w = Ci_bits[i]  # len 1..4
                _emit_word_mapped(w, cast(WordKind, "typesCi"))
                for (y, x), ch in zip(Ci[i], w):
                    if ch == "1":
                        sig_map[y, x] = 1
                        newly.append((y, x))

        # ---- tranG over subset with tranD=1 and any grandchild candidates
        G_indices = [i for i in D_indices if (tmax_D[i] == 1 and any(len(g) > 0 for g in groups_cands[i]))]
        tranG_bits = "".join("1" if (tmax_G[i] == 1) else "0" for i in G_indices)
        _emit_tran(cast(WordKind, "tranG"), tranG_bits)

        # For each such family i, only proceed if tranG bit == 1 (i.e., tmax_G[i] == 1)
        for i in G_indices:
            if tmax_G[i] != 1:
                continue  # this family’s grandchildren are all-zero at this plane → no tranHi, no typesHij

            # non-empty groups for this family
            group_idx = [j for j in range(4) if len(groups_cands[i][j]) > 0]

            # flags are tmax(Hij) over those non-empty groups
            bitsHi = "".join("1" if (tmax_Hij[i][j] == 1) else "0" for j in group_idx)

            # emit tranHi (len==0 -> nothing; len==1 -> raw; len>=2 -> mapped)
            _emit_tran(cast(WordKind, "tranHi"), bitsHi)

            # typesHij for groups with flag==1
            for pos, j in enumerate(group_idx):
                if pos < len(bitsHi) and bitsHi[pos] == "1":
                    Hij = groups_cands[i][j]
                    if len(Hij) == 0:
                        continue
                    w = groups_bits[i][j]  # 1..4; guaranteed not 4×"0000" because tmax_Hij[j]==1
                    _emit_word_mapped(w, cast(WordKind, "typesHij"))
                    for (y, x), ch in zip(Hij, w):
                        if ch == "1":
                            sig_map[y, x] = 1
                            newly.append((y, x))


        block += 1

    return bw.to_bytes(), newly



def _apply_types_bits(bits: str,
                      coords: List[Tuple[int,int]],
                      sig_map: np.ndarray,
                      newly_list: List[Tuple[int,int]]) -> None:
    for (y, x), ch in zip(coords, bits):
        if ch == "1":
            sig_map[y, x] = 1
            newly_list.append((y, x))


def _safe_read_bit(br: BitReader) -> int:
    try:
        return br.read_bits(1)
    except EOFError:
        return 0  # pad


def _read_word_with_length(kind: WordKind, length: int, br: BitReader) -> str:
    """
    Read a mapped 'types*' word of a known length.
    For length==1 it's raw; for >=2 read ID + codeword and unmap to bits.
    """
    if length <= 0:
        return ""
    if length == 1:
        return "1" if _safe_read_bit(br) else "0"

    N = infer_N_for_length(kind, length)
    try:
        option = _read_option_id(br, N)
        sym = decode_symbol_with_option(br, N, option)
        return symbol_to_word(int(sym), N, kind)
    except EOFError:
        return "0" * length
