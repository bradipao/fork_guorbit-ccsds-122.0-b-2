from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, cast
import numpy as np

from src.ac_stage0_stream import encode_stage0_combined

from .utils import subband_bitshift_lookup, subband_rects, _read_option_id
from .io_segmentation import BitWriter, BitReader
from .bpe_dc import CodeSel
from .bpe_ac_scan import iter_block_families
from .ac_word_builder import build_stage_words_for_block
from .ac_word_mapping import WordKind, infer_N_for_length, symbol_to_word, word_to_symbol
from .ac_vlc_tables import (
    best_option_for_symbols, write_id_bits,
    encode_symbol_with_option,
)
from .ac_vlc_adapter import split_into_gaggles, map_stage0_significance

# NEW: use the spec VLC tables (Tables 4-15..4-18)
from .ac_vlc_tables import (
    best_option_for_symbols,
    write_id_bits,
    encode_symbol_with_option,
    decode_symbol_with_option,
)

# Stage-0 mapping (family-aware). When you later add the true spec mapping,
# just change these functions; no other code needs to change.
from .ac_vlc_adapter import (
    split_into_gaggles,
    map_stage0_significance,
    unmap_stage0_significance,
)

@dataclass
class Stage0PlaneResult:
    plane_b: int
    newly_sig_coords: List[Tuple[int,int]]
    dc_bits_written: int
    ac_sig_bits_written: int


def _dc_plane_encode(coeffs: np.ndarray, levels: int, b: int, q: int) -> bytes:
    if b >= q:
        return b""
    H, W = coeffs.shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]
    y, x, h, w = LL.y, LL.x, LL.h, LL.w
    dc = coeffs[y:y+h, x:x+w].reshape(-1).astype(np.int64)

    mask = 1 << b
    bw = BitWriter()
    for v in dc:
        bw.write_bits(1 if (int(v) & mask) else 0, 1)
    return bw.to_bytes()


def _dc_plane_decode(shape: Tuple[int,int], levels: int, b: int, q: int, payload: bytes) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int32)
    if b >= q:
        return out

    br = BitReader(payload)
    LL = subband_rects(*shape, levels)[f"LL{levels}"]
    y, x, h, w = LL.y, LL.x, LL.h, LL.w
    dc = np.zeros(h*w, dtype=np.int64)

    bit = 1 << b
    for i in range(h*w):
        if br.read_bits(1):
            dc[i] |= bit
    out[y:y+h, x:x+w] = dc.reshape(h, w).astype(np.int32)
    return out

def encode_stage0_plane(coeffs: np.ndarray,
                        levels: int,
                        bitdepth_ac_per_block: np.ndarray,
                        b: int,
                        q: int,
                        significance_map: np.ndarray,
                        code_sel: CodeSel = "opt") -> Tuple[bytes, Stage0PlaneResult]:

    # --- DC plane first (if any) ---
    dc_bytes = _dc_plane_encode(coeffs, levels, b, q)
    dc_bits_written = len(dc_bytes) * 8

    ac_payload, newly = encode_stage0_combined(coeffs, levels, b, bitdepth_ac_per_block, significance_map)
    ac_bits_written = len(ac_payload) * 8

    return dc_bytes + ac_payload, Stage0PlaneResult(b, newly, dc_bits_written, ac_bits_written)



def decode_stage0_plane(shape: Tuple[int,int],
                        levels: int,
                        bitdepth_ac_per_block: np.ndarray,
                        b: int,
                        q: int,
                        significance_map: np.ndarray,
                        payload: bytes,
                        code_sel: str = "opt") -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Stage 0 (combined stream) for plane b.
      DC plane first (raw 1 bit per LL sample when b < q), then per block:
        • typesP over parent candidates
        • tranB (1 bit). If 0 -> skip descendants. If 1:
            - tranD over families with D_i candidates (|D|=1 -> raw bit; |D|>=2 -> mapped)
            - for families with tranD=1: typesCi with exact child candidate count
            - tranG over the subset with tranD=1 and Gi candidates (|G*|=1 -> raw; else mapped)
            - for each such family:
                · tranHi over its non-empty groups (len 1 -> raw; len>=2 -> mapped; never "0000")
                · for groups with tranHi=1: typesHij with that group's candidate count
    Updates significance_map in-place; returns (dc_plane_array, newly_coords).
    """
    H, W = shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    # ---- Split DC vs AC from payload ----
    LL_area = LL.h * LL.w
    dc_bits = LL_area if (b < q) else 0
    dc_bytes_len = (dc_bits + 7) // 8
    dc_bytes = payload[:dc_bytes_len]
    ac_bytes = payload[dc_bytes_len:]

    # DC plane (existing helper)
    dc_plane = _dc_plane_decode(shape, levels, b, q, dc_bytes)

    br = BitReader(ac_bytes)
    newly: List[Tuple[int,int]] = []

    # ---------------- helpers ----------------
    def _safe_read_bit() -> int:
        try:
            return br.read_bits(1)
        except EOFError:
            return 0  # robust bring-up

    def _apply_types_bits(bits: str, coords: List[Tuple[int,int]]) -> None:
        for (y, x), ch in zip(coords, bits):
            if ch == "1":
                significance_map[y, x] = 1
                newly.append((y, x))

    def _read_types_word(kind: WordKind, length: int) -> str:
        """
        Read one mapped 'types*' word of known length.
        length==0 -> "", length==1 -> raw bit, length>=2 -> mapped word (ID+code).
        Robustness: if the decoded symbol isn't valid for this 'kind' (e.g., typesHij→sym=15),
        fall back to an alternate kind for the same N to reconstruct the bitstring.
        """
        if length <= 0:
            return ""
        if length == 1:
            return "1" if _safe_read_bit() else "0"

        N = infer_N_for_length(kind, length)
        try:
            opt = _read_option_id(br, N)
            sym = decode_symbol_with_option(br, N, opt)
            try:
                bits = symbol_to_word(int(sym), N, kind)
            except (KeyError, ValueError):
                # Fallback mapping family for this N:
                #  - N=3: use 'typesP' (covers 0..7)
                #  - N=4: use 'typesCi' (covers 0..15)
                alt_kind: WordKind = cast(WordKind, ("typesP" if N == 3 else "typesCi"))
                bits = symbol_to_word(int(sym), N, alt_kind)
            # Clamp to expected length, just in case
            if len(bits) > length:
                bits = bits[:length]
            elif len(bits) < length:
                bits = bits + "0" * (length - len(bits))
            return bits
        except EOFError:
            return "0" * length


    def _read_tran_word(kind: WordKind, length: int) -> List[int]:
        """
        Read a transition word and return a list of 0/1 ints.
        length==1 -> raw bit; length>=2 -> mapped.
        Robustness: if the decoded symbol is not valid for `kind` (e.g., tranD symbol=7),
        fall back to the 'other' family mapping to reconstruct the bit pattern.
        """
        if length <= 0:
            return []
        if length == 1:
            return [_safe_read_bit()]

        N = infer_N_for_length(kind, length)
        try:
            opt = _read_option_id(br, N)
            sym = decode_symbol_with_option(br, N, opt)
            try:
                bits = symbol_to_word(int(sym), N, kind)
            except (KeyError, ValueError):
                # Fallback: use a mapping family that covers the full symbol set for this N.
                # For N=3, the "other" column (typesP) covers 0..7; for N=4, use typesCi/ typesHij-family.
                alt_kind: WordKind = cast(WordKind, ("typesP" if N == 3 else "typesCi"))
                bits = symbol_to_word(int(sym), N, alt_kind)
            # Clamp to expected length (defensive; mappings can yield same length already)
            if len(bits) > length:
                bits = bits[:length]
            elif len(bits) < length:
                bits = bits + "0" * (length - len(bits))
            return [1 if c == "1" else 0 for c in bits]
        except EOFError:
            return [0] * length

    # -----------------------------------------

    block = 0
    dummy = np.zeros(shape, dtype=np.int32)  # just for geometry in iter_block_families
    for bf in iter_block_families(dummy, levels):
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue

        # ---- Parents (one per family)
        parents = [fam.coords[0] for fam in bf.families]
        P = [(y, x) for (y, x) in parents if significance_map[y, x] == 0]
        if len(P) > 0:
            bitsP = _read_types_word(cast(WordKind, "typesP"), len(P))
            _apply_types_bits(bitsP, P)

        # ---- Build descendants candidate structure from current sig_map (mirror encoder)
        Ci: Dict[int, List[Tuple[int,int]]] = {}
        Gi_groups: Dict[int, List[List[Tuple[int,int]]]] = {}
        for i in (0, 1, 2):
            Ci[i] = [(y, x) for (y, x) in bf.families[i].coords[1:5] if significance_map[y, x] == 0]
            G16 = bf.families[i].coords[5:]
            groups = [G16[0:4], G16[4:8], G16[8:12], G16[12:16]]
            Gi_groups[i] = [[(y, x) for (y, x) in g if significance_map[y, x] == 0] for g in groups]

        # Families participating in D: any child or any non-empty grandchild group
        D_indices = [i for i in (0, 1, 2) if (len(Ci[i]) > 0 or any(len(g) > 0 for g in Gi_groups[i]))]

        # IMPORTANT: if there are no descendants at all, encoder wrote *no* tranB and nothing more
        if len(D_indices) == 0:
            block += 1
            continue

        # ---- tranB (raw 1 bit): 0 => all descendants Type-0 this plane, skip; 1 => continue
        tranB = _safe_read_bit()
        if tranB == 0:
            block += 1
            continue

        # ---- tranD over D_indices (len 1 -> raw; len>=2 -> mapped)
        tD_bits = _read_tran_word(cast(WordKind, "tranD"), len(D_indices))
        # map to per-family flag
        tD_flag: Dict[int, int] = {i: 0 for i in (0, 1, 2)}
        for pos, i in enumerate(D_indices):
            if pos < len(tD_bits):
                tD_flag[i] = tD_bits[pos]

        # For each family with tD=1, read typesCi with exact child candidate count
        for i in D_indices:
            if tD_flag[i] != 1:
                continue
            Lc = len(Ci[i])
            if Lc > 0:
                bitsCi = _read_types_word(cast(WordKind, "typesCi"), Lc)
                _apply_types_bits(bitsCi, Ci[i])

        # Families eligible for G: those with tD=1 AND any grandchild candidates
        G_indices = [i for i in D_indices if tD_flag[i] == 1 and any(len(g) > 0 for g in Gi_groups[i])]

        # ---- tranG over G_indices
        tG_bits = _read_tran_word(cast(WordKind, "tranG"), len(G_indices))
        tG_flag: Dict[int, int] = {i: 0 for i in (0, 1, 2)}
        for pos, i in enumerate(G_indices):
            if pos < len(tG_bits):
                tG_flag[i] = tG_bits[pos]

        # ---- For each such family with tranG==1, tranHi over its non-empty groups
        for i in G_indices:
            if tG_flag[i] != 1:
                continue
            # non-empty groups and their indices
            group_idx = [j for j in range(4) if len(Gi_groups[i][j]) > 0]

            # tranHi over those (len 1 -> raw; len>=2 -> mapped)
            tHi_bits = _read_tran_word(cast(WordKind, "tranHi"), len(group_idx))

            # For each flagged group, read typesHij with that group's candidate count
            for pos, j in enumerate(group_idx):
                if pos < len(tHi_bits) and tHi_bits[pos] == 1:
                    Hij = Gi_groups[i][j]
                    Lj = len(Hij)
                    if Lj > 0:
                        bitsHij = _read_types_word(cast(WordKind, "typesHij"), Lj)
                        _apply_types_bits(bitsHij, Hij)
                        
        block += 1

    return dc_plane, newly