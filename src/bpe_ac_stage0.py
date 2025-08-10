from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

from src.utils import subband_rects

from .io_segmentation import BitWriter, BitReader
from .bpe_dc import CodeSel
from .bpe_ac_scan import iter_block_families

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


def _read_option_id(br: BitReader, N: int) -> int:
    """Read per-gaggle option ID for pattern size N (Table 4-18)."""
    if N == 2:
        b = br.read_bits(1)
        return 1 if b == 1 else 0
    # N == 3 or 4 → 2 ID bits
    bits = (br.read_bits(1) << 1) | br.read_bits(1)
    if N == 3:
        # 00->0, 01->1, 11->3(uncoded); 10 not used
        return 0 if bits == 0b00 else 1 if bits == 0b01 else 3
    else:
        # N==4: 00->0, 01->1, 10->2, 11->3(uncoded)
        return bits  # 0..3


def encode_stage0_plane(coeffs: np.ndarray,
                        levels: int,
                        bitdepth_ac_per_block: np.ndarray,
                        b: int,
                        q: int,
                        significance_map: np.ndarray,
                        code_sel: CodeSel = "opt") -> Tuple[bytes, Stage0PlaneResult]:
    """
    Stage 0 for one plane b:
      • DC plane bits when b < q (raw, 1 bit per DC sample)
      • AC significance bits per-family, per gaggle (≤16), mapped then VLC-coded (Tables 4-15..4-18).
    Updates significance_map in-place; returns payload and list of newly significant coords.
    """
    H, W = coeffs.shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    # --- DC plane first (if any) ---
    dc_bytes = _dc_plane_encode(coeffs, levels, b, q)
    dc_bits_written = len(dc_bytes) * 8

    # --- Collect AC candidate bits by FAMILY to respect family-dependent alphabets ---
    fam_bits: Dict[int, List[int]] = {0: [], 1: [], 2: []}  # 0=F0,1=F1,2=F2
    newly: List[Tuple[int,int]] = []
    block = 0
    for bf in iter_block_families(coeffs, levels):
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue
        for fam in bf.families:
            fid = fam.family_id
            for (y, x) in fam.coords:
                # skip LL area (DC)
                if LL.y <= y < LL.y+LL.h and LL.x <= x < LL.x+LL.w:
                    continue
                # candidate only if not yet significant
                if significance_map[y, x] == 0:
                    bit = (abs(int(coeffs[y, x])) >> b) & 1
                    fam_bits[fid].append(bit)
                    if bit == 1:
                        significance_map[y, x] = 1
                        newly.append((y, x))
        block += 1

    # --- Map -> symbols & VLC per family sub-stream, gaggle by gaggle (≤16) ---
    bw = BitWriter()
    total_ac_bits = 0

    for fid in (0, 1, 2):
        vals, N = map_stage0_significance(fam_bits[fid], family_id=fid)  # → (symbols, N)
        sizes = split_into_gaggles(len(vals))
        off = 0
        for J in sizes:
            gaggle_syms = vals[off:off+J].tolist()
            if J == 0:
                off += J
                continue

            if N == 1:
                # 1-bit uncoded, no ID per gaggle
                for v in gaggle_syms:
                    bw.write_bits(int(v) & 1, 1)
                total_ac_bits += J
            else:
                # Choose the best option for this gaggle & write ID once (Table 4-18)
                option = best_option_for_symbols(gaggle_syms, N)
                write_id_bits(bw, N, option)
                total_ac_bits += (1 if N == 2 else 2)

                # Emit each codeword with the chosen option (Tables 4-15..4-17)
                for s in gaggle_syms:
                    encode_symbol_with_option(bw, int(s), N, option)
                # track bit count approximately (for debugging); exact count available if needed

            off += J

    ac_payload = bw.to_bytes()
    ac_bits_written = len(ac_payload) * 8  # total bits actually emitted

    return dc_bytes + ac_payload, Stage0PlaneResult(b, newly, dc_bits_written, ac_bits_written)


def decode_stage0_plane(shape: Tuple[int,int],
                        levels: int,
                        bitdepth_ac_per_block: np.ndarray,
                        b: int,
                        q: int,
                        significance_map: np.ndarray,
                        payload: bytes,
                        code_sel: CodeSel = "opt") -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Global-per-family decode to match the encoder:
      - build F0/F1/F2 candidate coord lists across the entire plane
      - read symbols in 16-sized gaggles per family
      - update significance_map and return 'newly' coords (in global scan order)
    """
    H, W = shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    # Split DC vs AC from payload (DC plane is 1 bit per LL sample if b<q)
    LL_area = LL.h * LL.w
    dc_bits = LL_area if (b < q) else 0
    dc_bytes_len = (dc_bits + 7) // 8
    dc_bytes = payload[:dc_bytes_len]
    ac_bytes = payload[dc_bytes_len:]

    # DC plane first
    dc_plane = _dc_plane_decode(shape, levels, b, q, dc_bytes)

    # ---- Build global candidate coord lists per family (same order as encoder) ----
    fam_coords: Dict[int, List[Tuple[int,int]]] = {0: [], 1: [], 2: []}
    block = 0
    for bf in iter_block_families(np.zeros(shape, dtype=np.int32), levels):
        if bitdepth_ac_per_block[block] > b:
            for fam in bf.families:
                fid = fam.family_id
                for (y, x) in fam.coords:
                    # skip LL (DC)
                    if LL.y <= y < LL.y + LL.h and LL.x <= x < LL.x + LL.w:
                        continue
                    if significance_map[y, x] == 0:
                        fam_coords[fid].append((y, x))
        block += 1

    # ---- Read the three family streams exactly in F0, F1, F2 order ----
    br = BitReader(ac_bytes)
    newly: List[Tuple[int,int]] = []

    for fid in (0, 1, 2):
        coords = fam_coords[fid]
        need = len(coords)
        if need == 0:
            continue

        # Ask the mapper which pattern size N applies for this family at Stage 0
        _, N = map_stage0_significance([], family_id=fid)

        # Decode across the entire family list in 16-sized gaggles
        vals_all: List[int] = []
        remaining = need
        idx = 0
        while remaining > 0:
            J = min(16, remaining)
            if N == 1:
                # raw bits, no ID
                vals_all.extend(br.read_bits(1) for _ in range(J))
            else:
                # ID once per gaggle, then J codewords
                option = _read_option_id(br, N)
                for _ in range(J):
                    vals_all.append(decode_symbol_with_option(br, N, option))
            remaining -= J
            idx += J

        # Unmap -> bits and update significance map in the same order
        bits_arr = unmap_stage0_significance(np.array(vals_all, dtype=np.int64), N, family_id=fid)
        for (y, x), bit in zip(coords, bits_arr.tolist()):
            if bit & 1:
                significance_map[y, x] = 1
                newly.append((y, x))

    return dc_plane, newly

