from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

from .io_segmentation import BitWriter, BitReader
from .bpe_dc import CodeSel
from .bpe_ac_scan import iter_block_families

# Spec VLC tables
from .ac_vlc_tables import (
    best_option_for_symbols,
    write_id_bits,
    encode_symbol_with_option,
    decode_symbol_with_option,
)

# Stage-2/3 mappers (family-aware; swap bodies later for spec alphabets)
from .ac_vlc_adapter import (
    split_into_gaggles,
    map_stage2_refinement, unmap_stage2_refinement,
    map_stage3_refinement, unmap_stage3_refinement,
)

def _coord_family_map(shape: Tuple[int,int], levels: int) -> Dict[Tuple[int,int], int]:
    """
    Build a map from (y,x) -> family_id using the block/family scan order.
    Safe to build once per segment; small overhead.
    """
    H, W = shape
    fam_of: Dict[Tuple[int,int], int] = {}
    dummy = np.zeros(shape, dtype=np.int32)
    for bf in iter_block_families(dummy, levels):
        for fam in bf.families:
            fid = fam.family_id
            for (y, x) in fam.coords:
                fam_of[(y, x)] = fid
    return fam_of

def _read_option_id(br: BitReader, N: int) -> int:
    if N == 2:
        b = br.read_bits(1)
        return 1 if b == 1 else 0
    bits = (br.read_bits(1) << 1) | br.read_bits(1)
    if N == 3:
        return 0 if bits == 0b00 else 1 if bits == 0b01 else 3
    else:
        return bits  # 0..3


# -------- Stage 2: refinement for NEWLY significant at this plane --------

def encode_stage2_plane(coeffs: np.ndarray,
                        newly_sig_coords: List[Tuple[int,int]],
                        b: int,
                        code_sel: CodeSel = "opt",
                        *,
                        levels: int | None = None,
                        shape_hint: Tuple[int,int] | None = None) -> bytes:
    if not newly_sig_coords:
        return b""

    # Partition coords by family, preserving order
    if levels is None or shape_hint is None:
        raise ValueError("encode_stage2_plane now requires levels and shape_hint for family partitioning")
    fam_of = _coord_family_map(shape_hint, levels)
    fam_bits: Dict[int, List[int]] = {0: [], 1: [], 2: []}

    for (y, x) in newly_sig_coords:
        fid = fam_of[(y, x)]
        bit = (abs(int(coeffs[y, x])) >> b) & 1
        fam_bits[fid].append(bit)

    bw = BitWriter()
    for fid in (0, 1, 2):
        vals, N = map_stage2_refinement(fam_bits[fid], family_id=fid)
        sizes = split_into_gaggles(len(vals))
        off = 0
        for J in sizes:
            gag = vals[off:off+J].tolist()
            if J == 0:
                off += J
                continue
            if N == 1:
                for v in gag:
                    bw.write_bits(int(v) & 1, 1)
            else:
                option = best_option_for_symbols(gag, N)
                write_id_bits(bw, N, option)
                for s in gag:
                    encode_symbol_with_option(bw, int(s), N, option)
            off += J
    return bw.to_bytes()


def decode_stage2_plane(shape: Tuple[int,int],
                        newly_sig_coords: List[Tuple[int,int]],
                        b: int,
                        payload: bytes,
                        code_sel: CodeSel = "opt",
                        *,
                        levels: int | None = None) -> np.ndarray:
    H, W = shape
    out = np.zeros(shape, dtype=np.int8)
    if not newly_sig_coords:
        return out
    if levels is None:
        raise ValueError("decode_stage2_plane now requires levels to recreate family partition")

    fam_of = _coord_family_map(shape, levels)
    # Partition the coord list by family, preserving order
    fam_coords: Dict[int, List[Tuple[int,int]]] = {0: [], 1: [], 2: []}
    for (y, x) in newly_sig_coords:
        fid = fam_of[(y, x)]
        fam_coords[fid].append((y, x))

    br = BitReader(payload)

    for fid in (0, 1, 2):
        coords = fam_coords[fid]
        need = len(coords)
        if need == 0:
            continue
        # Ask mapper for N for this family/stage
        _, N = map_stage2_refinement([], family_id=fid)

        vals_all: List[int] = []
        remaining = need
        while remaining > 0:
            J = min(16, remaining)
            if N == 1:
                vals_all.extend(br.read_bits(1) for _ in range(J))
            else:
                option = _read_option_id(br, N)
                for _ in range(J):
                    vals_all.append(decode_symbol_with_option(br, N, option))
            remaining -= J

        bits_arr = unmap_stage2_refinement(np.array(vals_all, dtype=np.int64), N, family_id=fid)
        for (y, x), bit in zip(coords, bits_arr.tolist()):
            if bit & 1:
                out[y, x] = 1

    return out


# -------- Stage 3: refinement for PREVIOUSLY significant (earlier planes) --------

def encode_stage3_plane(coeffs: np.ndarray,
                        prev_sig_coords: List[Tuple[int,int]],
                        b: int,
                        code_sel: CodeSel = "opt",
                        *,
                        levels: int | None = None,
                        shape_hint: Tuple[int,int] | None = None) -> bytes:
    if not prev_sig_coords:
        return b""
    if levels is None or shape_hint is None:
        raise ValueError("encode_stage3_plane now requires levels and shape_hint for family partitioning")

    fam_of = _coord_family_map(shape_hint, levels)
    fam_bits: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for (y, x) in prev_sig_coords:
        fid = fam_of[(y, x)]
        bit = (abs(int(coeffs[y, x])) >> b) & 1
        fam_bits[fid].append(bit)

    bw = BitWriter()
    for fid in (0, 1, 2):
        vals, N = map_stage3_refinement(fam_bits[fid], family_id=fid)
        sizes = split_into_gaggles(len(vals))
        off = 0
        for J in sizes:
            gag = vals[off:off+J].tolist()
            if J == 0:
                off += J
                continue
            if N == 1:
                for v in gag:
                    bw.write_bits(int(v) & 1, 1)
            else:
                option = best_option_for_symbols(gag, N)
                write_id_bits(bw, N, option)
                for s in gag:
                    encode_symbol_with_option(bw, int(s), N, option)
            off += J
    return bw.to_bytes()


def decode_stage3_plane(shape: Tuple[int,int],
                        prev_sig_coords: List[Tuple[int,int]],
                        b: int,
                        payload: bytes,
                        code_sel: CodeSel = "opt",
                        *,
                        levels: int | None = None) -> np.ndarray:
    H, W = shape
    out = np.zeros(shape, dtype=np.int8)
    if not prev_sig_coords:
        return out
    if levels is None:
        raise ValueError("decode_stage3_plane now requires levels to recreate family partition")

    fam_of = _coord_family_map(shape, levels)
    fam_coords: Dict[int, List[Tuple[int,int]]] = {0: [], 1: [], 2: []}
    for (y, x) in prev_sig_coords:
        fid = fam_of[(y, x)]
        fam_coords[fid].append((y, x))

    br = BitReader(payload)
    for fid in (0, 1, 2):
        coords = fam_coords[fid]
        need = len(coords)
        if need == 0:
            continue
        _, N = map_stage3_refinement([], family_id=fid)

        vals_all: List[int] = []
        remaining = need
        while remaining > 0:
            J = min(16, remaining)
            if N == 1:
                vals_all.extend(br.read_bits(1) for _ in range(J))
            else:
                option = _read_option_id(br, N)
                for _ in range(J):
                    vals_all.append(decode_symbol_with_option(br, N, option))
            remaining -= J

        bits_arr = unmap_stage3_refinement(np.array(vals_all, dtype=np.int64), N, family_id=fid)
        for (y, x), bit in zip(coords, bits_arr.tolist()):
            if bit & 1:
                out[y, x] = 1

    return out
