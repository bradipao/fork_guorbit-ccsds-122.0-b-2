from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from src.utils import subband_rects
from .io_segmentation import BitWriter, BitReader
from .bpe_dc import CodeSel
from .bpe_ac_scan import iter_block_families
from .gaggle_vlc import encode_gaggle, decode_gaggle
from .ac_vlc_adapter import split_into_gaggles, map_stage0_significance, unmap_stage0_significance

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
    y,x,h,w = LL.y, LL.x, LL.h, LL.w
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
    y,x,h,w = LL.y, LL.x, LL.h, LL.w
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
    """
    Stage 0 for one plane b:
      - DC plane bits when b < q (raw)
      - AC significance bits in gaggle+VLC framing (currently N=1 mapping)
    Updates significance_map in-place; returns payload and 'newly' list.
    """
    H, W = coeffs.shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    # DC plane
    dc_bytes = _dc_plane_encode(coeffs, levels, b, q)
    dc_bits_written = len(dc_bytes) * 8

    # Collect AC candidate bits in scan order
    ac_bits: List[int] = []
    newly: List[Tuple[int,int]] = []
    block = 0
    for bf in iter_block_families(coeffs, levels):
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue
        for fam in bf.families:
            for (y, x) in fam.coords:
                if LL.y <= y < LL.y+LL.h and LL.x <= x < LL.x+LL.w:
                    continue
                if significance_map[y, x] == 0:
                    bit = (abs(int(coeffs[y, x])) >> b) & 1
                    ac_bits.append(bit)
                    if bit == 1:
                        significance_map[y, x] = 1
                        newly.append((y, x))
        block += 1

    # Map -> ints (N) and gaggle encode
    vals, N = map_stage0_significance(ac_bits)
    sizes = split_into_gaggles(len(vals))
    off = 0
    bw = BitWriter()
    for i, J in enumerate(sizes):
        gag = vals[off:off+J]
        by = encode_gaggle(gag, N=N, first_gaggle=False, code_sel=code_sel)
        # append bytes
        for bch in by:
            bw.buf.append(bch)
        off += J
    ac_payload = bw.to_bytes()
    ac_bits_written = len(ac_payload) * 8

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
    Mirror of encode_stage0_plane. Returns (dc_plane_array, newly_sig_coords).
    """
    H, W = shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    # Split DC vs AC from payload (DC is 1 bit per LL sample if b<q)
    LL_area = LL.h * LL.w
    dc_bits = LL_area if (b < q) else 0
    dc_bytes_len = (dc_bits + 7) // 8
    dc_bytes = payload[:dc_bytes_len]
    ac_bytes = payload[dc_bytes_len:]

    # DC plane
    dc_plane = _dc_plane_decode(shape, levels, b, q, dc_bytes)

    # AC significance stream
    newly: List[Tuple[int,int]] = []
    br = BitReader(ac_bytes)

    block = 0
    for bf in iter_block_families(np.zeros(shape, dtype=np.int32), levels):
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue

        # build candidate coords (same order)
        cand_coords: List[Tuple[int,int]] = []
        for fam in bf.families:
            for (y, x) in fam.coords:
                if LL.y <= y < LL.y+LL.h and LL.x <= x < LL.x+LL.w:
                    continue
                if significance_map[y, x] == 0:
                    cand_coords.append((y, x))

        # read in gaggles of up to 16
        read_vals: List[int] = []
        remaining = len(cand_coords)
        while remaining > 0:
            J = min(16, remaining)
            N = 1  # current mapping
            arr = decode_gaggle(br, count=J, N=N, first_gaggle=False)
            read_vals.extend(list(arr))
            remaining -= J

        for (y, x), v in zip(cand_coords, read_vals):
            if (v & 1) == 1:
                significance_map[y, x] = 1
                newly.append((y, x))

        block += 1

    return dc_plane, newly
