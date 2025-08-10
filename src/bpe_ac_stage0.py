# ccsds122/bpe_ac_stage0.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .io_segmentation import BitWriter, BitReader
from .bpe_ac_scan import iter_block_families
from .utils import subband_rects

@dataclass
class Stage0PlaneResult:
    """What became significant at this bit-plane (for later stages/sign coding)."""
    plane_b: int
    newly_sig_coords: List[Tuple[int,int]]  # absolute (y,x)
    dc_bits_written: int
    ac_sig_bits_written: int

def _dc_plane_encode(coeffs: np.ndarray, levels: int, b: int, q: int) -> bytes:
    """Write remaining DC plane 'b' if b < q. Raw 1 bit per DC sample (raster)."""
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
    """Inverse of _dc_plane_encode. Returns an array with only that DC plane set."""
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
                        significance_map: np.ndarray) -> Tuple[bytes, Stage0PlaneResult]:
    """
    Stage 0 for a single bit-plane b (MSB→LSB).
    - Writes DC plane if b < q (raw).
    - Writes AC significance for all families & blocks (raw 1 bit per candidate).
    - Updates 'significance_map' in-place: nonzero entries mark significant.
    Returns payload and a report of newly significant coords.
    """
    H, W = coeffs.shape
    assert significance_map.shape == (H, W)
    LL = subband_rects(H, W, levels)[f"LL{levels}"]
    S = LL.h * LL.w                           # number of blocks

    # --- DC plane (if any) ---
    dc_bytes = _dc_plane_encode(coeffs, levels, b, q)
    dc_bits_written = len(dc_bytes) * 8

    # --- AC significance (raw) ---
    bw = BitWriter()
    newly = []

    # Walk blocks in raster
    block = 0
    for bf in iter_block_families(coeffs, levels):
        # BitDepthAC_Block_m gate: only if depth > b does block have candidates this plane
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue
        # families in order F0,F1,F2
        for fam in bf.families:
            for (y, x) in fam.coords:
                # outside image / padding was already filtered in scan
                # Only AC (outside LL)
                if LL.y <= y < LL.y + LL.h and LL.x <= x < LL.x + LL.w:
                    continue  # skip LL area (that’s DC)
                # Not yet significant?
                if significance_map[y, x] == 0:
                    # Candidate at this plane?
                    mag_bit = (abs(int(coeffs[y, x])) >> b) & 1
                    bw.write_bits(mag_bit, 1)     # raw for scaffold
                    if mag_bit == 1:
                        significance_map[y, x] = 1
                        newly.append((y, x))
        block += 1
    ac_payload = bw.to_bytes()
    ac_bits_written = len(ac_payload) * 8

    payload = dc_bytes + ac_payload
    return payload, Stage0PlaneResult(b, newly, dc_bits_written, ac_bits_written)

def decode_stage0_plane(shape: Tuple[int,int],
                        levels: int,
                        bitdepth_ac_per_block: np.ndarray,
                        b: int,
                        q: int,
                        significance_map: np.ndarray,
                        payload: bytes) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Mirror of encode_stage0_plane: consumes bytes and updates significance_map.
    Returns (dc_plane_array, newly_significant_coords).
    """
    H, W = shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]

    # Split payload into DC and AC portions:
    # For scaffold we can't know lengths a priori; compute them the same way encoder did:
    # DC bits: 1 bit per DC sample if b<q else 0.
    dc_bits = (LL.h * LL.w) if (b < q) else 0
    dc_bytes_len = (dc_bits + 7) // 8

    dc_bytes = payload[:dc_bytes_len]
    ac_bytes = payload[dc_bytes_len:]

    # DC plane
    dc_plane = _dc_plane_decode(shape, levels, b, q, dc_bytes)

    # AC significance (raw)
    newly = []
    br = BitReader(ac_bytes)
    block = 0
    for bf in iter_block_families(np.zeros(shape, dtype=np.int32), levels):
        if bitdepth_ac_per_block[block] <= b:
            block += 1
            continue
        for fam in bf.families:
            for (y, x) in fam.coords:
                if LL.y <= y < LL.y + LL.h and LL.x <= x < LL.x + LL.w:
                    continue
                if significance_map[y, x] == 0:
                    bit = br.read_bits(1)
                    if bit == 1:
                        significance_map[y, x] = 1
                        newly.append((y, x))
        block += 1

    return dc_plane, newly
