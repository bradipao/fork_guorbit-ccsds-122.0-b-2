# ccsds122/bpe_ac_depth.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Literal
from .shared import subband_rects
from .io_segmentation import BitWriter, BitReader
from .bpe_dc import _code_id_bits_len, _select_k_opt, _select_k_heur, CodeSel  # reuse DC helpers

@dataclass
class ACDepthInfo:
    BitDepthAC: int
    N: int
    S: int
    gaggle_sizes: List[int]
    opt_method: CodeSel

def _block_coords_LL3(H:int, W:int, levels:int) -> Tuple[np.ndarray, int, int]:
    """Return raster order DC coords in LL(top)."""
    LL = subband_rects(H, W, levels)[f"LL{levels}"]
    dc_r = np.arange(LL.h, dtype=np.int32)
    dc_c = np.arange(LL.w, dtype=np.int32)
    R, C = np.meshgrid(dc_r, dc_c, indexing="ij")
    coords = np.stack([R.reshape(-1), C.reshape(-1)], axis=1)
    return coords, LL.h, LL.w

def _gaggle_sizes_ac(S: int) -> List[int]:
    # first gaggle carries reference then 15 deltas; subsequent 16 deltas
    if S == 0:
        return []
    if S == 1:
        return [0]
    rem = S - 1
    out = [min(15, rem)]
    rem -= out[0]
    while rem > 0:
        t = min(16, rem)
        out.append(t)
        rem -= t
    return out

def _ac_magnitude_bitdepth(v: np.ndarray) -> int:
    amax = int(np.max(np.abs(v), initial=0))
    return int(np.ceil(np.log2(1 + amax))) if amax > 0 else 0

def _gather_block_ac(coeffs: np.ndarray, levels: int) -> Tuple[np.ndarray, int]:
    """
    Compute BitDepthAC_Block_m for all blocks in raster order.
    Returns (depths[S], BitDepthAC=max(depths)).
    """
    H, W = coeffs.shape
    coords, Hll, Wll = _block_coords_LL3(H, W, levels)
    # Subband rects for mapping (parents/children/grandchildren)
    rects = subband_rects(H, W, levels)
    # Access helpers:
    LL = rects[f"LL{levels}"]
    # Families/subbands per Table 4-1/4-2:
    # i=0: parents from HL3, children from HL2, grandchildren from (HL1/LH1/HH1 groups)
    def subview(name): 
        R = rects[name]
        return coeffs[R.y:R.y+R.h, R.x:R.x+R.w]

    # Parents p0..p2 live in {HL3, LH3, HH3}
    P = [subview("HL3"), subview("LH3"), subview("HH3")]
    # Children Ci live in {HL2, LH2, HH2}
    C = [subview("HL2"), subview("LH2"), subview("HH2")]
    # Grandchildren groups Hij live in {HL1, LH1, HH1} in 2x2 tiles around 4*(r,c)
    G = [subview("HL1"), subview("LH1"), subview("HH1")]

    depths = np.zeros(coords.shape[0], dtype=np.int32)
    for m, (r, c) in enumerate(coords):
        mags = []
        # parents (one per family i)
        # Parent coord for family i is (r,c) within P[i]
        for i in range(3):
            if 0 <= r < P[i].shape[0] and 0 <= c < P[i].shape[1]:
                mags.append(P[i][r, c])
        # children Ci: coords (2r+dr, 2c+dc) in C[i], dr,dc in {0,1}
        rr = 2*r; cc = 2*c
        for i in range(3):
            Hc, Wc = C[i].shape
            for dr in (0,1):
                for dc in (0,1):
                    ry = rr + dr; cx = cc + dc
                    if 0 <= ry < Hc and 0 <= cx < Wc:
                        mags.append(C[i][ry, cx])
        # grandchildren Hij: coords (4r+yr, 4c+yc) groups j=0..3, each 2x2 in G[i]
        rr2 = 4*r; cc2 = 4*c
        for i in range(3):
            Hg, Wg = G[i].shape
            for j in range(4):
                yoff = (j//2)*2; xoff = (j%2)*2
                for dr in (0,1):
                    for dc in (0,1):
                        ry = rr2 + yoff + dr; cx = cc2 + xoff + dc
                        if 0 <= ry < Hg and 0 <= cx < Wg:
                            mags.append(G[i][ry, cx])
        mags = np.array(mags, dtype=np.int64)
        depths[m] = _ac_magnitude_bitdepth(mags)
    return depths, int(depths.max(initial=0))

def encode_ac_bitdepths(coeffs: np.ndarray, levels: int, *, code_sel: CodeSel="opt") -> tuple[bytes, ACDepthInfo, np.ndarray]:
    """
    Implements §4.4. Returns (payload, info, depths_per_block[S]).
    """
    H, W = coeffs.shape
    depths, BitDepthAC = _gather_block_ac(coeffs, levels)
    S = depths.size

    if BitDepthAC == 0:
        return b"", ACDepthInfo(BitDepthAC, 0, S, [], code_sel), depths
    if BitDepthAC == 1:
        bw = BitWriter()
        for d in depths:
            bw.write_bits(int(d), 1)
        return bw.to_bytes(), ACDepthInfo(BitDepthAC, 1, S, [S], code_sel), depths

    # BitDepthAC >= 2
    N = int(np.ceil(np.log2(1 + BitDepthAC)))  # eq. 21
    xmin, xmax = 0, (1 << N) - 1

    # Differencing + mapping over nonnegative numbers
    sizes = _gaggle_sizes_ac(S)
    bw = BitWriter()
    idx = 0
    for g, J in enumerate(sizes):
        first = (g == 0)
        if first:
            ref = int(depths[0]); idx = 1
        deltas = np.empty(J, dtype=np.int64)
        for j in range(J):
            m = idx + j
            prev = int(depths[m-1])
            curr = int(depths[m])
            theta = min(prev - 1 - xmin, xmax - prev)
            dprim = curr - prev
            if 0 <= dprim <= theta:
                dm = 2 * dprim
            elif -theta <= dprim < 0:
                dm = 2 * abs(dprim) - 1
            else:
                dm = theta + abs(dprim)
            deltas[j] = dm

        # select k (opt or heuristic), same option ID table
        if N == 1:
            k = None
        else:
            if code_sel == "opt":
                k, _ = _select_k_opt(deltas.astype(np.int64), N, first)
            else:
                k = _select_k_heur(deltas.astype(np.int64), N, J)

        id_bits, _ = _code_id_bits_len(N, k)
        bw.write_bits(int(id_bits, 2), len(id_bits))
        if first:
            bw.write_bits(ref & ((1 << N) - 1), N)

        if k is None:
            for dm in deltas:
                bw.write_bits(int(dm) & ((1 << N) - 1), N)
        else:
            # unary first parts
            for dm in deltas:
                z = int(dm) >> k
                if z > 0:
                    bw.write_bits(0, z)
                bw.write_bits(1, 1)
            # k LSBs
            if k > 0:
                for dm in deltas:
                    bw.write_bits(int(dm) & ((1 << k) - 1), k)

        idx += J

    return bw.to_bytes(), ACDepthInfo(BitDepthAC, N, S, sizes, code_sel), depths

def decode_ac_bitdepths(shape: tuple[int,int], levels: int, payload: bytes, info: ACDepthInfo) -> np.ndarray:
    """
    Inverse of encode_ac_bitdepths. Returns depths[S] in raster order.
    """
    H, W = shape
    coords, Hll, Wll = _block_coords_LL3(H, W, levels)
    S = coords.shape[0]
    if info.BitDepthAC == 0:
        return np.zeros(S, dtype=np.int32)
    if info.BitDepthAC == 1:
        br = BitReader(payload)
        return np.fromiter((br.read_bits(1) for _ in range(S)), dtype=np.int32, count=S)

    N = info.N
    xmin, xmax = 0, (1 << N) - 1
    br = BitReader(payload)
    depths = np.zeros(S, dtype=np.int64)

    def read_id() -> int | None:
        if N == 2:
            return None if br.read_bits(1) == 1 else 0
        elif 2 < N <= 4:
            b = br.read_bits(2)
            return None if b == 0b11 else (0 if b == 0b00 else 1 if b == 0b01 else 2)
        elif 4 < N <= 8:
            b = br.read_bits(3)
            return None if b == 0b111 else b
        else:  # 8 < N <= 10 (but here N<=5 per note, still OK)
            b = br.read_bits(4)
            return None if b == 0b1111 else b

    sizes = info.gaggle_sizes
    idx = 0
    for g, J in enumerate(sizes):
        first = (g == 0)
        k = read_id()
        if first:
            ref = br.read_bits(N)
            depths[0] = ref
            idx = 1

        if k is None:
            deltas = np.fromiter((br.read_bits(N) for _ in range(J)), dtype=np.int64, count=J)
        else:
            z = []
            for _ in range(J):
                cnt = 0
                while True:
                    if br.read_bits(1) == 1: break
                    cnt += 1
                z.append(cnt)
            lsbs = [br.read_bits(k) if k > 0 else 0 for _ in range(J)]
            deltas = np.array([(z[i] << k) | lsbs[i] for i in range(J)], dtype=np.int64)

        for j in range(J):
            m = idx + j
            prev = int(depths[m-1])
            theta = min(prev - 1 - xmin, xmax - prev)
            dm = int(deltas[j])
            if dm <= 2*theta:
                dprim = dm // 2 if (dm & 1) == 0 else -((dm + 1)//2)
            else:
                mag = dm - theta
                # choose direction that keeps within [xmin,xmax]
                if prev + mag > xmax:
                    dprim = -mag
                elif prev - mag < xmin:
                    dprim = +mag
                else:
                    dprim = +mag if (xmax - prev) >= (prev - 1 - xmin) else -mag
            depths[m] = prev + dprim
        idx += J

    return depths.astype(np.int32)
