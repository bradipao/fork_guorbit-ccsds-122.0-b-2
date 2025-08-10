from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
import numpy as np
from .io_segmentation import BitWriter, BitReader
from .utils import subband_rects

CodeSel = Literal["opt", "heur"]

@dataclass
class DCEncodeInfo:
    BitDepthDC: int
    BitDepthAC: int
    q: int
    N: int
    S: int                 
    gaggle_sizes: List[int] 


def _bitdepth_dc(dc_vals: np.ndarray) -> int:
    # two's complement width (eq. 12, §4.2.2.2.4)
    if dc_vals.size == 0:
        return 1
    neg = dc_vals < 0
    pos = ~neg
    a = np.zeros_like(dc_vals, dtype=np.int64)
    a[neg] = 1 + np.ceil(np.log2(np.abs(dc_vals[neg]).astype(np.int64))).astype(np.int64)
    a[pos] = 1 + np.ceil(np.log2(1 + dc_vals[pos].astype(np.int64))).astype(np.int64)
    a[a < 1] = 1
    return int(a.max(initial=1))

def _bitdepth_ac(all_coeffs: np.ndarray, ll_rect, levels: int) -> int:
    # BitDepthAC is max over blocks of BitDepthAC_Blockm; this equals the max magnitude
    # over *all AC coefficients* in the segment (since it's a max-of-maxes). (§4.2.2.2.5)
    mask = np.ones(all_coeffs.shape, dtype=bool)
    # zero-out LL_top area so it's not counted as AC
    y,x,h,w = ll_rect.y, ll_rect.x, ll_rect.h, ll_rect.w
    mask[y:y+h, x:x+w] = False
    if not mask.any():
        return 0
    ac_max = int(np.max(np.abs(all_coeffs[mask]))) if mask.any() else 0
    return int(np.ceil(np.log2(1 + ac_max))) if ac_max > 0 else 0

def _table_4_8_qprime(BitDepthDC: int, BitDepthAC: int) -> int:
    # if BitDepthDC <= 3: q' = 0
    # elif BitDepthDC - (1+floor(BitDepthAC/2)) <= 1: q' = BitDepthDC - 3
    # elif BitDepthDC - (1+floor(BitDepthAC/2)) > 10: q' = BitDepthDC - 10
    # else: q' = 1 + floor(BitDepthAC/2)
    if BitDepthDC <= 3:
        return 0
    t = BitDepthDC - (1 + (BitDepthAC // 2))
    if t <= 1:
        return BitDepthDC - 3
    if t > 10:
        return BitDepthDC - 10
    return 1 + (BitDepthAC // 2)

def _gaggle_sizes(S: int) -> List[int]:
    # first gaggle: 15 mapped deltas (because c'0 sent as reference), subsequent: 16, last may be shorter (§4.3.2.5)
    if S == 0:
        return []
    if S == 1:
        return [0]  # only reference sample
    rem = S - 1
    sizes = [min(15, rem)]
    rem -= sizes[0]
    while rem > 0:
        take = min(16, rem)
        sizes.append(take)
        rem -= take
    return sizes

def _code_id_bits_len(N: int, k: int | None) -> Tuple[str,int]:
    # k=None means "uncoded"
    if N == 2:
        return ("1", 1) if k is None else ("0", 1)  # k=0
    elif 2 < N <= 4:
        if k is None:    return ("11", 2)
        if k == 0:       return ("00", 2)
        if k == 1:       return ("01", 2)
        if k == 2:       return ("10", 2)
        raise ValueError(f"k out of spec range {k!r} for N={N}")
    elif 4 < N <= 8:
        if k is None: return ("111", 3)
        # k in 0..6 (spec lists up to 6 here)
        return (format(k, "03b"), 3)  # 000..110 for k=0..6
    elif 8 < N <= 10:
        if k is None: return ("1111", 4)
        # k in 0..8 -> 0000..1000
        return (format(k, "04b"), 4)
    else:
        raise ValueError(f"N out of spec range: {N}")

def _estimate_bits_uncoded(J: int, N: int, first_gaggle: bool) -> int:
    id_bits = _code_id_bits_len(N, None)[1]
    ref_bits = N if first_gaggle else 0
    return id_bits + ref_bits + J * N

def _estimate_bits_vlc(deltas: np.ndarray, k: int, N: int, first_gaggle: bool) -> int:
    # VLC per §4.3.2.9–.10: unary part (z zeros + 1) then k LSBs; concatenate first parts then second parts.
    id_bits = _code_id_bits_len(N, k)[1]
    ref_bits = N if first_gaggle else 0
    z_sum = int(np.sum(deltas >> k))     # floor(δ / 2^k) summed
    return id_bits + ref_bits + (z_sum + deltas.size) + (k * deltas.size)

def _select_k_opt(deltas: np.ndarray, N: int, first_gaggle: bool) -> Tuple[int | None, int]:
    # Try uncoded and allowed k per N, pick minimum bits; tie-breaking per §4.3.2.13
    best_k = None
    best_bits = _estimate_bits_uncoded(len(deltas), N, first_gaggle)
    # allowed k set by N
    if N == 2: ks = [0]
    elif 2 < N <= 4: ks = [0,1,2]
    elif 4 < N <= 8: ks = list(range(0,7))
    elif 8 < N <= 10: ks = list(range(0,9))
    else: ks = []
    for k in ks:
        bits = _estimate_bits_vlc(deltas, k, N, first_gaggle)
        if bits < best_bits or (bits == best_bits and best_k is not None and k < best_k):
            best_bits, best_k = bits, k
    return best_k, best_bits

def _select_k_heur(deltas: np.ndarray, N: int, J: int) -> int | None:
    # Spec heuristic (Table 4-10). §4.3.2.11
    Delta = int(np.sum(deltas))
    if 64*Delta >= (23 * J * (1 << N)):
        return None  # uncoded
    if 207*J > 128*Delta:
        return 0
    if (J*(1<<N) + 5) <= (128*Delta + 49*J):
        return N-2
    # otherwise largest k <= N-2 s.t. J*2^k + 7 <= 128*Delta + 49*J
    rhs = 128*Delta + 49*J
    k = min(N-2, 31)
    while k >= 0 and not (J*(1<<k) + 7 <= rhs):
        k -= 1
    return max(k, 0)

# ------------------------
# Public API
# ------------------------

def encode_dc(coeffs: np.ndarray, levels: int, bitshift_ll3: int = 0,
              code_sel: CodeSel = "opt") -> Tuple[bytes, DCEncodeInfo]:
    """
    Encode DC per CCSDS §4.3 for one segment's coefficients.
    Returns (payload, info). Payload contains concatenated gaggles (IDs, ref sample for first, and data).
    """
    H, W = coeffs.shape
    rects = subband_rects(H, W, levels)
    LL = rects[f"LL{levels}"]; y,x,h,w = LL.y, LL.x, LL.h, LL.w
    dc = coeffs[y:y+h, x:x+w].reshape(-1).astype(np.int64)
    S = dc.size

    BitDepthDC = _bitdepth_dc(dc)
    BitDepthAC = _bitdepth_ac(coeffs, LL, levels)
    qprime = _table_4_8_qprime(BitDepthDC, BitDepthAC)
    q = max(qprime, bitshift_ll3)  # §4.3.1.3
    N = max(BitDepthDC - q, 1)     # eq. 17

    # quantized c' (eq. 16)
    cprime = (dc >> q).astype(np.int64)  # floor division since nonnegative shift; safe for negatives in Python

    bw = BitWriter()
    # First gaggle: write reference sample (N bits two's-complement) after the option ID (§4.3.2.3 + Fig 4-9a)
    sizes = _gaggle_sizes(S)

    # walk gaggles
    idx = 0
    for g, J in enumerate(sizes):
        first = (g == 0)
        # deltas to map (eqs. 18–19)
        if first:
            ref = int(cprime[0])
            idx = 1
        # build θ for each δ'm
        deltas = np.empty(J, dtype=np.int64)
        for j in range(J):
            m = idx + j
            prev = int(cprime[m-1])
            curr = int(cprime[m])
            # bounds for N-bit two's comp: xmin=-2^{N-1}, xmax=2^{N-1}-1
            xmin = -(1 << (N-1))
            xmax =  (1 << (N-1)) - 1
            theta = min(prev - 1 - xmin, xmax - prev)
            dprim = curr - prev  # δ'
            if 0 <= dprim <= theta:
                dm = 2 * dprim
            elif -theta <= dprim < 0:
                dm = 2 * abs(dprim) - 1
            else:
                dm = theta + abs(dprim)
            deltas[j] = dm

        # choose code option
        if N == 1:
            # still emit ID for 'uncoded' for consistency of stream layout
            k = None
        else:
            if code_sel == "opt":
                k, _ = _select_k_opt(deltas.astype(np.int64), N, first)
            else:
                k = _select_k_heur(deltas.astype(np.int64), N, J)

        # write ID field (§4.3.2.7, Table 4-9)
        id_bits, _ = _code_id_bits_len(N, k)
        bw.write_bits(int(id_bits, 2), len(id_bits))

        # reference sample for first gaggle
        if first:
            # write N-bit two's complement for ref
            v = ref & ((1 << N) - 1)
            bw.write_bits(v, N)

        # write gaggle payload
        if k is None:
            # uncoded, write each δm as N-bit unsigned (§4.3.2.8)
            for dm in deltas:
                bw.write_bits(int(dm) & ((1 << N) - 1), N)
        else:
            # first parts: unary z zeros + 1, where z=floor(dm/2^k) (§4.3.2.9–.10)
            for dm in deltas:
                z = int(dm) >> k
                if z > 0:
                    bw.write_bits(0, z)
                bw.write_bits(1, 1)
            # second parts: k LSBs of dm
            if k > 0:
                for dm in deltas:
                    bw.write_bits(int(dm) & ((1 << k) - 1), k)

        idx += J

    return bw.to_bytes(), DCEncodeInfo(BitDepthDC, BitDepthAC, q, N, S, sizes)

def decode_dc(coeffs_shape: Tuple[int,int], levels: int, payload: bytes, info: DCEncodeInfo) -> np.ndarray:
    """
    Inverse of encode_dc: reconstructs c' (quantized DC). You’ll add DC refinement (4.3.3) later.
    """
    H, W = coeffs_shape
    out = np.zeros((H, W), dtype=np.int32)
    br = BitReader(payload)
    N = info.N
    S = info.S

    # helper to read ID, mirrored Table 4-9
    def read_id() -> int | None:
        if N == 2:
            b = br.read_bits(1)
            return None if b == 1 else 0
        elif 2 < N <= 4:
            b = br.read_bits(2)
            if b == 0b11: return None
            if b == 0b00: return 0
            if b == 0b01: return 1
            if b == 0b10: return 2
        elif 4 < N <= 8:
            b = br.read_bits(3)
            if b == 0b111: return None
            return b  # 0..6
        else:  # 8 < N <= 10
            b = br.read_bits(4)
            if b == 0b1111: return None
            return b  # 0..8

    # walk gaggles
    HLL = subband_rects(H, W, levels)[f"LL{levels}"]
    y,x,h,w = HLL.y, HLL.x, HLL.h, HLL.w
    cprime = np.zeros(h*w, dtype=np.int64)

    sizes = info.gaggle_sizes
    idx = 0
    for g, J in enumerate(sizes):
        first = (g == 0)
        k = read_id()

        if first:
            # N-bit two's complement
            ref = br.read_bits(N)
            if ref & (1 << (N-1)):  # negative
                ref = ref - (1 << N)
            cprime[0] = ref
            idx = 1

        if k is None:
            # uncoded fixed N-bit words
            deltas = np.fromiter((br.read_bits(N) for _ in range(J)), dtype=np.int64, count=J)
        else:
            # first parts: unary, count zeros to the 1
            z = []
            for _ in range(J):
                count = 0
                while True:
                    bit = br.read_bits(1)
                    if bit == 1: break
                    count += 1
                z.append(count)
            # second parts: k LSBs
            lsbs = [br.read_bits(k) if k > 0 else 0 for _ in range(J)]
            deltas = np.array([(z[i] << k) | lsbs[i] for i in range(J)], dtype=np.int64)

        # invert mapping (eq. 19)
        for j in range(J):
            m = idx + j
            prev = int(cprime[m-1])
            xmin = -(1 << (N-1))
            xmax =  (1 << (N-1)) - 1
            theta = min(prev - 1 - xmin, xmax - prev)
            dm = int(deltas[j])

            if dm <= 2*theta:
                if (dm & 1) == 0:
                    dprim = dm // 2
                else:
                    dprim = -((dm + 1) // 2)
            else:
                # “otherwise” branch: δm = θ + |δ'|
                # choose sign so that c'm stays within [xmin, xmax] and far side of θ
                # if prev + (theta + t) would exceed xmax, it must be negative overflow
                mag = dm - theta
                if prev + mag > xmax:
                    dprim = -mag
                elif prev - mag < xmin:
                    dprim = +mag
                else:
                    # if both fit, lean toward direction that *exceeds* theta region
                    dprim = +mag if (xmax - prev) >= (prev - 1 - xmin) else -mag

            cprime[m] = prev + dprim
        idx += J

    # place back into LL_top rect
    out[y:y+h, x:x+w] = cprime.reshape(h, w).astype(np.int32)
    return out

@dataclass
class DCRefineInfo:
    q: int
    start_b: int           # = q-1
    stop_b: int            # = max(BitDepthAC, BitShift(LL3))
    planes: int            # = start_b - stop_b + 1, or 0 if start<stop
    S: int                 # number of DC samples in segment

def encode_dc_refinement(coeffs: np.ndarray, levels: int, *,
                         q: int,
                         bitdepth_ac: int,
                         bitshift_ll3: int = 0) -> tuple[bytes, DCRefineInfo]:
    """
    Implements CCSDS 122.0-B-2 §4.3.3. Uncoded planes of the *original* DC coefficients.
    """
    H, W = coeffs.shape
    LL = subband_rects(H, W, levels)[f"LL{levels}"]
    y,x,h,w = LL.y, LL.x, LL.h, LL.w
    dc = coeffs[y:y+h, x:x+w].reshape(-1).astype(np.int64)  # original (pre-quantization)

    start_b = q - 1
    stop_b  = max(bitdepth_ac, bitshift_ll3)
    planes = max(0, start_b - stop_b + 1)
    if planes == 0:
        return b"", DCRefineInfo(q, start_b, stop_b, 0, dc.size)

    bw = BitWriter()
    for b in range(start_b, stop_b - 1, -1):
        mask = 1 << b
        for v in dc:
            bw.write_bits(1 if (int(v) & mask) else 0, 1)
    return bw.to_bytes(), DCRefineInfo(q, start_b, stop_b, planes, dc.size)

def decode_dc_refinement(shape: tuple[int,int], levels: int, payload: bytes, info: DCRefineInfo) -> np.ndarray:
    """
    Inverse of encode_dc_refinement: returns an array with only the refined DC bit-planes set.
    Combine this with c' (from decode_dc) to fully reassemble DC later (or leave for Stage 0 in §4.5).
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.int32)
    if info.planes == 0:
        return out

    br = BitReader(payload)
    LL = subband_rects(H, W, levels)[f"LL{levels}"]
    y,x,h,w = LL.y, LL.x, LL.h, LL.w
    S = h * w
    dc_ref = np.zeros(S, dtype=np.int64)

    for b in range(info.start_b, info.stop_b - 1, -1):
        bit = 1 << b
        for i in range(S):
            if br.read_bits(1):
                dc_ref[i] |= bit

    out[y:y+h, x:x+w] = dc_ref.reshape(h, w).astype(np.int32)
    return out
