from __future__ import annotations
import numpy as np
from .io_segmentation import BitWriter, BitReader
from .bpe_dc import _select_k_opt, _select_k_heur, _code_id_bits_len, CodeSel

def encode_gaggle(values: np.ndarray, N: int, first_gaggle: bool, code_sel: CodeSel) -> bytes:
    """
    Encode a gaggle of nonnegative integers with parameter N bits per uncoded word.
    Chooses a VLC option k (or 'uncoded') per Table 4-9.
    For N==1 we emit raw 1 bit per symbol (no option ID).
    """
    if values.size == 0:
        return b""
    bw = BitWriter()

    if N == 1:
        for v in values:
            bw.write_bits(int(v) & 1, 1)
        return bw.to_bytes()

    # choose option
    if code_sel == "opt":
        k, _ = _select_k_opt(values.astype(np.int64), N, first_gaggle)
    else:
        k = _select_k_heur(values.astype(np.int64), N, len(values))

    # ID bits
    id_bits, _ = _code_id_bits_len(N, k)
    bw.write_bits(int(id_bits, 2), len(id_bits))

    # payload
    if k is None:  # uncoded N-bit
        for v in values:
            bw.write_bits(int(v) & ((1 << N) - 1), N)
    else:
        # unary first parts
        for v in values:
            z = int(v) >> k
            if z > 0:
                bw.write_bits(0, z)
            bw.write_bits(1, 1)
        # LSBs
        if k > 0:
            for v in values:
                bw.write_bits(int(v) & ((1 << k) - 1), k)

    return bw.to_bytes()

def decode_gaggle(br: BitReader, count: int, N: int, first_gaggle: bool) -> np.ndarray:
    """
    Read exactly 'count' symbols from 'br' that were encoded by encode_gaggle.
    """
    if count == 0:
        return np.zeros(0, dtype=np.int64)

    if N == 1:
        return np.fromiter((br.read_bits(1) for _ in range(count)), dtype=np.int64, count=count)

    # read option ID
    def read_id() -> int | None:
        if N == 2:
            return None if br.read_bits(1) == 1 else 0
        elif 2 < N <= 4:
            b = br.read_bits(2)
            return None if b == 0b11 else (0 if b == 0b00 else 1 if b == 0b01 else 2)
        elif 4 < N <= 8:
            b = br.read_bits(3)
            return None if b == 0b111 else b
        else:  # 8 < N <= 10
            b = br.read_bits(4)
            return None if b == 0b1111 else b

    k = read_id()
    vals = np.zeros(count, dtype=np.int64)

    if k is None:
        for i in range(count):
            vals[i] = br.read_bits(N)
        return vals

    # unary first parts
    zlist = []
    for _ in range(count):
        z = 0
        while True:
            if br.read_bits(1) == 1:
                break
            z += 1
        zlist.append(z)

    # LSBs
    lsbs = [br.read_bits(k) if k > 0 else 0 for _ in range(count)]
    for i in range(count):
        vals[i] = (zlist[i] << k) | lsbs[i]
    return vals
