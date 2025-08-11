from __future__ import annotations
from typing import Dict, Tuple, List
from .io_segmentation import BitWriter, BitReader

#  ID bits 
# For mapped pattern size N, we write these IDs once per gaggle per N *before* the first codeword of that N.
# N=2: 1 ID bit; N=3,4: 2 ID bits. 'uncoded' is the last option listed in the table.
_ID_BITS = {
    2: {0: "0", 1: "1"},                       # 0: option0, 1: uncoded
    3: {0: "00", 1: "01", 3: "11"},            # 0,1, (3=uncoded)
    4: {0: "00", 1: "01", 2: "10", 3: "11"},   # 0,1,2, (3=uncoded)
}

# VLC code tables (Tables 4-15..4-17)
_VLC_2BIT: Dict[int, Dict[int, str]] = {
    0: {0: "1",   1: "01",  2: "001", 3: "000"},
    1: {0: "00",  1: "01",  2: "10",  3: "11"},  
}
_VLC_3BIT: Dict[int, Dict[int, str]] = {
    0: {0:"1", 1:"01", 2:"001", 3:"00000", 4:"00001", 5:"00010", 6:"000110", 7:"000111"},
    1: {0:"10",1:"11", 2:"010", 3:"011",   4:"0010",  5:"0011",  6:"0000",   7:"0001"},
    3: {0:"000",1:"001",2:"010",3:"011",   4:"100",   5:"101",   6:"110",    7:"111"},
}
_VLC_4BIT: Dict[int, Dict[int, str]] = {
    0: {
        0:"1", 1:"01", 2:"001", 3:"0001", 4:"0000000", 5:"0000001", 6:"0000010", 7:"0000011",
        8:"00001000", 9:"00001001", 10:"00001010", 11:"00001011", 12:"00001100", 13:"00001101",
        14:"00001110", 15:"00001111",
    },
    1: {
        0:"10",1:"11",2:"010",3:"011",4:"0010",5:"0011",6:"000000",7:"000001",
        8:"000010",9:"000011",10:"000100",11:"000101",12:"0001100",13:"0001101",
        14:"0001110",15:"0001111",
    },
    2: {
        0:"100",1:"101",2:"110",3:"111",4:"0100",5:"0101",6:"0110",7:"0111",
        8:"00100",9:"00101",10:"00110",11:"00111",12:"00000",13:"00001",
        14:"00010",15:"00011",
    },
    3: {  # uncoded (4 bits)
        0:"0000",1:"0001",2:"0010",3:"0011",4:"0100",5:"0101",6:"0110",7:"0111",
        8:"1000",9:"1001",10:"1010",11:"1011",12:"1100",13:"1101",14:"1110",15:"1111",
    },
}

def _vlc_table(N: int):
    if N == 2: return _VLC_2BIT
    if N == 3: return _VLC_3BIT
    if N == 4: return _VLC_4BIT
    raise ValueError("N must be 2, 3, or 4")

def best_option_for_symbols(symbols: List[int], N: int) -> int:
    """
    Pick option that minimizes total length (ties -> smallest option index),
    per §4.5.3.3.4–.3.5 and Table 4-18. Options available:
      N=2: {0, 1(uncoded)}; N=3: {0,1,3(uncoded)}; N=4: {0,1,2,3(uncoded)}.
    """
    tbl = _vlc_table(N)
    # ID bits cost (once per gaggle when the first codeword of this N appears)
    id_bits = len(_ID_BITS[N][list(_ID_BITS[N].keys())[0]]) 
    best = None
    best_len = None
    for opt, codebook in tbl.items():
        # cost = ID (if we have any symbols) + sum codeword lengths
        if len(symbols) == 0:
            total = 0
        else:
            total = (1 if N == 2 else 2)  # Table 4-18
            total += sum(len(codebook[s]) for s in symbols)
        if best_len is None or total < best_len or (total == best_len and (best is None or opt < best)):
            best = opt
            best_len = total
    assert best is not None
    return best

def write_id_bits(bw: BitWriter, N: int, option: int) -> None:
    bits = _ID_BITS[N][option]
    bw.write_bits(int(bits, 2), len(bits))

def encode_symbol_with_option(bw: BitWriter, symbol: int, N: int, option: int) -> None:
    code = _vlc_table(N)[option][symbol]
    bw.write_bits(int(code, 2), len(code))

def decode_symbol_with_option(br: BitReader, N: int, option: int) -> int:
    # Scalar decoder: read bits until a codeword in the chosen table matches.
    # Build reverse dict once per call:
    inv = {v: k for k, v in _vlc_table(N)[option].items()}
    acc = ""
    while True:
        bit = br.read_bits(1)
        acc += "1" if bit else "0"
        if acc in inv:
            return inv[acc]
