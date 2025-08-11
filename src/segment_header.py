# Segment Header
from __future__ import annotations
from dataclasses import dataclass
from .io_segmentation import BitWriter, BitReader

MAGIC = 0xCC  

@dataclass
class SegmentHeader:
    levels: int
    h: int
    w: int
    band: int

    def encode(self) -> bytes:
        bw = BitWriter()
        bw.write_bits(MAGIC, 8)
        bw.write_bits(self.levels & 0x7, 3)
        bw.write_bits(self.band & 0x7, 3)
        bw.write_bits(self.h, 16)
        bw.write_bits(self.w, 16)
        return bw.to_bytes()

    @staticmethod
    def decode(data: bytes) -> tuple["SegmentHeader", int]:
        br = BitReader(data)
        magic = br.read_bits(8)
        if magic != MAGIC:
            raise ValueError("Bad segment magic")
        levels = br.read_bits(3)
        band = br.read_bits(3)
        h = br.read_bits(16)
        w = br.read_bits(16)
        # Return header and number of bytes consumed
        used = (8+3+3+16+16 + (8 - ((8+3+3+16+16) % 8))%8) // 8
        return SegmentHeader(levels=levels, h=h, w=w, band=band), used
