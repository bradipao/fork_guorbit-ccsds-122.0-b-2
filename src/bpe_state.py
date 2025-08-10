# ccsds122/bpe_state.py
import numpy as np

class SigRefState:
    """
    Tracks significance and plane-wise sets:
      - sig_map[y,x] == 1 if coefficient is significant
      - new_list[plane] = list of coords that became significant at that plane
      - prev_list[plane] = list of coords that were already significant before this plane
                            (computed lazily when encoding Stage 3)
    """
    def __init__(self, shape):
        H, W = shape
        self.sig_map = np.zeros((H, W), dtype=np.uint8)
        self.new_by_plane: dict[int, list[tuple[int,int]]] = {}
        self.prev_by_plane: dict[int, list[tuple[int,int]]] = {}

    def note_new(self, plane_b: int, coords):
        self.new_by_plane.setdefault(plane_b, []).extend(coords)
        for (y, x) in coords:
            self.sig_map[y, x] = 1

    def compute_prev_for_plane(self, plane_b: int):
        """All coords significant BEFORE plane_b (i.e., seen in any plane > b)."""
        prev = []
        for p in self.new_by_plane:
            if p > plane_b:
                prev.extend(self.new_by_plane[p])
        self.prev_by_plane[plane_b] = prev
        return prev
