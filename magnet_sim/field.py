"""由若干区域组成的复合磁场（磁场方向垂直纸面）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .regions import Region

OUT = "out"  # 垂直纸面向外（⊙，记为 +z）
IN = "in"  # 垂直纸面向里（⊗，记为 -z）

DIRECTION_LABELS = {OUT: "穿出纸面 ·", IN: "穿入纸面 ×"}


@dataclass
class FieldRegion:
    region: Region
    B: float  # 磁感应强度大小，>= 0；B = 0 可用来"挖空"其他区域
    direction: str = OUT
    B_text: Optional[str] = None  # 用户输入的原始文字，便于编辑时回填

    @property
    def bz(self) -> float:
        """磁场的 z 分量：向外为正，向里为负。"""
        return self.B if self.direction == OUT else -self.B


class MagneticField:
    """复合磁场。区域重叠时，列表中靠前的区域优先；不在任何区域内则 B = 0。"""

    def __init__(self, regions: Optional[List[FieldRegion]] = None):
        self.regions: List[FieldRegion] = list(regions or [])

    def lookup(self, x: float, y: float) -> Tuple[int, float]:
        """返回 (所在区域的序号, Bz)，不在任何区域内时为 (-1, 0)。"""
        for i, fr in enumerate(self.regions):
            if fr.region.contains(x, y):
                return i, fr.bz
        return -1, 0.0

    def region_index(self, x: float, y: float) -> int:
        return self.lookup(x, y)[0]

    def index_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """对网格逐点求所在区域序号（-1 表示无场）。"""
        idx = np.full(np.shape(X), -1, dtype=int)
        # 倒序赋值，使靠前的区域最终覆盖靠后的区域
        for i in range(len(self.regions) - 1, -1, -1):
            mask = np.asarray(self.regions[i].region.contains(X, Y), dtype=bool)
            idx[mask] = i
        return idx

    def bz_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        idx = self.index_grid(X, Y)
        values = np.array([fr.bz for fr in self.regions] + [0.0])
        return values[idx]  # idx = -1 时取到最后一个元素 0
