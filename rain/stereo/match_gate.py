"""MatchGate 实现：极线约束之后的第二道门控，用简单的物理量硬性排除
明显不可能是同一颗雨滴的候选（不需要标定，纯粹比较两个检测自身的形状）。

和 EpipolarGate 的分工：
    EpipolarGate 回答"这两个点的空间几何关系合不合理"（需要标定）；
    MatchGate    回答"这两个目标看起来像不像同一颗雨滴"（不需要标定，
                 只看检测自身的面积、长宽比）。
两道门都要过，才轮到 CostMatrixBuilder 算加权总代价。
"""

from __future__ import annotations

from ..registry import register_match_gate
from ..types import Detection


@register_match_gate("size_shape_gate")
class MatchGate:
    """面积比 + 长宽比差 的硬性门控。

    参数：
        max_area_ratio  : 两个检测面积比（大/小）的上限。同一颗雨滴在
                          A、B 两个视角下投影面积可能不同，但不应该
                          相差太悬殊。
        max_aspect_diff : 两个检测长宽比之差的上限。
    """

    def __init__(self, max_area_ratio: float = 3.0, max_aspect_diff: float = 2.0):
        self.max_area_ratio = float(max_area_ratio)
        self.max_aspect_diff = float(max_aspect_diff)

    def passes(self, det_a: Detection, det_b: Detection) -> tuple[bool, str]:
        area_ratio = self._area_ratio(det_a.area_px, det_b.area_px)
        if area_ratio > self.max_area_ratio:
            return False, "area_ratio_exceeded"

        aspect_a = self._aspect(det_a)
        aspect_b = self._aspect(det_b)
        if abs(aspect_a - aspect_b) > self.max_aspect_diff:
            return False, "aspect_diff_exceeded"

        return True, "ok"

    @staticmethod
    def _area_ratio(area_a: float, area_b: float) -> float:
        lo = max(min(area_a, area_b), 1e-6)
        hi = max(area_a, area_b)
        return hi / lo

    @staticmethod
    def _aspect(det: Detection) -> float:
        w, h = det.bbox[2], det.bbox[3]
        return max(w, h) / max(min(w, h), 1)
