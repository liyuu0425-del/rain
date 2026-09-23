"""阶段4双路匹配模块的接口定义。

和 rain/interfaces.py、rain/detection/interfaces.py、
rain/tracking/interfaces.py 是同一类东西。HungarianSolver 复用
rain/tracking/interfaces.py 里已经定义的协议，这里不重复定义。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Detection
from .types import CostBreakdown, StereoCalibration, StereoFrameResult


@runtime_checkable
class EpipolarGate(Protocol):
    """极线约束：判断 A、B 两个点是否满足对极几何关系。

    没有标定文件时，is_available 为 False，任何候选都判不通过，
    原因固定是 "calibration_unavailable"——不得虚构参数硬算一个"看似
    正常"的距离。
    """

    @property
    def is_available(self) -> bool: ...

    def passes(
        self, pt_a: tuple[float, float], pt_b: tuple[float, float]
    ) -> tuple[bool, float | None, str]:
        """返回 (是否通过, 极线距离或None, 原因)。"""
        ...


@runtime_checkable
class MatchGate(Protocol):
    """极线约束之后的第二道门控：不需要标定，只比较检测自身的形状/大小。"""

    def passes(self, det_a: Detection, det_b: Detection) -> tuple[bool, str]:
        ...


@runtime_checkable
class CostMatrixBuilder(Protocol):
    """把四个子代价（极线/位置/面积/形状）归一化后按权重合成总代价。"""

    def compute(
        self,
        det_a: Detection,
        det_b: Detection,
        epipolar_distance_px: float | None,
        calibration: StereoCalibration | None,
    ) -> CostBreakdown:
        ...


@runtime_checkable
class NearestNeighborSolver(Protocol):
    """简单场景（无候选冲突）下的贪心最近邻分配。"""

    def solve(self, cost_matrix) -> list[tuple[int, int]]:
        ...


@runtime_checkable
class StereoMatcher(Protocol):
    """组合模块：极线门控 -> 匹配门控 -> 代价矩阵 -> 最近邻/匈牙利分配。"""

    def match_frame(
        self, pair_index: int, dets_a: list[Detection], dets_b: list[Detection]
    ) -> StereoFrameResult:
        ...
