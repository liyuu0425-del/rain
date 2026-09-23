"""EpipolarGate 实现：用基础矩阵 F 判断"A图像里的点"和"B图像里的点"
是否满足极线约束。

任务书的硬性约束：没有标定文件时不得虚构参数。这个类的行为分两种情况：
    - 有标定 (calibration is not None) : 真正计算点到极线的距离，
      距离超过 max_epipolar_dist_px 就判定不通过。
    - 没有标定 (calibration is None)   : 任何候选都判定不通过，原因固定
      标记为 "calibration_unavailable"——不是"极线距离超标"，两者含义
      完全不同，报告和 CSV 里必须能区分开。
"""

from __future__ import annotations

import numpy as np

from ..registry import register_epipolar_gate
from .types import StereoCalibration


@register_epipolar_gate("fundamental_matrix_gate")
class EpipolarGate:
    """基于基础矩阵的极线门控。

    参数：
        calibration           : StereoCalibration 或 None（无标定文件时）。
                                 由调用方（StereoMatcher）通过
                                 CalibrationLoader 加载后传入，这个类本身
                                 不负责读文件。
        max_epipolar_dist_px  : 点到极线的距离阈值（像素），来自
                                 config/pipeline.yaml，不写死。
    """

    def __init__(self, calibration: StereoCalibration | None = None, max_epipolar_dist_px: float = 5.0):
        self.calibration = calibration
        self.max_epipolar_dist_px = float(max_epipolar_dist_px)

    @property
    def is_available(self) -> bool:
        return self.calibration is not None

    def epipolar_distance(self, pt_a: tuple[float, float], pt_b: tuple[float, float]) -> float | None:
        """返回 pt_b 到"pt_a 对应的极线"的距离（像素）。

        没有标定或极线退化（方向向量接近零）时返回 None——调用方不能把
        None 当成 0 处理，那样等于"默认通过"，是错误的默认行为。
        """
        if self.calibration is None:
            return None

        F = self.calibration.fundamental_matrix
        a_h = np.array([pt_a[0], pt_a[1], 1.0])
        line = F @ a_h  # [l1, l2, l3]，B 图像里满足 l1*x + l2*y + l3 = 0 的直线

        denom = float(np.hypot(line[0], line[1]))
        if denom < 1e-9:
            return None

        b_h = np.array([pt_b[0], pt_b[1], 1.0])
        return float(abs(line @ b_h) / denom)

    def passes(self, pt_a: tuple[float, float], pt_b: tuple[float, float]) -> tuple[bool, float | None, str]:
        """返回 (是否通过门控, 极线距离或None, 原因)。

        原因取值：
            "ok"                       : 通过
            "calibration_unavailable"  : 没有标定文件，无法计算，直接判不通过
            "degenerate_epipolar_line" : 标定数值退化，无法计算距离
            "epipolar_gate_rejected"   : 算出了距离，但超过阈值
        """
        if not self.is_available:
            return False, None, "calibration_unavailable"

        d = self.epipolar_distance(pt_a, pt_b)
        if d is None:
            return False, None, "degenerate_epipolar_line"
        if d > self.max_epipolar_dist_px:
            return False, d, "epipolar_gate_rejected"
        return True, d, "ok"
