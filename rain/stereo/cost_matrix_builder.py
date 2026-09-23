"""CostMatrixBuilder 实现：把四个子代价归一化后按权重合成总代价。

四个子代价（任务书要求至少这四个）：
    epipolar  : 点到极线的距离（来自 EpipolarGate，这里只做归一化+加权，
                不重新计算几何）
    position  : 质心位置差。需要标定文件里的 rectify_a_to_b 把 A 的坐标
                投影到 B 的坐标系，没有这个矩阵就是不可用（None），不能
                直接比较两个不同相机、未标定的像素坐标系。
    area      : 等效直径差（像素）。两个相机分辨率相同（阶段1已确认都是
                2064x1544），在"看起来差不多大"这个粗糙意义上可以直接比较，
                但这不是严格的物理尺寸比较——像素-毫米标定完成前，
                这仍然只是像素量纲的差值。
    aspect    : 长宽比差。

某一项不可用时，剩余项的权重按比例重新归一化（不是当 0 分处理）。
"""

from __future__ import annotations

from ..registry import register_cost_matrix_builder
from ..types import Detection
from .types import CostBreakdown, StereoCalibration


@register_cost_matrix_builder("weighted_normalized_cost")
class CostMatrixBuilder:
    """四项子代价归一化 + 加权求和。

    参数（全部来自 config/pipeline.yaml 的 stereo.cost_matrix_builder）：
        weights    : dict，四个键 epipolar/position/area/aspect 的权重
        norm_scale : dict，四个键对应的归一化尺度——子代价除以这个数再
                     截断到 [0,1]，尺度选多大直接决定"多少像素算满分惩罚"，
                     必须来自配置，不能写死在代码里。
    """

    def __init__(
        self,
        weights: dict | None = None,
        norm_scale: dict | None = None,
    ):
        self.weights = weights or {"epipolar": 0.4, "position": 0.3, "area": 0.2, "aspect": 0.1}
        self.norm_scale = norm_scale or {
            "epipolar_px": 5.0, "position_px": 50.0, "area_px": 20.0, "aspect": 2.0,
        }

    def compute(
        self,
        det_a: Detection,
        det_b: Detection,
        epipolar_distance_px: float | None,
        calibration: StereoCalibration | None,
    ) -> CostBreakdown:
        raw = {
            "epipolar": self._norm(epipolar_distance_px, self.norm_scale["epipolar_px"]),
            "position": self._position_cost(det_a, det_b, calibration),
            "area": self._norm(abs(det_a.d_eq_px - det_b.d_eq_px), self.norm_scale["area_px"]),
            "aspect": self._norm(abs(self._aspect(det_a) - self._aspect(det_b)), self.norm_scale["aspect"]),
        }

        available = {k: v for k, v in raw.items() if v is not None}
        if not available:
            # 四项全部不可用（比如既没标定也没有效数值）——总代价没有意义，
            # 用 1.0（最差）而不是 0.0（最好），避免"什么都不知道"被误当成"完美匹配"。
            total = 1.0
        else:
            w_sum = sum(self.weights[k] for k in available)
            total = sum(self.weights[k] * v for k, v in available.items()) / w_sum

        return CostBreakdown(
            cost_epipolar=raw["epipolar"],
            cost_position=raw["position"],
            cost_area=raw["area"],
            cost_aspect=raw["aspect"],
            total_cost=float(total),
        )

    def _position_cost(
        self, det_a: Detection, det_b: Detection, calibration: StereoCalibration | None
    ) -> float | None:
        if calibration is None or calibration.rectify_a_to_b is None:
            return None
        import numpy as np

        a_h = np.array([det_a.cx, det_a.cy, 1.0])
        proj = calibration.rectify_a_to_b @ a_h
        if abs(proj[2]) < 1e-9:
            return None
        proj_x, proj_y = proj[0] / proj[2], proj[1] / proj[2]
        dist = float(np.hypot(proj_x - det_b.cx, proj_y - det_b.cy))
        return self._norm(dist, self.norm_scale["position_px"])

    @staticmethod
    def _aspect(det: Detection) -> float:
        w, h = det.bbox[2], det.bbox[3]
        return max(w, h) / max(min(w, h), 1)

    @staticmethod
    def _norm(value: float | None, scale: float) -> float | None:
        if value is None:
            return None
        if scale <= 0:
            return None
        return min(float(value) / float(scale), 1.0)
