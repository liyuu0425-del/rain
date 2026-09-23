"""LinearVelocityEstimator 实现：对单条轨迹拟合 z(t) = z0 + v*t。

z 取轨迹点的纵坐标 cy（像素，向下为正——雨滴下落是主运动方向），
t 取 TrackPoint.device_time_s（阶段3已经保证来自
a_device_relative_us / b_device_relative_us，不是文件名毫秒、host接收
时间或帧号/固定帧率）。

拟合规则（任务书"建议规则"，全部做成可配置参数）：
    n <  2 : 不输出速度（点数不足）
    n == 2 : 两点直接定线，输出速度但标"低置信度"（没有统计意义上的
             残差/R²，任何测量误差都会被这两点直接放大成斜率误差）
    n >= 3 : 最小二乘拟合，同时算残差和R²，按残差/R²/点数分级置信度
"""

from __future__ import annotations

import numpy as np

from ..registry import register_velocity_estimator
from ..tracking.types import TrackRecord
from .types import VelocityFitResult

_SPATIAL_STATUS = "spatial_calibration_unavailable"


@register_velocity_estimator("linear_velocity_estimator")
class LinearVelocityEstimator:
    """线性最小二乘测速。

    参数（全部来自 config/pipeline.yaml 的 measurement.linear_velocity_estimator）：
        min_points_for_fit        : 至少几个真实检测点才做"正规"拟合（默认3）。
        max_residual_px           : 均方根残差超过这个值(像素)，判定"残差过大"。
        min_r_squared              : R² 低于这个值，判定"拟合质量差"。
        high_confidence_min_points : 点数达到这个数量、且残差/R²都合格时，
                                     才给"高"置信度；否则给"中"。
    """

    def __init__(
        self,
        min_points_for_fit: int = 3,
        max_residual_px: float = 5.0,
        min_r_squared: float = 0.8,
        high_confidence_min_points: int = 5,
    ):
        self.min_points_for_fit = int(min_points_for_fit)
        self.max_residual_px = float(max_residual_px)
        self.min_r_squared = float(min_r_squared)
        self.high_confidence_min_points = int(high_confidence_min_points)

    def estimate(self, track: TrackRecord) -> VelocityFitResult:
        # 防御性过滤：只用真实检测关联点。阶段3的 TrackRecord.points 设计上
        # 已经不包含"仅预测"点，这里再显式过滤一遍，双重保险——
        # 测速模块不能信任"上游一定没传错"，必须自己也守住这条线。
        pts = [p for p in track.points if p.detected is not None]
        n = len(pts)

        base_kwargs = dict(
            track_id=track.track_id, camera=track.camera, n_points=n,
            spatial_calibration_status=_SPATIAL_STATUS,
        )

        if n < 2:
            return VelocityFitResult(
                **base_kwargs, duration_s=None, velocity_px_per_s=None, z0_px=None,
                vx_px_per_s=None, residual_rms_px=None, r_squared=None,
                time_quality="n/a", unit="", confidence="none",
                status="insufficient_points", reason="insufficient_points",
            )

        # 按 device_time_s 排序只影响最小二乘的计算顺序，不影响拟合结果本身
        # （回归对输入顺序不敏感）；时间质量检查用排序前的原始顺序——
        # 轨迹点本来就是按 pair_index 递增顺序 append 进 track.points 的，
        # 如果排序前后顺序不一致，说明设备时间本身有倒退/重复，
        # 这正是要检测的问题，不能先排序掩盖掉。
        t_raw = np.array([p.device_time_s for p in pts])
        time_quality = self._check_time_quality(t_raw)

        order = np.argsort(t_raw)
        t = t_raw[order]
        z = np.array([pts[i].detected.cy for i in order])
        x = np.array([pts[i].detected.cx for i in order])

        duration = float(t[-1] - t[0])

        if time_quality == "degenerate_time_span":
            return VelocityFitResult(
                **base_kwargs, duration_s=duration, velocity_px_per_s=None, z0_px=None,
                vx_px_per_s=None, residual_rms_px=None, r_squared=None,
                time_quality=time_quality, unit="", confidence="none",
                status="rejected", reason="degenerate_time_span",
            )

        if n == 2:
            v = float((z[1] - z[0]) / (t[1] - t[0]))
            z0 = float(z[0] - v * t[0])
            vx = float((x[1] - x[0]) / (t[1] - t[0]))
            confidence = "low" if time_quality == "ok" else "none"
            reason = "two_point_fit_low_confidence" if time_quality == "ok" else f"time_quality_{time_quality}"
            return VelocityFitResult(
                **base_kwargs, duration_s=duration, velocity_px_per_s=v, z0_px=z0,
                vx_px_per_s=vx, residual_rms_px=0.0, r_squared=None,
                time_quality=time_quality, unit="px/s", confidence=confidence,
                status="fitted_two_point", reason=reason,
            )

        # n >= 3：最小二乘线性拟合
        v, z0, residual_rms, r2 = self._fit_linear(t, z)
        vx, _, _, _ = self._fit_linear(t, x)

        confidence, reason = self._grade(n, residual_rms, r2, time_quality)
        status = "fitted" if confidence != "none" else "rejected"

        return VelocityFitResult(
            **base_kwargs, duration_s=duration, velocity_px_per_s=v, z0_px=z0,
            vx_px_per_s=vx, residual_rms_px=residual_rms, r_squared=r2,
            time_quality=time_quality, unit="px/s", confidence=confidence,
            status=status, reason=reason,
        )

    # -------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------

    @staticmethod
    def _check_time_quality(t_raw: np.ndarray) -> str:
        # 先看整条轨迹的真实时间跨度（首尾最大最小值之差，不依赖点的原始
        # 顺序）——如果所有点其实都挤在同一个时间点上，这是比"局部有
        # 重复"更严重、更具体的问题，必须优先报出来，不能被下面逐点的
        # diff 检查捂住。
        span = float(t_raw.max() - t_raw.min())
        if span < 1e-9:
            return "degenerate_time_span"

        diffs = np.diff(t_raw)
        if np.any(diffs < 0):
            return "non_monotonic"
        if np.any(diffs == 0):
            return "duplicate_timestamps"
        return "ok"

    @staticmethod
    def _fit_linear(t: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float]:
        """最小二乘拟合 z = v*t + z0，返回 (v, z0, 残差RMS, R²)。"""
        v, z0 = np.polyfit(t, z, 1)
        pred = v * t + z0
        residuals = z - pred
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((z - np.mean(z)) ** 2))
        if ss_tot < 1e-9:
            # 所有点 z 几乎相同（轨迹几乎不动）：残差几乎为0时拟合"完美"，
            # 否则说明这点数据本身噪声就很大，R²退化没有意义，给0不给nan。
            r2 = 1.0 if ss_res < 1e-9 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        return float(v), float(z0), rms, float(r2)

    def _grade(self, n: int, residual_rms: float, r2: float, time_quality: str) -> tuple[str, str]:
        if time_quality != "ok":
            return "low", f"time_quality_{time_quality}"
        if n < self.min_points_for_fit:
            # 理论上 n>=3 就会进入这个分支（n==2 走上面的两点专用逻辑），
            # 但 min_points_for_fit 可以在配置里调得比3更高——那样点数
            # 不够"正规拟合"标准的轨迹仍然给出拟合结果，只是不能给中/高置信度。
            return "low", "below_min_points_for_fit"
        if residual_rms > self.max_residual_px:
            return "low", "high_residual"
        if r2 < self.min_r_squared:
            return "low", "low_r_squared"
        if n >= self.high_confidence_min_points:
            return "high", "ok"
        return "medium", "ok"
