"""阶段5/6测速与统计模块的接口定义。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..tracking.types import TrackRecord
from .types import ConcentrationResult, DropSizeDistributionResult, QualityVerdict, VelocityFitResult


@runtime_checkable
class LinearVelocityEstimator(Protocol):
    """对单条轨迹做 z(t) = z0 + v*t 的线性拟合，估计竖直方向速度。"""

    def estimate(self, track: TrackRecord) -> VelocityFitResult:
        """
        约定：
        - 只使用 track.points 里 detected 不为 None 的点（真实检测关联点）；
          卡尔曼"仅预测"的点不参与拟合，即使不小心混进来也必须被过滤掉。
        - 时间一律用 TrackPoint.device_time_s（阶段3已经保证它来自
          a_device_relative_us / b_device_relative_us，不是文件名毫秒、
          host接收时间或帧号/固定帧率）。
        - 没有像素-毫米标定，spatial_calibration_status 恒为
          "spatial_calibration_unavailable"，unit 恒为 "px/s"，
          不得输出 mm/s 或 m/s。
        """
        ...


@runtime_checkable
class StatisticsQualityController(Protocol):
    """判断一条轨迹是否够格进入"高可信统计"（阶段6专用质量门限，
    独立于阶段2的 Detection QC 位——这里是"要不要计入统计"的业务判断，
    不是"检测本身有没有问题"的算法判断，虽然会用到后者的结果）。
    """

    def evaluate(self, track: TrackRecord) -> QualityVerdict:
        ...


@runtime_checkable
class NumberConcentrationEstimator(Protocol):
    """名义数量浓度：C = N / V，V 来自配置的名义体积（未标定）。"""

    def estimate(self, n_valid_targets: int) -> ConcentrationResult:
        ...


@runtime_checkable
class DropSizeDistributionEstimator(Protocol):
    """按像素等效直径分箱统计目标数量分布。"""

    def estimate(self, camera: str, sizes_px: list[float]) -> DropSizeDistributionResult:
        ...


@runtime_checkable
class WindowStatistics(Protocol):
    """组合模块：按时间窗口切分 + 质量门限 + 数量浓度 + 粒径分布。

    本身不包含具体的门限判断/浓度公式实现，只负责按窗口把数据分组，
    调用 StatisticsQualityController / NumberConcentrationEstimator /
    DropSizeDistributionEstimator 完成计算。
    """

    def compute(
        self,
        camera: str,
        tracks: list[TrackRecord],
        frame_times_by_pair: dict[int, float],
        frame_detections_by_pair: dict,
        flagged_pairs: set[int],
    ):
        ...
