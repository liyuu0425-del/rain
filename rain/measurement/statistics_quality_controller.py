"""StatisticsQualityController 实现：判断一条轨迹是否够格进入"高可信统计"。

和阶段2的 Detection QC 位（touches_border/merged/too_small……）是不同
层面的东西：那一层回答"这个检测本身有没有问题"，这一层回答"这条轨迹
要不要计入数量浓度/粒径分布的正式统计"——会用到前者的结果，但多一层
业务判断（比如轨迹长度要求）。
"""

from __future__ import annotations

from ..registry import register_stats_quality_controller
from ..tracking.types import TrackRecord
from ..types import QC_NAMES
from .types import QualityVerdict

# QC_NAMES 是 {位掩码: 名字}，这里反过来建一张 {名字: 位掩码}，
# 方便配置文件里用可读的名字（"touches_border"）而不是数字位掩码。
_NAME_TO_BIT = {name: bit for bit, name in QC_NAMES.items()}

_DEFAULT_DISQUALIFYING_FLAGS = list(_NAME_TO_BIT.keys())  # 默认：任何 QC 标记都不合格，从严


@register_stats_quality_controller("qc_flag_and_track_length_controller")
class StatisticsQualityController:
    """按 QC 标记 + 轨迹长度判断一条轨迹是否进入高可信统计。

    参数（来自 config/pipeline.yaml 的 statistics.quality_controller.params）：
        disqualifying_qc_flags   : 文本名字列表，只要轨迹里有任意一个真实
                                    检测点命中其中一个标记，整条轨迹就不合格。
                                    默认是"全部标记都不合格"（从严）。
        min_track_len_for_stats  : 轨迹至少要有几个真实检测点才计入统计。
                                    默认1——哪怕只出现一帧，也是一次真实的
                                    目标观测，浓度统计不要求跨帧确认
                                    （这和阶段5测速要求>=2点不是一回事，
                                    测速需要至少两点才能算位移，计数不需要）。
    """

    def __init__(
        self,
        disqualifying_qc_flags: list[str] | None = None,
        min_track_len_for_stats: int = 1,
    ):
        names = disqualifying_qc_flags if disqualifying_qc_flags is not None else _DEFAULT_DISQUALIFYING_FLAGS
        unknown = [n for n in names if n not in _NAME_TO_BIT]
        if unknown:
            raise ValueError(f"未知的 QC 标记名字: {unknown}，可用名字: {sorted(_NAME_TO_BIT)}")
        self.disqualifying_mask = 0
        for n in names:
            self.disqualifying_mask |= _NAME_TO_BIT[n]
        self.min_track_len_for_stats = int(min_track_len_for_stats)

    def evaluate(self, track: TrackRecord) -> QualityVerdict:
        real_points = [p for p in track.points if p.detected is not None]
        reasons: list[str] = []

        if len(real_points) < self.min_track_len_for_stats:
            reasons.append("track_too_short")

        hit_flags = 0
        for p in real_points:
            hit_flags |= p.detected.qc_flags & self.disqualifying_mask
        if hit_flags:
            hit_names = [name for name, bit in _NAME_TO_BIT.items() if hit_flags & bit]
            reasons.append("qc_flags:" + "|".join(sorted(hit_names)))

        return QualityVerdict(is_high_confidence=len(reasons) == 0, reasons=reasons)
