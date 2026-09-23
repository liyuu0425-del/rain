"""阶段3数据契约：卡尔曼状态、单帧上下文、轨迹点、轨迹记录、关联结果。

和 `rain/types.py`（Detection/Track，服务于老 Demo）、`rain/data/types.py`
（服务于阶段1）是同一类东西，单独一份是为了不牵动其它阶段已经跑通的代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Detection

# ---------------------------------------------------------------------------
# 轨迹级质量标记（Track-level QC）。
#
# 和 rain/types.py 里 Detection 的 QC_* 是两套独立的位标记——那一套描述
# "这个检测本身有没有问题"（触边/太小/形状异常……），这一套描述
# "这个轨迹点是怎么来的"（是真检测关联上的，还是卡尔曼预测撑住的）。
# 两者互不覆盖，写进 TrackPoint.quality_flags 时按位或在一起即可区分来源。
# ---------------------------------------------------------------------------
TQC_OK = 0
TQC_PREDICTED_ONLY = 1 << 8   # 本帧没有真实检测关联，只有卡尔曼预测撑住（不计入轨迹长度）
TQC_NEW_TRACK = 1 << 9        # 该点是新建轨迹的第一个点
TQC_CANDIDATE_CONFLICT = 1 << 10  # 该点的关联经过了 Hungarian 冲突消解（而不是直接最近邻）

TQC_NAMES = {
    TQC_PREDICTED_ONLY: "predicted_only",
    TQC_NEW_TRACK: "new_track",
    TQC_CANDIDATE_CONFLICT: "candidate_conflict",
}


def track_qc_to_text(flags: int) -> str:
    names = [name for bit, name in TQC_NAMES.items() if flags & bit]
    return "|".join(names) if names else "ok"


@dataclass
class KalmanState:
    """卡尔曼滤波器的内部状态：位置+速度的估计值和不确定度。

    x : 状态向量 [cx, cy, vx, vy]，形状 (4,)。位置像素，速度像素/秒。
    P : 状态协方差矩阵，形状 (4,4)，对角线是每个状态分量的方差，
        非对角线是分量之间的协方差。数值越大代表这个估计越不确定。
    """

    x: np.ndarray
    P: np.ndarray

    @property
    def cx(self) -> float:
        return float(self.x[0])

    @property
    def cy(self) -> float:
        return float(self.x[1])

    @property
    def vx(self) -> float:
        return float(self.x[2])

    @property
    def vy(self) -> float:
        return float(self.x[3])


@dataclass
class FrameContext:
    """跟踪器逐帧消费的输入单元：一帧的真实设备时间 + 这一帧的检测结果。

    device_time_s 必须来自 a_device_relative_us（A路）或 b_device_relative_us
    （B路）换算而来（除以1e6转成秒），不得来自文件名毫秒、host接收时间、
    帧号或固定帧率——这是任务书对阶段3的硬性约束，在
    pipeline/stage3_tracking.py 组装 FrameContext 时保证。
    """

    pair_index: int
    device_time_s: float
    detections: list[Detection]


@dataclass
class TrackPoint:
    """tracks.csv 的一行：某条轨迹在某一帧的完整记录。

    detected 为 None 时，代表这一帧没有真实检测关联上这条轨迹，
    只有 predicted_cx/predicted_cy 这个卡尔曼预测位置撑着（轨迹还在
    max_lost_frames 的宽限期内，还没被判定结束）。这类点不计入
    "轨迹长度"统计，只是审计意义上保留，让人看清楚轨迹在丢失期间
    经过了哪里，如果之后重新关联上了，能看到中间空档发生了什么。
    """

    track_id: int
    camera: str  # "a" | "b"
    pair_index: int
    device_time_s: float
    detected: Detection | None
    predicted_cx: float
    predicted_cy: float
    association_method: str  # "new_track" | "gated_nn" | "hungarian" | "predicted_only"
    quality_flags: int = TQC_OK

    @property
    def is_real_detection(self) -> bool:
        return self.detected is not None


@dataclass
class TrackRecord:
    """一条完整轨迹：track_id 一旦分配，终生不变、不复用。"""

    track_id: int
    camera: str
    points: list[TrackPoint] = field(default_factory=list)  # 只含真实检测关联上的点
    status: str = "active"  # active | ended | ended_short

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def pair_index_span(self) -> tuple[int, int] | None:
        if not self.points:
            return None
        idxs = [p.pair_index for p in self.points]
        return (min(idxs), max(idxs))


@dataclass
class AssociationResult:
    """TrackAssociator.associate() 的返回值。

    matches 里的下标是"预测位置列表"和"检测列表"各自的下标（从0开始），
    调用方负责把下标翻译回真实的 track_id / Detection。
    """

    matches: list[tuple[int, int]]
    match_costs: dict[tuple[int, int], float]
    match_methods: dict[tuple[int, int], str]  # "gated_nn" | "hungarian"
    unmatched_tracks: list[int]
    unmatched_detections: list[int]
    n_candidate_conflicts: int


@dataclass
class StepResult:
    """TrackManager.step() 的返回值：这一帧产出的轨迹点 + 内部阶段耗时。

    耗时拆成 predict/associate/manage 三段，供性能统计（阶段3要求的
    "检测、预测、关联、轨迹管理" 四段耗时里除"检测"以外的三段）。
    "检测"耗时在 TrackManager 之外，由调用方（pipeline/stage3_tracking.py）
    单独计时——检测不是跟踪模块的职责。
    """

    points: list[TrackPoint]
    predict_ms: float
    associate_ms: float
    manage_ms: float
