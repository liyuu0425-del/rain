"""TrackManager 实现：轨迹创建、更新、暂时丢失、结束，稳定 track_id 管理。

这是阶段3三个模块里唯一有状态的一个——它记着所有轨迹的当前生死状态，
内部持有一个 KalmanPredictor 和一个 TrackAssociator（构造时按配置注入，
两者都可以单独替换而不用改这个类）。

生命周期规则（对应任务书的硬性约束）：
    - 卡尔曼只用来预测和缩小搜索范围：只有真实检测关联上的点才会被
      append 进 TrackRecord.points（计入轨迹长度）；丢失宽限期内的
      "仅预测"点只进 tracks.csv 做审计追溯，不计入轨迹。
    - 连续丢失超过 max_lost_frames 帧，轨迹立即结束，不再等待。
    - track_id 单调递增分配，从不复用，轨迹结束后 track_id 依然保留在
      历史记录里可查。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..registry import build_kalman_predictor, build_track_associator, register_track_manager
from .types import (
    TQC_CANDIDATE_CONFLICT,
    TQC_NEW_TRACK,
    TQC_PREDICTED_ONLY,
    FrameContext,
    KalmanState,
    StepResult,
    TrackPoint,
    TrackRecord,
)


@dataclass
class _LiveTrack:
    """内部可变状态，不对外暴露——外部只看得到 TrackRecord（不可变快照的语义）。"""

    track_id: int
    kalman_state: KalmanState
    last_device_time_s: float
    n_consecutive_lost: int = 0
    points: list[TrackPoint] = field(default_factory=list)


@register_track_manager("gated_kalman_track_manager")
class GatedKalmanTrackManager:
    """卡尔曼预测 + 门控关联 + 生命周期管理的轨迹管理器。

    参数（全部来自 config/pipeline.yaml 的 tracking.track_manager.params）：
        camera            : "a" | "b"，写进每个 TrackPoint/TrackRecord
        max_lost_frames   : 连续丢失多少帧后判定轨迹结束
        min_track_len     : 轨迹真实点数少于此值时，finalize 阶段标记为
                             "ended_short"（仍然保留，不是丢弃——短轨迹
                             不能进高可信统计，但数据本身不能被抹掉）
        kalman_predictor  : 嵌套配置，构造 KalmanPredictor
        track_associator  : 嵌套配置，构造 TrackAssociator
    """

    def __init__(
        self,
        camera: str = "a",
        max_lost_frames: int = 3,
        min_track_len: int = 2,
        kalman_predictor: dict | None = None,
        track_associator: dict | None = None,
    ):
        self.camera = camera
        self.max_lost_frames = int(max_lost_frames)
        self.min_track_len = int(min_track_len)
        self.predictor = build_kalman_predictor(
            kalman_predictor or {"name": "constant_velocity_kalman", "params": {}}
        )
        self.associator = build_track_associator(
            track_associator or {"name": "gated_nearest_neighbor", "params": {}}
        )

        self._active: dict[int, _LiveTrack] = {}
        self._ended: list[TrackRecord] = []
        self._next_id = 1

        # 汇总计数器，供 get_summary_counters() 直接返回，验收报告不用
        # 自己重新遍历一遍轨迹去算。
        self._n_lost_events = 0          # 轨迹"刚刚进入丢失状态"的次数（不是丢失帧数总和）
        self._n_candidate_conflicts = 0  # 触发 Hungarian 求解的冲突组总数
        self._n_unassociated_detections = 0  # 关联阶段未能连上现有轨迹的检测总数
        # （这些检测随后会各自新建轨迹，不代表数据丢失；数值越高，
        #   说明越多检测没能延续已有轨迹——可能是新雨滴进入视场，
        #   也可能是门控参数偏紧导致该续上的没续上，需要结合真实数据复核）

    def step(self, frame: FrameContext) -> StepResult:
        track_ids = list(self._active.keys())

        # --- 1. 预测：把每条活跃轨迹外推到这一帧的真实设备时间 --------------
        t0 = time.perf_counter()
        predicted_positions: list[tuple[float, float]] = []
        for tid in track_ids:
            live = self._active[tid]
            dt = frame.device_time_s - live.last_device_time_s
            live.kalman_state = self.predictor.predict(live.kalman_state, dt)
            predicted_positions.append((live.kalman_state.cx, live.kalman_state.cy))
        predict_ms = (time.perf_counter() - t0) * 1000.0

        # --- 2. 关联：门控 + 最近邻/匈牙利 ----------------------------------
        t0 = time.perf_counter()
        result = self.associator.associate(predicted_positions, frame.detections)
        associate_ms = (time.perf_counter() - t0) * 1000.0

        self._n_candidate_conflicts += result.n_candidate_conflicts

        # --- 3. 更新/创建/丢失/结束：轨迹生命周期管理 ------------------------
        t0 = time.perf_counter()
        points: list[TrackPoint] = []
        matched_track_local_idx: set[int] = set()

        for local_i, local_j in result.matches:
            tid = track_ids[local_i]
            live = self._active[tid]
            det = frame.detections[local_j]

            live.kalman_state = self.predictor.update(live.kalman_state, det.cx, det.cy)
            live.n_consecutive_lost = 0
            live.last_device_time_s = frame.device_time_s

            method = result.match_methods[(local_i, local_j)]
            qc = det.qc_flags | (TQC_CANDIDATE_CONFLICT if method == "hungarian" else 0)
            pt = TrackPoint(
                track_id=tid, camera=self.camera, pair_index=frame.pair_index,
                device_time_s=frame.device_time_s, detected=det,
                predicted_cx=predicted_positions[local_i][0],
                predicted_cy=predicted_positions[local_i][1],
                association_method=method, quality_flags=qc,
            )
            live.points.append(pt)
            points.append(pt)
            matched_track_local_idx.add(local_i)

        # 未关联上的活跃轨迹：丢失计数 +1，超限则结束；否则产出一个
        # "仅预测"的审计点（不计入轨迹长度）。
        for local_i, tid in enumerate(track_ids):
            if local_i in matched_track_local_idx:
                continue
            live = self._active[tid]
            live.n_consecutive_lost += 1
            if live.n_consecutive_lost == 1:
                self._n_lost_events += 1

            if live.n_consecutive_lost > self.max_lost_frames:
                self._finalize_one(tid)
                continue

            pt = TrackPoint(
                track_id=tid, camera=self.camera, pair_index=frame.pair_index,
                device_time_s=frame.device_time_s, detected=None,
                predicted_cx=predicted_positions[local_i][0],
                predicted_cy=predicted_positions[local_i][1],
                association_method="predicted_only", quality_flags=TQC_PREDICTED_ONLY,
            )
            points.append(pt)

        # 未关联上任何轨迹的检测：新建轨迹。卡尔曼状态用检测位置初始化，
        # "预测位置"用检测本身占位（新轨迹还没有预测这回事）。
        for local_j in result.unmatched_detections:
            det = frame.detections[local_j]
            tid = self._next_id
            self._next_id += 1

            kstate = self.predictor.init_state(det.cx, det.cy)
            live = _LiveTrack(track_id=tid, kalman_state=kstate, last_device_time_s=frame.device_time_s)

            pt = TrackPoint(
                track_id=tid, camera=self.camera, pair_index=frame.pair_index,
                device_time_s=frame.device_time_s, detected=det,
                predicted_cx=det.cx, predicted_cy=det.cy,
                association_method="new_track", quality_flags=det.qc_flags | TQC_NEW_TRACK,
            )
            live.points.append(pt)
            self._active[tid] = live
            points.append(pt)

        self._n_unassociated_detections += len(result.unmatched_detections)
        manage_ms = (time.perf_counter() - t0) * 1000.0

        return StepResult(points=points, predict_ms=predict_ms, associate_ms=associate_ms, manage_ms=manage_ms)

    def finalize(self) -> list[TrackRecord]:
        for tid in list(self._active.keys()):
            self._finalize_one(tid)
        return sorted(self._ended, key=lambda t: t.track_id)

    def get_summary_counters(self) -> dict:
        return {
            "n_lost_events": self._n_lost_events,
            "n_candidate_conflicts": self._n_candidate_conflicts,
            "n_unassociated_detections": self._n_unassociated_detections,
        }

    def _finalize_one(self, track_id: int) -> None:
        live = self._active.pop(track_id)
        status = "ended" if len(live.points) >= self.min_track_len else "ended_short"
        self._ended.append(
            TrackRecord(track_id=live.track_id, camera=self.camera, points=live.points, status=status)
        )
