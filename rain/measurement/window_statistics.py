"""WindowStatistics 实现：按时间窗口切分 + 质量门限 + 数量浓度 + 粒径分布。

这是阶段6的组合模块——本身不包含具体的门限判断/浓度公式/分箱算法，
只负责：
    1. 把整段运行时间切成固定长度的窗口；
    2. 把每条轨迹按"首次出现的真实检测点时间"分配到唯一一个窗口
       （这一步是避免重复计数的关键：一条轨迹不管跨了多少帧，
       只在它第一次被看到的那个窗口里计数一次，不会因为轨迹长
       就被多算）；
    3. 调用 StatisticsQualityController 判断每条轨迹合不合格；
    4. 调用 NumberConcentrationEstimator 算名义浓度；
    5. 全局（跨全部窗口）调用 DropSizeDistributionEstimator 算像素粒径分布。

!!! 重要 !!!
这个类产出的是 SingleCameraProxyStat（单路调试统计），不是最终浓度。
590.4 cm³ 是双路共同观测体积，单路（A或B单独）目标数不能代表共同空间
里的目标数——真正对应这块体积的统计是 StereoCommonVolumeStatistics
（rain/measurement/stereo_common_volume_statistics.py），依赖阶段4的
双路匹配结果，而阶段4当前没有标定文件，恒为0个已验证匹配。
"""

from __future__ import annotations

import math
from ..registry import (
    build_concentration_estimator,
    build_size_distribution_estimator,
    build_stats_quality_controller,
    register_window_statistics,
)
from ..tracking.types import TrackRecord
from ..types import Detection
from .types import DropSizeDistributionResult, SingleCameraProxyStat


def compute_window_bounds(t_min: float, t_max: float, window_seconds: float) -> list[tuple[int, float, float]]:
    """把 [t_min, t_max] 切成固定长度的窗口，返回 [(窗口序号, 起, 止), ...]。

    最后一个窗口右边界可能超过 t_max（除非窗口长度恰好整除总时长），
    调用方判断"某个时刻属于哪个窗口"时，最后一个窗口要把 t==t_max
    这个边界点也算进去（左闭右开会漏掉最后一个点）。

    这是 SingleCameraProxyStat 和 StereoCommonVolumeStat 共用的切窗口逻辑
    ——两条统计线用同一套窗口边界，才能在报告里对得上号，所以抽成
    独立函数而不是在两个类里各写一遍。
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds 必须大于0，收到 {window_seconds}")

    # 用 ceil 而不是"floor+1"：span 恰好是 window_seconds 整数倍时（比如
    # span=1.0, window=1.0），floor+1 会多切出一个必然是空的尾部窗口
    # （因为 t_max 已经被"最后一个窗口含右端点"这条规则纳入前一个窗口了）。
    # round(...,9) 是防止浮点误差把 2.0 算成 1.9999999998 或 2.0000000002，
    # 导致 ceil 结果多一或少一。
    span = t_max - t_min
    n_windows = max(1, math.ceil(round(span / window_seconds, 9)))
    return [(w, t_min + w * window_seconds, t_min + (w + 1) * window_seconds) for w in range(n_windows)]


def time_in_window(t: float, w_start: float, w_end: float, is_last_window: bool, t_max: float) -> bool:
    """判断时刻 t 是否属于 [w_start, w_end) 这个窗口（最后一个窗口含右端点）。"""
    return w_start <= t < w_end or (is_last_window and t == t_max)


@register_window_statistics("fixed_length_window_statistics")
class WindowStatistics:
    """固定长度时间窗口统计（单路调试统计，见模块开头说明）。

    参数（来自 config/pipeline.yaml 的 statistics 段）：
        window_seconds              : 每个时间窗口的长度（秒）
        quality_controller           : 嵌套配置，构造 StatisticsQualityController
        concentration_estimator      : 嵌套配置，构造 NumberConcentrationEstimator
        size_distribution             : 嵌套配置，构造 DropSizeDistributionEstimator
        spatial_calibration_status   : 空间标定可信状态文本，从配置读取，
                                       默认 "spatial_calibration_unavailable"
    """

    def __init__(
        self,
        window_seconds: float = 0.1,
        quality_controller: dict | None = None,
        concentration_estimator: dict | None = None,
        size_distribution: dict | None = None,
        spatial_calibration_status: str = "spatial_calibration_unavailable",
    ):
        if window_seconds <= 0:
            raise ValueError(f"window_seconds 必须大于0，收到 {window_seconds}")
        self.window_seconds = float(window_seconds)
        self.spatial_calibration_status = str(spatial_calibration_status)
        self.quality_controller = build_stats_quality_controller(
            quality_controller or {"name": "qc_flag_and_track_length_controller", "params": {}}
        )
        self.concentration_estimator = build_concentration_estimator(
            concentration_estimator or {"name": "nominal_volume_concentration", "params": {}}
        )
        self.size_distribution_estimator = build_size_distribution_estimator(
            size_distribution or {"name": "pixel_size_histogram", "params": {}}
        )

    def compute(
        self,
        camera: str,
        tracks: list[TrackRecord],
        frame_times_by_pair: dict[int, float],
        frame_detections_by_pair: dict[int, list[Detection]],
        flagged_pairs: set[int],
    ) -> tuple[list[SingleCameraProxyStat], DropSizeDistributionResult]:
        if not frame_times_by_pair:
            # 没有任何帧信息，连窗口都切不出来——返回空结果而不是抛异常，
            # 调用方（空批次/空相机）应该能拿到一个"合法的空结果"。
            empty_dist = self.size_distribution_estimator.estimate(camera, [])
            return [], empty_dist

        t_min = min(frame_times_by_pair.values())
        t_max = max(frame_times_by_pair.values())
        bounds = compute_window_bounds(t_min, t_max, self.window_seconds)
        n_windows = len(bounds)

        # 先给每条轨迹判定质量、算出"首次出现时间"和"代表粒径"，
        # 只算一次，后面按窗口分组时直接用，不重复算。
        verdicts = {t.track_id: self.quality_controller.evaluate(t) for t in tracks}
        entry_time = {}
        rep_size_px = {}
        for t in tracks:
            real_points = [p for p in t.points if p.detected is not None]
            if not real_points:
                continue
            real_points.sort(key=lambda p: p.device_time_s)
            entry_time[t.track_id] = real_points[0].device_time_s
            sizes = sorted(p.detected.d_eq_px for p in real_points)
            rep_size_px[t.track_id] = sizes[len(sizes) // 2]  # 中位数，比首末点更稳健

        rows: list[SingleCameraProxyStat] = []
        all_valid_sizes: list[float] = []

        for w, w_start, w_end in bounds:
            is_last = w == n_windows - 1
            member_pairs = [p for p, t in frame_times_by_pair.items() if time_in_window(t, w_start, w_end, is_last, t_max)]
            sample_count = sum(len(frame_detections_by_pair.get(p, [])) for p in member_pairs)
            time_status = "degraded" if any(p in flagged_pairs for p in member_pairs) else "ok"

            valid_count = 0
            excluded_count = 0
            for t in tracks:
                et = entry_time.get(t.track_id)
                if et is None or not time_in_window(et, w_start, w_end, is_last, t_max):
                    continue
                if verdicts[t.track_id].is_high_confidence:
                    valid_count += 1
                    all_valid_sizes.append(rep_size_px[t.track_id])
                else:
                    excluded_count += 1

            conc = self.concentration_estimator.estimate(valid_count)

            if sample_count == 0:
                quality_status = "empty"
            elif valid_count == 0:
                quality_status = "low_confidence"
            else:
                quality_status = "nominal"

            rows.append(SingleCameraProxyStat(
                camera=camera, window_index=w,
                window_start_s=w_start, window_end_s=w_end,
                sample_count=sample_count,
                valid_target_count=valid_count,
                excluded_target_count=excluded_count,
                size_bin_edges_px=list(self.size_distribution_estimator.bin_edges_px),
                nominal_concentration_m3=conc.concentration_m3,
                volume_status=conc.volume_status,
                time_status=time_status,
                spatial_calibration_status=self.spatial_calibration_status,
                quality_status=quality_status,
                is_single_camera_proxy=True,
                usable_for_final_concentration=False,
            ))

        size_dist = self.size_distribution_estimator.estimate(camera, all_valid_sizes)
        return rows, size_dist
