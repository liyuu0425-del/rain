"""StereoCommonVolumeStatistics 实现：590.4 cm³ 名义体积真正对应的统计。

和 WindowStatistics（单路代理统计）的核心区别：这里的 N 只能来自双路
匹配成功（阶段4 StereoMatcher 判定 status="matched"）的目标数，绝不
拿 A 路或 B 路单独的目标数顶替——单路看到的目标不能证明它真的落在
两台正交相机的共同视场里，只有双路都确认，才是"共同空间"意义上的目标。

当前现状（阶段4已确认）：没有真实双相机标定文件，StereoMatcher 在这种
情况下让所有候选都判定不通过，真实数据上 n_matched_pairs 恒为0。
这个类因此把"没有标定"和"标定了但这一刻真的没有匹配目标"严格区分开：
    calibration_available=False -> concentration_status="unavailable"
    calibration_available=True 且 n_matched_pairs=0 -> concentration_status="computed"，
        数值就是0（这是一个真实的、可信的"0浓度"测量结果）
"""

from __future__ import annotations

from ..registry import build_concentration_estimator, register_stereo_common_volume_statistics
from .types import StereoCommonVolumeStat
from .window_statistics import compute_window_bounds, time_in_window


@register_stereo_common_volume_statistics("gated_stereo_common_volume_statistics")
class StereoCommonVolumeStatistics:
    """双路共同体积统计。

    参数（来自 config/pipeline.yaml 的 statistics.stereo_common_volume 段）：
        window_seconds              : 时间窗口长度（秒），应该和单路代理统计
                                       用同一个值，两条产出线才能对得上号。
        concentration_estimator      : 嵌套配置，构造 NumberConcentrationEstimator
                                       （和单路代理统计共用同一个 C=N/V 公式实现，
                                       只是喂进去的 N 不同）。
        spatial_calibration_status   : 从配置读取，默认 "spatial_calibration_unavailable"。
    """

    def __init__(
        self,
        window_seconds: float = 0.1,
        concentration_estimator: dict | None = None,
        spatial_calibration_status: str = "spatial_calibration_unavailable",
    ):
        if window_seconds <= 0:
            raise ValueError(f"window_seconds 必须大于0，收到 {window_seconds}")
        self.window_seconds = float(window_seconds)
        self.spatial_calibration_status = str(spatial_calibration_status)
        self.concentration_estimator = build_concentration_estimator(
            concentration_estimator or {"name": "nominal_volume_concentration", "params": {}}
        )

    def compute(
        self,
        calibration_available: bool,
        t_min: float,
        t_max: float,
        matched_pair_entry_times: list[float],
    ) -> list[StereoCommonVolumeStat]:
        """
        参数：
            calibration_available     : 阶段4的标定是否可用（来自
                                         CalibrationLoader.load(...) is not None）。
            t_min, t_max               : 用哪段时间切窗口（建议用A路
                                         device_relative_us，阶段1已确认的
                                         主记录时间基准）。
            matched_pair_entry_times   : 每一个双路匹配成功(status="matched")的
                                         目标，其对应帧的设备时间——每个匹配对
                                         只出现一次（阶段4的匹配本身就是
                                         "一个pair_index一次尝试"，不存在同一个
                                         匹配对跨帧重复的问题，不需要额外去重）。
                                         calibration_available=False 时这个列表
                                         必须是空列表（调用方不得在没有标定时
                                         传入任何"看起来是匹配"的数据）。
        """
        if not calibration_available and matched_pair_entry_times:
            raise ValueError(
                "calibration_available=False 但传入了非空的 matched_pair_entry_times——"
                "没有标定时不能有任何'匹配成功'的目标，这是数据不一致，不能悄悄接受。"
            )

        bounds = compute_window_bounds(t_min, t_max, self.window_seconds)
        n_windows = len(bounds)
        rows: list[StereoCommonVolumeStat] = []

        for w, w_start, w_end in bounds:
            is_last = w == n_windows - 1

            if not calibration_available:
                rows.append(StereoCommonVolumeStat(
                    window_index=w, window_start_s=w_start, window_end_s=w_end,
                    n_matched_pairs=0, valid_target_count=0,
                    concentration_m3=None, concentration_status="unavailable",
                    volume_status=self.concentration_estimator.volume_status,
                    stereo_match_status="calibration_unavailable",
                    spatial_calibration_status=self.spatial_calibration_status,
                    quality_status="unavailable",
                ))
                continue

            n = sum(1 for t in matched_pair_entry_times if time_in_window(t, w_start, w_end, is_last, t_max))
            conc = self.concentration_estimator.estimate(n)
            rows.append(StereoCommonVolumeStat(
                window_index=w, window_start_s=w_start, window_end_s=w_end,
                n_matched_pairs=n, valid_target_count=n,
                concentration_m3=conc.concentration_m3, concentration_status="computed",
                volume_status=conc.volume_status,
                stereo_match_status="ok",
                spatial_calibration_status=self.spatial_calibration_status,
                quality_status="nominal" if n > 0 else "empty",
            ))

        return rows
