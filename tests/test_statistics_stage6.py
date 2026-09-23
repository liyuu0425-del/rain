"""阶段6模块测试：StatisticsQualityController / NumberConcentrationEstimator /
DropSizeDistributionEstimator / WindowStatistics。

任务书要求覆盖的场景：
    空窗口、单目标、重复轨迹不重复计数、低质量目标排除、体积单位换算、
    像素粒径统计、未标定时不输出毫米粒径和最终雨强。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  触发所有实现的注册
from rain.measurement.drop_size_distribution_estimator import DropSizeDistributionEstimator
from rain.measurement.number_concentration_estimator import NumberConcentrationEstimator
from rain.measurement.statistics_quality_controller import StatisticsQualityController
from rain.measurement.window_statistics import WindowStatistics
from rain.tracking.types import TrackPoint, TrackRecord
from rain.types import QC_MERGED, QC_OK, QC_TOUCHES_BORDER, Detection


def _det(cx=100.0, cy=100.0, area=50.0, qc_flags=QC_OK) -> Detection:
    return Detection(
        frame_idx=0, cx=cx, cy=cy, area_px=area,
        bbox=(int(cx - 5), int(cy - 5), 10, 10), d_eq_px=Detection.d_eq_from_area(area),
        qc_flags=qc_flags,
    )


def _point(track_id, cam, pair_index, t, det) -> TrackPoint:
    return TrackPoint(
        track_id=track_id, camera=cam, pair_index=pair_index, device_time_s=t,
        detected=det, predicted_cx=det.cx, predicted_cy=det.cy,
        association_method="gated_nn", quality_flags=det.qc_flags,
    )


def _track(track_id, cam, points, status="ended") -> TrackRecord:
    return TrackRecord(track_id=track_id, camera=cam, points=points, status=status)


# ---------------------------------------------------------------------------
# NumberConcentrationEstimator —— "体积单位换算"
# ---------------------------------------------------------------------------


def test_concentration_formula_and_unit():
    est = NumberConcentrationEstimator(volume_cm3=590.4, volume_m3=0.0005904)
    result = est.estimate(n_valid_targets=10)

    assert result.concentration_m3 == pytest.approx(10 / 0.0005904)
    assert result.unit == "m^-3"
    assert result.volume_status == "nominal_unvalidated"


def test_concentration_volume_cm3_to_m3_conversion_consistency():
    """590.4 cm³ 必须精确等于 0.0005904 m³，这是单位换算本身要测的东西。"""
    est = NumberConcentrationEstimator(volume_cm3=590.4, volume_m3=0.0005904)
    assert est.volume_m3 == pytest.approx(est.volume_cm3 * 1e-6, abs=1e-12)


def test_concentration_rejects_inconsistent_volume_config():
    with pytest.raises(ValueError):
        NumberConcentrationEstimator(volume_cm3=590.4, volume_m3=999.0)  # 明显对不上


def test_concentration_zero_targets_is_zero_not_none():
    est = NumberConcentrationEstimator()
    result = est.estimate(n_valid_targets=0)
    assert result.concentration_m3 == 0.0


def test_concentration_rejects_negative_count():
    est = NumberConcentrationEstimator()
    with pytest.raises(ValueError):
        est.estimate(n_valid_targets=-1)


# ---------------------------------------------------------------------------
# DropSizeDistributionEstimator —— "像素粒径统计" + "未标定时不叫毫米雨滴谱"
# ---------------------------------------------------------------------------


def test_size_distribution_bins_pixel_sizes():
    est = DropSizeDistributionEstimator(bin_edges_px=[0, 10, 20, 30])
    result = est.estimate("a", [5.0, 15.0, 15.5, 25.0])

    assert result.bin_edges_px == [0, 10, 20, 30]
    assert result.counts == [1, 2, 1]
    assert result.n_total == 4


def test_size_distribution_spatial_status_and_note_forbid_mm_claim():
    est = DropSizeDistributionEstimator()
    result = est.estimate("a", [5.0, 10.0])
    assert result.spatial_calibration_status == "spatial_calibration_unavailable"
    assert "毫米" in result.note or "mm" not in result.note.lower()
    assert "雨滴谱" not in result.note or "非正式" in result.note  # 只能是"非正式"表述


def test_size_distribution_empty_sizes_returns_zero_counts():
    est = DropSizeDistributionEstimator(bin_edges_px=[0, 10, 20])
    result = est.estimate("a", [])
    assert result.counts == [0, 0]
    assert result.n_total == 0


def test_size_distribution_rejects_non_increasing_edges():
    with pytest.raises(ValueError):
        DropSizeDistributionEstimator(bin_edges_px=[0, 10, 5, 20])


# ---------------------------------------------------------------------------
# StatisticsQualityController —— "低质量目标排除"
# ---------------------------------------------------------------------------


def test_quality_controller_passes_clean_track():
    controller = StatisticsQualityController()
    track = _track(1, "a", [_point(1, "a", 0, 0.0, _det(qc_flags=QC_OK))])
    verdict = controller.evaluate(track)
    assert verdict.is_high_confidence is True
    assert verdict.reasons == []


def test_quality_controller_rejects_flagged_detection():
    controller = StatisticsQualityController()  # 默认：任何QC标记都不合格
    track = _track(1, "a", [_point(1, "a", 0, 0.0, _det(qc_flags=QC_TOUCHES_BORDER))])
    verdict = controller.evaluate(track)
    assert verdict.is_high_confidence is False
    assert any("touches_border" in r for r in verdict.reasons)


def test_quality_controller_custom_disqualifying_flags():
    """只把 merged 设成不合格标记时，touches_border 应该被放行。"""
    controller = StatisticsQualityController(disqualifying_qc_flags=["merged"])
    track = _track(1, "a", [_point(1, "a", 0, 0.0, _det(qc_flags=QC_TOUCHES_BORDER))])
    verdict = controller.evaluate(track)
    assert verdict.is_high_confidence is True


def test_quality_controller_rejects_short_track():
    controller = StatisticsQualityController(min_track_len_for_stats=3)
    track = _track(1, "a", [_point(1, "a", 0, 0.0, _det())])  # 只有1个点
    verdict = controller.evaluate(track)
    assert verdict.is_high_confidence is False
    assert "track_too_short" in verdict.reasons


def test_quality_controller_unknown_flag_name_raises():
    with pytest.raises(ValueError):
        StatisticsQualityController(disqualifying_qc_flags=["这个标记不存在"])


# ---------------------------------------------------------------------------
# WindowStatistics —— "空窗口" "单目标" "重复轨迹不重复计数"
# ---------------------------------------------------------------------------


def _ws(**overrides) -> WindowStatistics:
    params = dict(window_seconds=0.1)
    params.update(overrides)
    return WindowStatistics(**params)


def test_window_statistics_empty_window():
    """没有任何检测、任何帧的极端情况，必须返回合法空结果而不是崩溃。"""
    ws = _ws()
    rows, dist = ws.compute("a", tracks=[], frame_times_by_pair={}, frame_detections_by_pair={}, flagged_pairs=set())
    assert rows == []
    assert dist.n_total == 0
    assert dist.counts == [0] * (len(dist.bin_edges_px) - 1)


def test_window_statistics_window_with_no_detections_marked_empty():
    ws = _ws(window_seconds=1.0)
    frame_times = {0: 0.0, 1: 0.1}
    rows, _ = ws.compute("a", tracks=[], frame_times_by_pair=frame_times,
                          frame_detections_by_pair={}, flagged_pairs=set())
    assert len(rows) == 1
    assert rows[0].quality_status == "empty"
    assert rows[0].sample_count == 0
    assert rows[0].valid_target_count == 0
    assert rows[0].nominal_concentration_m3 == 0.0


def test_window_statistics_single_target():
    """单目标：一帧一个检测，构成一条只有1个点的轨迹，应该被计为1个有效目标。"""
    ws = _ws(window_seconds=1.0)
    det = _det(cx=100, cy=100, area=50)
    track = _track(1, "a", [_point(1, "a", 0, 0.0, det)])
    frame_times = {0: 0.0}
    frame_dets = {0: [det]}

    rows, dist = ws.compute("a", tracks=[track], frame_times_by_pair=frame_times,
                             frame_detections_by_pair=frame_dets, flagged_pairs=set())

    assert len(rows) == 1
    assert rows[0].sample_count == 1
    assert rows[0].valid_target_count == 1
    assert rows[0].excluded_target_count == 0
    assert rows[0].quality_status == "nominal"
    assert dist.n_total == 1


def test_window_statistics_multi_frame_track_counted_once_not_per_point():
    """核心要求：一条跨多帧的轨迹在浓度计算里只能算1个目标，不是按点数算。"""
    ws = _ws(window_seconds=1.0)
    dets = [_det(cx=100, cy=100 + i * 10, area=50) for i in range(5)]  # 5帧同一颗雨滴
    points = [_point(1, "a", i, i * 0.01, d) for i, d in enumerate(dets)]
    track = _track(1, "a", points)

    frame_times = {i: i * 0.01 for i in range(5)}
    frame_dets = {i: [dets[i]] for i in range(5)}

    rows, dist = ws.compute("a", tracks=[track], frame_times_by_pair=frame_times,
                             frame_detections_by_pair=frame_dets, flagged_pairs=set())

    # sample_count 允许按原始检测数累加(=5，诊断用途)，
    # 但 valid_target_count 必须只有1个——这才是防止"跨帧重复计数"的关键断言。
    assert rows[0].sample_count == 5
    assert rows[0].valid_target_count == 1
    assert dist.n_total == 1  # 粒径分布同样只应该收到1个代表值，不是5个


def test_window_statistics_two_tracks_two_targets():
    ws = _ws(window_seconds=1.0)
    det1 = _det(cx=50, cy=50, area=30)
    det2 = _det(cx=500, cy=500, area=80)
    track1 = _track(1, "a", [_point(1, "a", 0, 0.0, det1)])
    track2 = _track(2, "a", [_point(2, "a", 0, 0.0, det2)])

    frame_times = {0: 0.0}
    frame_dets = {0: [det1, det2]}

    rows, dist = ws.compute("a", tracks=[track1, track2], frame_times_by_pair=frame_times,
                             frame_detections_by_pair=frame_dets, flagged_pairs=set())

    assert rows[0].valid_target_count == 2
    assert dist.n_total == 2


def test_window_statistics_low_quality_target_excluded_but_retained_in_count():
    """低质量目标不能进浓度分子，但要出现在 excluded_target_count 里，不能被吞掉。"""
    ws = _ws(window_seconds=1.0)
    good_det = _det(cx=50, cy=50, area=30, qc_flags=QC_OK)
    bad_det = _det(cx=500, cy=500, area=80, qc_flags=QC_MERGED)
    track1 = _track(1, "a", [_point(1, "a", 0, 0.0, good_det)])
    track2 = _track(2, "a", [_point(2, "a", 0, 0.0, bad_det)])

    frame_times = {0: 0.0}
    frame_dets = {0: [good_det, bad_det]}

    rows, dist = ws.compute("a", tracks=[track1, track2], frame_times_by_pair=frame_times,
                             frame_detections_by_pair=frame_dets, flagged_pairs=set())

    assert rows[0].valid_target_count == 1
    assert rows[0].excluded_target_count == 1
    assert rows[0].sample_count == 2  # 原始检测数没有因为质量不合格而减少
    assert dist.n_total == 1  # 粒径分布只收合格目标


def test_window_statistics_track_counted_in_its_entry_window_only():
    """轨迹跨越两个窗口时，只应该在"首次出现"的那个窗口里计数一次。"""
    ws = _ws(window_seconds=0.1)
    dets = [_det(cx=100, cy=100 + i * 10) for i in range(3)]
    points = [
        _point(1, "a", 0, 0.05, dets[0]),   # 落在窗口0 [0,0.1)
        _point(1, "a", 1, 0.15, dets[1]),   # 落在窗口1 [0.1,0.2)
        _point(1, "a", 2, 0.25, dets[2]),   # 落在窗口2 [0.2,0.3)
    ]
    track = _track(1, "a", points)
    frame_times = {0: 0.05, 1: 0.15, 2: 0.25}
    frame_dets = {0: [dets[0]], 1: [dets[1]], 2: [dets[2]]}

    rows, _ = ws.compute("a", tracks=[track], frame_times_by_pair=frame_times,
                          frame_detections_by_pair=frame_dets, flagged_pairs=set())

    valid_counts = [r.valid_target_count for r in rows]
    assert sum(valid_counts) == 1  # 三个窗口加起来只计一次，不是三次
    assert rows[0].valid_target_count == 1  # 计在了首次出现的窗口0


def test_window_statistics_time_status_degraded_when_flagged():
    ws = _ws(window_seconds=1.0)
    frame_times = {0: 0.0, 1: 0.1}
    rows, _ = ws.compute("a", tracks=[], frame_times_by_pair=frame_times,
                          frame_detections_by_pair={}, flagged_pairs={1})
    assert rows[0].time_status == "degraded"


def test_window_statistics_spatial_and_volume_status_always_present():
    ws = _ws()
    rows, dist = ws.compute("a", tracks=[], frame_times_by_pair={0: 0.0},
                             frame_detections_by_pair={}, flagged_pairs=set())
    assert rows[0].spatial_calibration_status == "spatial_calibration_unavailable"
    assert rows[0].volume_status == "nominal_unvalidated"
    assert dist.spatial_calibration_status == "spatial_calibration_unavailable"


# ---------------------------------------------------------------------------
# 未标定时不输出毫米粒径/最终雨强
# ---------------------------------------------------------------------------


def test_no_module_outputs_mm_or_rain_rate_fields():
    """穷举所有阶段6结果对象的字段名，不能出现 mm 粒径或雨强相关字段。
    这是一个"结构性"测试——防止未来有人不小心加了个 diameter_mm 字段。"""
    import dataclasses

    from rain.measurement.types import ConcentrationResult, DropSizeDistributionResult, WindowStatRow

    forbidden_substrings = ["_mm", "rain_rate", "rainfall_rate", "intensity_mm", "m_per_s", "mm_per"]
    for cls in (ConcentrationResult, DropSizeDistributionResult, WindowStatRow):
        field_names = [f.name for f in dataclasses.fields(cls)]
        for name in field_names:
            for forbidden in forbidden_substrings:
                assert forbidden not in name, f"{cls.__name__}.{name} 疑似输出了未标定的物理量"



# ---------------------------------------------------------------------------
# 体积适用范围修正：single_camera_proxy vs stereo_common_volume
# ---------------------------------------------------------------------------


def test_single_camera_proxy_rows_marked_informal():
    """WindowStatistics 产出的每一行都必须显式标注"非正式、不可用于最终浓度"，
    不依赖看报告的人自己记住 590.4cm³ 是双路共同体积这件事。"""
    ws = _ws(window_seconds=1.0)
    det = _det(cx=100, cy=100, area=50)
    track = _track(1, "a", [_point(1, "a", 0, 0.0, det)])
    rows, _ = ws.compute("a", tracks=[track], frame_times_by_pair={0: 0.0},
                          frame_detections_by_pair={0: [det]}, flagged_pairs=set())

    assert len(rows) == 1
    assert rows[0].is_single_camera_proxy is True
    assert rows[0].usable_for_final_concentration is False


def test_number_concentration_estimator_volume_status_configurable():
    """volume_status 必须能通过构造参数改变，不是代码里焊死的常量。"""
    est_default = NumberConcentrationEstimator()
    assert est_default.estimate(5).volume_status == "nominal_unvalidated"

    est_custom = NumberConcentrationEstimator(volume_status="validated_by_bench_test")
    assert est_custom.estimate(5).volume_status == "validated_by_bench_test"


def test_size_distribution_spatial_status_configurable():
    est_default = DropSizeDistributionEstimator()
    assert est_default.estimate("a", [5.0]).spatial_calibration_status == "spatial_calibration_unavailable"

    est_custom = DropSizeDistributionEstimator(spatial_calibration_status="validated_by_target_test")
    assert est_custom.estimate("a", [5.0]).spatial_calibration_status == "validated_by_target_test"


def test_window_statistics_spatial_status_configurable():
    ws_default = _ws(window_seconds=1.0)
    rows, _ = ws_default.compute("a", [], {0: 0.0}, {}, set())
    assert rows[0].spatial_calibration_status == "spatial_calibration_unavailable"

    ws_custom = _ws(window_seconds=1.0, spatial_calibration_status="validated_by_target_test")
    rows2, _ = ws_custom.compute("a", [], {0: 0.0}, {}, set())
    assert rows2[0].spatial_calibration_status == "validated_by_target_test"


def test_stereo_common_volume_unavailable_without_calibration():
    """没有标定文件时：n_matched_pairs必须是0，concentration_status必须是
    'unavailable'（不是'算出来是0'），不能拿单路目标数顶替。"""
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    stats = StereoCommonVolumeStatistics(window_seconds=0.1)
    rows = stats.compute(
        calibration_available=False, t_min=0.0, t_max=0.5, matched_pair_entry_times=[],
    )

    assert len(rows) > 0
    for r in rows:
        assert r.n_matched_pairs == 0
        assert r.valid_target_count == 0
        assert r.concentration_m3 is None
        assert r.concentration_status == "unavailable"
        assert r.stereo_match_status == "calibration_unavailable"
        assert r.quality_status == "unavailable"


def test_stereo_common_volume_rejects_matches_without_calibration():
    """calibration_available=False 时传入非空匹配列表必须报错——这是数据
    不一致（没标定哪来的匹配成功？），不能悄悄接受。"""
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    stats = StereoCommonVolumeStatistics(window_seconds=0.1)
    with pytest.raises(ValueError):
        stats.compute(calibration_available=False, t_min=0.0, t_max=0.5, matched_pair_entry_times=[0.05])


def test_stereo_common_volume_computed_when_calibration_available():
    """有标定、有真实匹配对时，浓度应该正常算出来，concentration_status='computed'。"""
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    stats = StereoCommonVolumeStatistics(
        window_seconds=1.0,
        concentration_estimator={"name": "nominal_volume_concentration",
                                  "params": {"volume_cm3": 590.4, "volume_m3": 0.0005904}},
    )
    rows = stats.compute(
        calibration_available=True, t_min=0.0, t_max=1.0,
        matched_pair_entry_times=[0.1, 0.2, 0.3],
    )

    assert len(rows) == 1
    assert rows[0].n_matched_pairs == 3
    assert rows[0].valid_target_count == 3
    assert rows[0].concentration_status == "computed"
    assert rows[0].concentration_m3 == pytest.approx(3 / 0.0005904)
    assert rows[0].stereo_match_status == "ok"
    assert rows[0].quality_status == "nominal"


def test_stereo_common_volume_zero_matches_with_calibration_is_computed_not_unavailable():
    """有标定但这一刻确实没有匹配目标：这是"真实的0浓度"，
    不是"没法测"，两种状态不能混淆。"""
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    stats = StereoCommonVolumeStatistics(window_seconds=1.0)
    rows = stats.compute(calibration_available=True, t_min=0.0, t_max=1.0, matched_pair_entry_times=[])

    assert rows[0].n_matched_pairs == 0
    assert rows[0].concentration_status == "computed"
    assert rows[0].concentration_m3 == 0.0
    assert rows[0].quality_status == "empty"


def test_stereo_common_volume_and_single_camera_use_same_window_bounds():
    """两条统计线用同一套切窗口逻辑，结果里的窗口数、起止时间必须完全对得上。"""
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    ws = _ws(window_seconds=0.1)
    det = _det(cx=100, cy=100)
    track = _track(1, "a", [_point(1, "a", 0, 0.0, det)])
    proxy_rows, _ = ws.compute("a", [track], {0: 0.0, 1: 0.45}, {0: [det], 1: []}, set())

    stereo_stats = StereoCommonVolumeStatistics(window_seconds=0.1)
    stereo_rows = stereo_stats.compute(calibration_available=False, t_min=0.0, t_max=0.45,
                                        matched_pair_entry_times=[])

    assert len(proxy_rows) == len(stereo_rows)
    for p, s in zip(proxy_rows, stereo_rows):
        assert p.window_start_s == pytest.approx(s.window_start_s)
        assert p.window_end_s == pytest.approx(s.window_end_s)


def test_stereo_common_volume_status_fields_configurable():
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    stats = StereoCommonVolumeStatistics(
        window_seconds=1.0, spatial_calibration_status="validated_by_target_test",
        concentration_estimator={"name": "nominal_volume_concentration",
                                  "params": {"volume_status": "validated_by_bench_test"}},
    )
    rows = stats.compute(calibration_available=True, t_min=0.0, t_max=1.0, matched_pair_entry_times=[])
    assert rows[0].spatial_calibration_status == "validated_by_target_test"
    assert rows[0].volume_status == "validated_by_bench_test"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
