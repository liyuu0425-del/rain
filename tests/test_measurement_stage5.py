"""阶段5模块测试：LinearVelocityEstimator。

任务书要求覆盖的场景（每个都有对应测试）：
    已知斜率的合成轨迹、时间不等间隔、时间重复或倒退、点数不足、
    predicted_only点排除、高残差轨迹拒绝、单位和标定状态正确。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  触发所有实现的注册
from rain.measurement.linear_velocity_estimator import LinearVelocityEstimator
from rain.tracking.types import TQC_PREDICTED_ONLY, TrackPoint, TrackRecord
from rain.types import Detection


def _det(cx: float, cy: float, area: float = 50.0) -> Detection:
    return Detection(
        frame_idx=0, cx=cx, cy=cy, area_px=area,
        bbox=(int(cx - 5), int(cy - 5), 10, 10), d_eq_px=Detection.d_eq_from_area(area),
    )


def _real_point(track_id, cam, pair_index, t, cx, cy, method="gated_nn") -> TrackPoint:
    det = _det(cx, cy)
    return TrackPoint(
        track_id=track_id, camera=cam, pair_index=pair_index, device_time_s=t,
        detected=det, predicted_cx=cx, predicted_cy=cy,
        association_method=method, quality_flags=det.qc_flags,
    )


def _predicted_only_point(track_id, cam, pair_index, t, cx, cy) -> TrackPoint:
    return TrackPoint(
        track_id=track_id, camera=cam, pair_index=pair_index, device_time_s=t,
        detected=None, predicted_cx=cx, predicted_cy=cy,
        association_method="predicted_only", quality_flags=TQC_PREDICTED_ONLY,
    )


def _track(points: list[TrackPoint], track_id=1, cam="a", status="ended") -> TrackRecord:
    return TrackRecord(track_id=track_id, camera=cam, points=points, status=status)


# ---------------------------------------------------------------------------
# 已知斜率的合成轨迹
# ---------------------------------------------------------------------------


def test_known_slope_synthetic_track_recovers_exact_velocity():
    """z(t) = 100 + 80*t 的精确合成数据，拟合出的速度必须精确等于80px/s。"""
    v_true, z0_true = 80.0, 100.0
    ts = [0.0, 0.0032, 0.0064, 0.0096, 0.0128]
    points = [_real_point(1, "a", i, t, cx=500.0, cy=z0_true + v_true * t) for i, t in enumerate(ts)]

    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.velocity_px_per_s == pytest.approx(v_true, abs=1e-6)
    assert result.z0_px == pytest.approx(z0_true, abs=1e-6)
    assert result.residual_rms_px == pytest.approx(0.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.confidence == "high"
    assert result.status == "fitted"


def test_known_slope_with_small_noise_recovers_approximate_velocity():
    # 用接近真实数据量级的速度(参考老Demo实测dy 20~140px/帧、周期3200us，
    # 换算约6000~44000 px/s)，噪声0.3px相对这个信号幅度可以忽略，
    # 这样测的是"小噪声下拟合仍然准确"，而不是"信号比噪声还小"的退化场景。
    v_true, z0_true = 10000.0, 200.0
    rng = np.random.default_rng(0)
    ts = np.array([0.0, 0.0032, 0.0064, 0.0096, 0.0128, 0.0160])
    noise = rng.normal(0, 0.3, size=len(ts))  # 很小的噪声，像素级
    zs = z0_true + v_true * ts + noise
    points = [_real_point(1, "a", i, float(t), 500.0, float(z)) for i, (t, z) in enumerate(zip(ts, zs))]

    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.velocity_px_per_s == pytest.approx(v_true, rel=0.02)
    assert result.r_squared > 0.999


# ---------------------------------------------------------------------------
# 时间不等间隔
# ---------------------------------------------------------------------------


def test_irregular_time_intervals_still_fit_correctly():
    """最小二乘对时间间隔是否均匀不敏感，只要是精确直线就必须精确拟合。"""
    v_true, z0_true = 30.0, 0.0
    ts = [0.0, 0.001, 0.0035, 0.01, 0.0102, 0.02]  # 故意不等间隔
    points = [_real_point(1, "a", i, t, 100.0, z0_true + v_true * t) for i, t in enumerate(ts)]

    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.velocity_px_per_s == pytest.approx(v_true, abs=1e-6)
    assert result.n_points == len(ts)
    assert result.duration_s == pytest.approx(ts[-1] - ts[0], abs=1e-9)


# ---------------------------------------------------------------------------
# 时间重复或倒退
# ---------------------------------------------------------------------------


def test_duplicate_timestamps_flagged_and_low_confidence():
    points = [
        _real_point(1, "a", 0, 0.0, 100, 100),
        _real_point(1, "a", 1, 0.0032, 100, 130),
        _real_point(1, "a", 2, 0.0032, 100, 160),  # 和上一个时间戳完全相同
    ]
    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.time_quality == "duplicate_timestamps"
    assert result.confidence == "low"
    assert result.reason == "time_quality_duplicate_timestamps"


def test_time_backwards_flagged_and_low_confidence():
    points = [
        _real_point(1, "a", 0, 0.0064, 100, 100),
        _real_point(1, "a", 1, 0.0032, 100, 130),  # 比上一条的时间还早——时间倒退
        _real_point(1, "a", 2, 0.0096, 100, 160),
    ]
    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.time_quality == "non_monotonic"
    assert result.confidence == "low"
    assert result.reason == "time_quality_non_monotonic"


def test_all_same_timestamp_is_degenerate_and_rejected():
    points = [
        _real_point(1, "a", 0, 1.0, 100, 100),
        _real_point(1, "a", 1, 1.0, 100, 130),
        _real_point(1, "a", 2, 1.0, 100, 160),
    ]
    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.time_quality == "degenerate_time_span"
    assert result.status == "rejected"
    assert result.velocity_px_per_s is None
    assert result.confidence == "none"


# ---------------------------------------------------------------------------
# 点数不足 / 两点低置信度
# ---------------------------------------------------------------------------


def test_zero_points_no_velocity():
    result = LinearVelocityEstimator().estimate(_track([]))
    assert result.velocity_px_per_s is None
    assert result.status == "insufficient_points"
    assert result.reason == "insufficient_points"
    assert result.confidence == "none"
    assert result.unit == ""


def test_one_point_no_velocity():
    points = [_real_point(1, "a", 0, 0.0, 100, 100)]
    result = LinearVelocityEstimator().estimate(_track(points))
    assert result.velocity_px_per_s is None
    assert result.status == "insufficient_points"


def test_two_points_low_confidence_velocity():
    points = [
        _real_point(1, "a", 0, 0.0, 100, 100),
        _real_point(1, "a", 1, 0.0032, 100, 100 + 32 * 0.0032),  # 精确构造出 v=32px/s(仅示意)
    ]
    # 直接用简单数字重算：dt=0.0032, dz=?, 让 v 精确可算
    points = [
        _real_point(1, "a", 0, 0.0, 100, 100.0),
        _real_point(1, "a", 1, 0.01, 100, 100.0 + 25.0 * 0.01),
    ]
    result = LinearVelocityEstimator().estimate(_track(points))

    assert result.n_points == 2
    assert result.status == "fitted_two_point"
    assert result.confidence == "low"
    assert result.velocity_px_per_s == pytest.approx(25.0, abs=1e-6)
    assert result.r_squared is None  # 两点拟合没有统计意义上的R²
    assert result.unit == "px/s"


def test_below_min_points_for_fit_downgrades_confidence():
    """min_points_for_fit=5 时，3个点虽然能算出拟合值，但不能给中/高置信度。"""
    points = [_real_point(1, "a", i, i * 0.0032, 100, 100 + 40 * i * 0.0032) for i in range(3)]
    result = LinearVelocityEstimator(min_points_for_fit=5).estimate(_track(points))

    assert result.n_points == 3
    assert result.velocity_px_per_s is not None  # 仍然给出拟合结果
    assert result.confidence == "low"
    assert result.reason == "below_min_points_for_fit"


# ---------------------------------------------------------------------------
# predicted_only 点排除
# ---------------------------------------------------------------------------


def test_predicted_only_points_are_excluded_from_fit():
    """即使 track.points 里混进了 predicted_only 点（正常上游不会发生，
    但测速模块必须自己防御），也不能把它们当成真实观测参与拟合。"""
    real_points = [
        _real_point(1, "a", 0, 0.0, 100, 100.0),
        _real_point(1, "a", 2, 0.0064, 100, 100.0 + 60 * 0.0064),
        _real_point(1, "a", 4, 0.0128, 100, 100.0 + 60 * 0.0128),
    ]
    # 混入一个"仅预测"点，位置故意偏得离谱，如果没被过滤，会严重拉低R²或改变斜率
    fake_predicted = _predicted_only_point(1, "a", 1, 0.0032, 100, 99999.0)
    mixed_points = [real_points[0], fake_predicted, real_points[1], real_points[2]]

    result_with_noise = LinearVelocityEstimator().estimate(_track(mixed_points))
    result_clean = LinearVelocityEstimator().estimate(_track(real_points))

    assert result_with_noise.n_points == 3  # 混入的那个不算数
    assert result_with_noise.velocity_px_per_s == pytest.approx(result_clean.velocity_px_per_s, abs=1e-6)
    assert result_with_noise.r_squared == pytest.approx(result_clean.r_squared, abs=1e-6)


# ---------------------------------------------------------------------------
# 高残差轨迹拒绝
# ---------------------------------------------------------------------------


def test_high_residual_track_marked_low_confidence():
    """点严重偏离直线（残差很大）时，即使点数够、R²可能也不高，必须标低置信度。"""
    points = [
        _real_point(1, "a", 0, 0.0, 100, 100.0),
        _real_point(1, "a", 1, 0.0032, 100, 500.0),   # 剧烈跳变，不像匀速直线运动
        _real_point(1, "a", 2, 0.0064, 100, 105.0),
        _real_point(1, "a", 3, 0.0096, 100, 600.0),
    ]
    result = LinearVelocityEstimator(max_residual_px=5.0, min_r_squared=0.8).estimate(_track(points))

    assert result.confidence == "low"
    assert result.reason in ("high_residual", "low_r_squared")
    # 仍然给出拟合值（不是直接丢弃数据），只是标记质量差
    assert result.velocity_px_per_s is not None


def test_low_residual_track_gets_higher_confidence():
    points = [_real_point(1, "a", i, i * 0.0032, 100, 200 + 45 * i * 0.0032) for i in range(6)]
    result = LinearVelocityEstimator(max_residual_px=5.0, min_r_squared=0.8).estimate(_track(points))
    assert result.confidence in ("medium", "high")
    assert result.reason == "ok"


# ---------------------------------------------------------------------------
# 单位和标定状态
# ---------------------------------------------------------------------------


def test_unit_is_px_per_s_when_velocity_available():
    points = [_real_point(1, "a", i, i * 0.0032, 100, 100 + 40 * i * 0.0032) for i in range(4)]
    result = LinearVelocityEstimator().estimate(_track(points))
    assert result.unit == "px/s"


def test_unit_empty_when_no_velocity():
    result = LinearVelocityEstimator().estimate(_track([]))
    assert result.unit == ""


def test_spatial_calibration_status_always_unavailable():
    """无论有没有速度结果，spatial_calibration_status 必须恒为
    spatial_calibration_unavailable——项目没有像素-毫米标定，
    这不是一个会变化的状态，是一个写死的事实声明。"""
    empty_result = LinearVelocityEstimator().estimate(_track([]))
    assert empty_result.spatial_calibration_status == "spatial_calibration_unavailable"

    points = [_real_point(1, "a", i, i * 0.0032, 100, 100 + 40 * i * 0.0032) for i in range(4)]
    fitted_result = LinearVelocityEstimator().estimate(_track(points))
    assert fitted_result.spatial_calibration_status == "spatial_calibration_unavailable"


def test_never_outputs_mm_or_m_per_s_unit():
    """穷举各种场景，unit 字段只能是 'px/s' 或 ''，绝不能是 mm/s 或 m/s。"""
    scenarios = [
        [],
        [_real_point(1, "a", 0, 0.0, 100, 100)],
        [_real_point(1, "a", 0, 0.0, 100, 100), _real_point(1, "a", 1, 0.01, 100, 105)],
        [_real_point(1, "a", i, i * 0.0032, 100, 100 + 40 * i * 0.0032) for i in range(6)],
    ]
    for points in scenarios:
        result = LinearVelocityEstimator().estimate(_track(points))
        assert result.unit in ("px/s", "")


# ---------------------------------------------------------------------------
# camera 字段正确透传（A用a_device_relative_us，B用b_device_relative_us
# 这条约束在阶段3已经通过 TrackManager(camera=...) 保证，这里只验证
# LinearVelocityEstimator 正确地把 track.camera 原样带到结果里）
# ---------------------------------------------------------------------------


def test_camera_field_propagated():
    points_a = [_real_point(1, "a", i, i * 0.0032, 100, 100 + 40 * i * 0.0032) for i in range(3)]
    points_b = [_real_point(2, "b", i, i * 0.0032, 100, 100 + 40 * i * 0.0032) for i in range(3)]

    result_a = LinearVelocityEstimator().estimate(_track(points_a, track_id=1, cam="a"))
    result_b = LinearVelocityEstimator().estimate(_track(points_b, track_id=2, cam="b"))

    assert result_a.camera == "a" and result_a.track_id == 1
    assert result_b.camera == "b" and result_b.track_id == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
