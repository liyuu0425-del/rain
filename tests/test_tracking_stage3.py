"""阶段3模块测试：KalmanPredictor / HungarianSolver / TrackAssociator / TrackManager。

不依赖真实数据，构造小规模合成场景直接验证数值和状态机行为；
真实数据（A/B各180帧）上的完整运行结果单独在 pipeline/stage3_tracking.py
产出的 stage3_report.json 里报告。
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  触发所有实现的注册
from rain.tracking.hungarian_solver import HungarianSolver
from rain.tracking.kalman_predictor import ConstantVelocityKalmanPredictor
from rain.tracking.track_associator import GatedNearestNeighborAssociator
from rain.tracking.track_manager import GatedKalmanTrackManager
from rain.tracking.types import FrameContext, TQC_NEW_TRACK, TQC_PREDICTED_ONLY
from rain.types import Detection


def _make_det(cx: float, cy: float, frame_idx: int = 0, area: float = 50.0) -> Detection:
    return Detection(
        frame_idx=frame_idx, cx=cx, cy=cy, area_px=area,
        bbox=(int(cx - 5), int(cy - 5), 10, 10),
        d_eq_px=Detection.d_eq_from_area(area),
    )


# ---------------------------------------------------------------------------
# KalmanPredictor
# ---------------------------------------------------------------------------


def test_kalman_init_state_has_zero_velocity():
    kf = ConstantVelocityKalmanPredictor()
    st = kf.init_state(100.0, 200.0)
    assert st.cx == 100.0 and st.cy == 200.0
    assert st.vx == 0.0 and st.vy == 0.0


def test_kalman_predict_moves_position_by_velocity_times_dt():
    kf = ConstantVelocityKalmanPredictor()
    st = kf.init_state(0.0, 0.0)
    st.x[2] = 10.0  # 手动注入速度：vx=10 px/s
    st.x[3] = 5.0   # vy=5 px/s

    st2 = kf.predict(st, dt=2.0)

    assert st2.cx == pytest.approx(20.0, abs=1e-6)
    assert st2.cy == pytest.approx(10.0, abs=1e-6)
    assert st2.vx == pytest.approx(10.0)  # 匀速模型：速度本身不变


def test_kalman_predict_negative_dt_raises():
    kf = ConstantVelocityKalmanPredictor()
    st = kf.init_state(0.0, 0.0)
    with pytest.raises(ValueError):
        kf.predict(st, dt=-1.0)


def test_kalman_predict_does_not_mutate_input():
    kf = ConstantVelocityKalmanPredictor()
    st = kf.init_state(10.0, 20.0)
    x_copy, P_copy = st.x.copy(), st.P.copy()

    kf.predict(st, dt=1.0)

    assert np.array_equal(st.x, x_copy)
    assert np.array_equal(st.P, P_copy)


def test_kalman_update_pulls_estimate_toward_measurement():
    kf = ConstantVelocityKalmanPredictor(measurement_noise=1.0, init_velocity_var=1.0)
    st = kf.init_state(0.0, 0.0)
    st2 = kf.update(st, cx=100.0, cy=0.0)
    # 更新后的估计应该往量测方向移动，但因为有先验（P不是无穷大），
    # 不会一步跳到测量值本身。
    assert 0.0 < st2.cx < 100.0


def test_kalman_update_does_not_mutate_input():
    kf = ConstantVelocityKalmanPredictor()
    st = kf.init_state(10.0, 20.0)
    x_copy, P_copy = st.x.copy(), st.P.copy()

    kf.update(st, cx=15.0, cy=25.0)

    assert np.array_equal(st.x, x_copy)
    assert np.array_equal(st.P, P_copy)


def test_kalman_repeated_updates_converge_toward_consistent_velocity():
    """连续喂匀速运动的量测，速度估计应该逐渐逼近真实速度(0, 80)px/s。"""
    kf = ConstantVelocityKalmanPredictor(
        process_noise_pos=1.0, process_noise_vel=50.0, measurement_noise=4.0
    )
    st = kf.init_state(100.0, 0.0)
    true_vy = 80.0
    dt = 1.0
    for i in range(1, 20):
        st = kf.predict(st, dt=dt)
        st = kf.update(st, cx=100.0, cy=true_vy * i)

    assert st.vy == pytest.approx(true_vy, rel=0.1)


# ---------------------------------------------------------------------------
# HungarianSolver（用暴力穷举对拍验证最优性）
# ---------------------------------------------------------------------------


def _brute_force_min_cost(cost: np.ndarray) -> float:
    n, m = cost.shape
    if n > m:
        return _brute_force_min_cost(cost.T)
    best = None
    for cols in itertools.permutations(range(m), n):
        total = sum(cost[r, c] for r, c in enumerate(cols))
        if best is None or total < best:
            best = total
    return best


def test_hungarian_matches_brute_force_on_random_matrices():
    solver = HungarianSolver()
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    for _ in range(50):
        n = rng.randint(1, 5)
        m = rng.randint(1, 5)
        cost = np_rng.uniform(0, 100, size=(n, m))

        matches = solver.solve(cost)
        got = sum(cost[r, c] for r, c in matches)
        expected = _brute_force_min_cost(cost)

        assert len(matches) == min(n, m)
        assert got == pytest.approx(expected, abs=1e-6)


def test_hungarian_empty_matrix_returns_empty():
    solver = HungarianSolver()
    assert solver.solve(np.zeros((0, 0))) == []


def test_hungarian_single_cell():
    solver = HungarianSolver()
    matches = solver.solve(np.array([[5.0]]))
    assert matches == [(0, 0)]


def test_hungarian_picks_globally_optimal_not_greedy():
    """经典反例：贪心按行取最小值会选到次优解，Hungarian 必须选全局最优。

    行0最近的是列0(代价1)，行1最近的也是列0(代价2)——贪心抢列0会把
    行1挤到列1(代价100)，总代价101；全局最优应该是行0->列1(10)+行1->列0(2)=12。
    """
    cost = np.array([
        [1.0, 10.0],
        [2.0, 100.0],
    ])
    solver = HungarianSolver()
    matches = solver.solve(cost)
    total = sum(cost[r, c] for r, c in matches)
    assert total == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# TrackAssociator
# ---------------------------------------------------------------------------


def test_associator_simple_match_within_gate():
    assoc = GatedNearestNeighborAssociator(gate_dx_max=20, gate_dy_max=100, max_match_cost=200)
    result = assoc.associate([(100.0, 100.0)], [_make_det(105.0, 150.0)])

    assert result.matches == [(0, 0)]
    assert result.match_methods[(0, 0)] == "gated_nn"
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []
    assert result.n_candidate_conflicts == 0


def test_associator_rejects_outside_gate():
    assoc = GatedNearestNeighborAssociator(gate_dx_max=10, gate_dy_max=10, max_match_cost=200)
    result = assoc.associate([(100.0, 100.0)], [_make_det(200.0, 200.0)])

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == [0]


def test_associator_rejects_cost_above_threshold_even_inside_box_gate():
    # dx=15 在 gate_dx_max=20 以内，但 cost_dx_weight=10 把代价拉到远超 max_match_cost。
    assoc = GatedNearestNeighborAssociator(
        gate_dx_max=20, gate_dy_max=100, cost_dx_weight=10.0, cost_dy_weight=1.0,
        max_match_cost=50.0,
    )
    result = assoc.associate([(100.0, 100.0)], [_make_det(115.0, 100.0)])

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == [0]


def test_associator_conflict_triggers_hungarian():
    """两条轨迹的门控范围重叠到同一批检测，必须触发冲突消解。"""
    assoc = GatedNearestNeighborAssociator(gate_dx_max=50, gate_dy_max=50, max_match_cost=200)
    predicted = [(100.0, 100.0), (110.0, 100.0)]  # 两条轨迹离得很近
    dets = [_make_det(102.0, 100.0), _make_det(108.0, 100.0)]  # 两个检测也离得很近

    result = assoc.associate(predicted, dets)

    assert result.n_candidate_conflicts >= 1
    assert len(result.matches) == 2  # 应该找到一个完整匹配
    assert all(result.match_methods[m] == "hungarian" for m in result.matches)
    # 全局最优应该是"就近配对"：轨迹0->检测0，轨迹1->检测1
    assert (0, 0) in result.matches
    assert (1, 1) in result.matches


def test_associator_no_tracks_all_detections_unmatched():
    assoc = GatedNearestNeighborAssociator()
    result = assoc.associate([], [_make_det(1, 1), _make_det(2, 2)])
    assert result.matches == []
    assert result.unmatched_detections == [0, 1]
    assert result.unmatched_tracks == []


def test_associator_no_detections_all_tracks_unmatched():
    assoc = GatedNearestNeighborAssociator()
    result = assoc.associate([(1, 1), (2, 2)], [])
    assert result.matches == []
    assert result.unmatched_tracks == [0, 1]
    assert result.unmatched_detections == []


# ---------------------------------------------------------------------------
# TrackManager：生命周期状态机
# ---------------------------------------------------------------------------


def _mk_manager(**overrides):
    params = dict(
        camera="a",
        max_lost_frames=2,
        min_track_len=2,
        kalman_predictor={"name": "constant_velocity_kalman", "params": {}},
        track_associator={
            "name": "gated_nearest_neighbor",
            "params": {"gate_dx_max": 20, "gate_dy_max": 150, "max_match_cost": 300},
        },
    )
    params.update(overrides)
    return GatedKalmanTrackManager(**params)


def test_track_manager_creates_new_track_on_first_detection():
    mgr = _mk_manager()
    frame = FrameContext(pair_index=0, device_time_s=0.0, detections=[_make_det(50, 50, frame_idx=0)])

    result = mgr.step(frame)

    assert len(result.points) == 1
    pt = result.points[0]
    assert pt.association_method == "new_track"
    assert pt.track_id == 1
    assert pt.quality_flags & TQC_NEW_TRACK


def test_track_manager_links_falling_drop_across_frames():
    """一条匀速下落的轨迹应该被连成同一个 track_id，跨多帧不断裂。"""
    mgr = _mk_manager()
    track_ids_seen = set()

    for i in range(6):
        frame = FrameContext(
            pair_index=i, device_time_s=i * 0.0032,  # 真实周期3200us
            detections=[_make_det(100 + i * 0.5, 50 + i * 40, frame_idx=i)],
        )
        result = mgr.step(frame)
        for pt in result.points:
            track_ids_seen.add(pt.track_id)

    assert track_ids_seen == {1}  # 全程只有一个 track_id，没有断裂成多条

    records = mgr.finalize()
    assert len(records) == 1
    assert records[0].n_points == 6
    assert records[0].status == "ended"  # n_points(6) >= min_track_len(2)


def test_track_manager_track_id_never_reused_and_monotonic():
    mgr = _mk_manager(max_lost_frames=0)  # 一丢就立刻结束，方便制造多条轨迹
    ids = []
    for i in range(3):
        frame = FrameContext(
            pair_index=i, device_time_s=i * 1.0,
            detections=[_make_det(500 + i * 1000, 500, frame_idx=i)],  # 每次都离得很远，各自成新轨迹
        )
        result = mgr.step(frame)
        ids.extend(pt.track_id for pt in result.points if pt.association_method == "new_track")

    assert ids == [1, 2, 3]  # 单调递增，不重复


def test_track_manager_ends_track_after_max_lost_frames():
    mgr = _mk_manager(max_lost_frames=2)

    mgr.step(FrameContext(0, 0.0, [_make_det(100, 100, frame_idx=0)]))
    mgr.step(FrameContext(1, 0.0032, [_make_det(100, 140, frame_idx=1)]))  # 关联上，续上轨迹

    # 接下来连续 3 帧都没有检测（超过 max_lost_frames=2），轨迹应该被结束
    r2 = mgr.step(FrameContext(2, 0.0064, []))
    r3 = mgr.step(FrameContext(3, 0.0096, []))
    r4 = mgr.step(FrameContext(4, 0.0128, []))

    assert all(pt.association_method == "predicted_only" for pt in r2.points)
    assert all(pt.quality_flags & TQC_PREDICTED_ONLY for pt in r2.points)

    records = mgr.finalize()
    assert len(records) == 1
    assert records[0].status == "ended"
    assert records[0].n_points == 2  # 只有两个真实检测点计入轨迹长度

    counters = mgr.get_summary_counters()
    assert counters["n_lost_events"] == 1  # 只在"刚进入丢失"那一刻计一次


def test_track_manager_short_track_marked_ended_short():
    mgr = _mk_manager(max_lost_frames=0, min_track_len=3)
    mgr.step(FrameContext(0, 0.0, [_make_det(100, 100, frame_idx=0)]))
    # 没有第二帧续上，max_lost_frames=0 意味着下一帧只要没关联上就立刻结束
    mgr.step(FrameContext(1, 0.0032, []))

    records = mgr.finalize()
    assert len(records) == 1
    assert records[0].n_points == 1
    assert records[0].status == "ended_short"  # 1 < min_track_len(3)


def test_track_manager_does_not_force_match_beyond_gate():
    """检测远超门控范围时，必须新建轨迹而不是强行续上旧轨迹。"""
    mgr = _mk_manager()
    mgr.step(FrameContext(0, 0.0, [_make_det(100, 100, frame_idx=0)]))
    result = mgr.step(FrameContext(1, 0.0032, [_make_det(900, 900, frame_idx=1)]))

    new_track_points = [p for p in result.points if p.association_method == "new_track"]
    predicted_only_points = [p for p in result.points if p.association_method == "predicted_only"]

    assert len(new_track_points) == 1
    assert new_track_points[0].track_id == 2  # 不是延续 track_id=1
    assert len(predicted_only_points) == 1
    assert predicted_only_points[0].track_id == 1  # 旧轨迹没被强行续上，只能靠预测撑着

    counters = mgr.get_summary_counters()
    # n_unassociated_detections 统计的是"关联阶段没能连上任何现有轨迹的检测数"，
    # 不管这个检测后续有没有被用来新建轨迹——这里两帧各有一个检测都没连上
    # 现有轨迹(第0帧时还没有任何轨迹；第1帧的检测在门控范围外)，所以是2，
    # 它们各自都变成了新轨迹，但"关联失败"这个事件本身仍然被计数。
    assert counters["n_unassociated_detections"] == 2


def test_track_manager_finalize_is_idempotent_safe():
    """finalize() 返回"迄今为止全部结算过的轨迹"，重复调用应该得到同一份结果，
    而不是第二次调用就清空——调用方（pipeline脚本）只在所有帧跑完后调一次，
    但这里验证就算意外调用两次也不会丢数据或崩溃。"""
    mgr = _mk_manager()
    mgr.step(FrameContext(0, 0.0, [_make_det(50, 50, frame_idx=0)]))
    records1 = mgr.finalize()
    records2 = mgr.finalize()
    assert len(records1) == 1
    assert records2 == records1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
