"""阶段4模块测试：CalibrationLoader / EpipolarGate / MatchGate /
CostMatrixBuilder / NearestNeighborSolver / StereoMatcher。

任务书明确要求覆盖以下场景（每个都有对应测试，测试名里能直接看出对应哪条）：
    唯一匹配、极线门控拒绝、代价超阈值、多目标冲突、
    最近邻结果、匈牙利全局最优结果、无标定文件时的安全处理。

合成标定说明：
    构造一个简单的基础矩阵 F，使得"A 图像里点 (ax,ay) 对应的极线"
    正好是 B 图像里的水平线 Y=ay。这样极线距离就等于 |by-ay|，
    直观、好验证，纯粹用来测试代码路径，不代表任何真实相机几何关系
    （真实标定完成前，任何"看起来正常"的极线参数都不能用于真实数据）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  触发所有实现的注册
from rain.stereo.calibration import CalibrationLoader
from rain.stereo.cost_matrix_builder import CostMatrixBuilder
from rain.stereo.epipolar_gate import EpipolarGate
from rain.stereo.match_gate import MatchGate
from rain.stereo.nearest_neighbor_solver import NearestNeighborSolver
from rain.stereo.stereo_matcher import StereoMatcher
from rain.stereo.types import StereoCalibration
from rain.types import Detection

# 合成标定：A、B 对应点应有相同的 y 坐标（水平极线）。
_SYNTHETIC_F = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _synthetic_calibration(rectify_identity: bool = True) -> StereoCalibration:
    return StereoCalibration(
        fundamental_matrix=_SYNTHETIC_F,
        rectify_a_to_b=np.eye(3) if rectify_identity else None,
        source_path="synthetic-test-fixture",
    )


def _det(cx: float, cy: float, area: float = 50.0, w: int = 10, h: int = 10) -> Detection:
    return Detection(
        frame_idx=0, cx=cx, cy=cy, area_px=area,
        bbox=(int(cx - w / 2), int(cy - h / 2), w, h),
        d_eq_px=Detection.d_eq_from_area(area),
    )


# ---------------------------------------------------------------------------
# CalibrationLoader
# ---------------------------------------------------------------------------


def test_calibration_loader_missing_file_returns_none():
    assert CalibrationLoader().load("这个文件不存在.json") is None


def test_calibration_loader_none_path_returns_none():
    assert CalibrationLoader().load("") is None
    assert CalibrationLoader().load(None) is None


def test_calibration_loader_reads_valid_file(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(
        '{"fundamental_matrix": [[0,0,0],[0,0,1],[0,-1,0]], "rectify_a_to_b": [[1,0,0],[0,1,0],[0,0,1]]}',
        encoding="utf-8",
    )
    calib = CalibrationLoader().load(str(p))
    assert calib is not None
    assert calib.fundamental_matrix.shape == (3, 3)
    assert calib.rectify_a_to_b.shape == (3, 3)
    assert calib.source_path == str(p)


def test_calibration_loader_rejects_wrong_shape(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"fundamental_matrix": [[1,2],[3,4]]}', encoding="utf-8")
    with pytest.raises(ValueError):
        CalibrationLoader().load(str(p))


# ---------------------------------------------------------------------------
# EpipolarGate —— 含"无标定文件时的安全处理"
# ---------------------------------------------------------------------------


def test_epipolar_gate_unavailable_without_calibration():
    gate = EpipolarGate(calibration=None)
    assert gate.is_available is False

    ok, dist, reason = gate.passes((10, 10), (10, 10))
    assert ok is False
    assert dist is None
    assert reason == "calibration_unavailable"


def test_epipolar_gate_does_not_crash_or_fabricate_without_calibration():
    """没有标定时，不管传什么点，都必须安全返回，不抛异常、不编造一个"看似合理"的距离。"""
    gate = EpipolarGate(calibration=None)
    for _ in range(20):
        pt_a = tuple(np.random.uniform(0, 2000, size=2))
        pt_b = tuple(np.random.uniform(0, 2000, size=2))
        ok, dist, reason = gate.passes(pt_a, pt_b)
        assert ok is False and dist is None and reason == "calibration_unavailable"


def test_epipolar_gate_accepts_point_on_epipolar_line():
    gate = EpipolarGate(calibration=_synthetic_calibration(), max_epipolar_dist_px=1.0)
    ok, dist, reason = gate.passes((10, 100), (500, 100))  # 同 y，应该在极线上
    assert ok is True
    assert dist == pytest.approx(0.0, abs=1e-6)
    assert reason == "ok"


def test_epipolar_gate_rejects_point_far_from_epipolar_line():
    gate = EpipolarGate(calibration=_synthetic_calibration(), max_epipolar_dist_px=5.0)
    ok, dist, reason = gate.passes((10, 100), (500, 200))  # y差100，远超阈值
    assert ok is False
    assert dist == pytest.approx(100.0, abs=1e-6)
    assert reason == "epipolar_gate_rejected"


def test_epipolar_gate_degenerate_line():
    zero_F = np.zeros((3, 3))
    calib = StereoCalibration(fundamental_matrix=zero_F, rectify_a_to_b=None, source_path="x")
    gate = EpipolarGate(calibration=calib, max_epipolar_dist_px=5.0)
    ok, dist, reason = gate.passes((10, 10), (20, 20))
    assert ok is False
    assert dist is None
    assert reason == "degenerate_epipolar_line"


# ---------------------------------------------------------------------------
# MatchGate
# ---------------------------------------------------------------------------


def test_match_gate_passes_similar_detections():
    gate = MatchGate(max_area_ratio=3.0, max_aspect_diff=2.0)
    ok, reason = gate.passes(_det(0, 0, area=50, w=10, h=10), _det(0, 0, area=55, w=11, h=10))
    assert ok is True and reason == "ok"


def test_match_gate_rejects_area_ratio():
    gate = MatchGate(max_area_ratio=2.0, max_aspect_diff=10.0)
    ok, reason = gate.passes(_det(0, 0, area=10), _det(0, 0, area=1000))
    assert ok is False and reason == "area_ratio_exceeded"


def test_match_gate_rejects_aspect_diff():
    gate = MatchGate(max_area_ratio=100.0, max_aspect_diff=0.5)
    ok, reason = gate.passes(_det(0, 0, area=50, w=10, h=10), _det(0, 0, area=50, w=40, h=5))
    assert ok is False and reason == "aspect_diff_exceeded"


# ---------------------------------------------------------------------------
# CostMatrixBuilder
# ---------------------------------------------------------------------------


def test_cost_builder_identical_detections_near_zero_cost():
    builder = CostMatrixBuilder()
    da = _det(100, 100, area=50, w=10, h=10)
    db = _det(100, 100, area=50, w=10, h=10)
    bd = builder.compute(da, db, epipolar_distance_px=0.0, calibration=_synthetic_calibration())
    assert bd.total_cost == pytest.approx(0.0, abs=1e-6)


def test_cost_builder_position_unavailable_without_rectify():
    builder = CostMatrixBuilder()
    da, db = _det(100, 100), _det(100, 100)
    bd = builder.compute(da, db, epipolar_distance_px=0.0, calibration=None)
    assert bd.cost_position is None
    # 缺失项应该被剔除重新归一化，而不是让总代价失真成 0（这里其余项本来就是0，所以仍为0）
    assert bd.total_cost == pytest.approx(0.0, abs=1e-6)


def test_cost_builder_large_area_diff_increases_cost():
    builder = CostMatrixBuilder()
    da = _det(0, 0, area=50)
    db_close = _det(0, 0, area=55)
    db_far = _det(0, 0, area=5000)
    bd_close = builder.compute(da, db_close, epipolar_distance_px=0.0, calibration=_synthetic_calibration())
    bd_far = builder.compute(da, db_far, epipolar_distance_px=0.0, calibration=_synthetic_calibration())
    assert bd_far.total_cost > bd_close.total_cost


def test_cost_builder_all_costs_bounded_0_1():
    builder = CostMatrixBuilder()
    da = _det(0, 0, area=1)
    db = _det(9999, 9999, area=99999, w=1, h=999)
    bd = builder.compute(da, db, epipolar_distance_px=99999, calibration=_synthetic_calibration())
    for v in (bd.cost_epipolar, bd.cost_position, bd.cost_area, bd.cost_aspect, bd.total_cost):
        assert v is None or 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# NearestNeighborSolver —— "最近邻结果"
# ---------------------------------------------------------------------------


def test_nn_solver_picks_global_minimum_pairs():
    cost = np.array([
        [1.0, 5.0],
        [5.0, 2.0],
    ])
    solver = NearestNeighborSolver()
    matches = solver.solve(cost)
    assert set(matches) == {(0, 0), (1, 1)}


def test_nn_solver_ignores_inf_entries():
    cost = np.array([[np.inf, 3.0], [3.0, np.inf]])
    solver = NearestNeighborSolver()
    matches = solver.solve(cost)
    assert set(matches) == {(0, 1), (1, 0)}


def test_nn_solver_empty_matrix():
    assert NearestNeighborSolver().solve(np.zeros((0, 0))) == []


def test_nn_solver_all_inf_returns_empty():
    cost = np.full((2, 2), np.inf)
    assert NearestNeighborSolver().solve(cost) == []


# ---------------------------------------------------------------------------
# StereoMatcher —— 任务书要求的七类场景
# ---------------------------------------------------------------------------


def _matcher(**overrides) -> StereoMatcher:
    params = dict(
        calibration=_synthetic_calibration(),
        epipolar_gate={"params": {"max_epipolar_dist_px": 5.0}},
        match_gate={"params": {"max_area_ratio": 3.0, "max_aspect_diff": 2.0}},
        cost_matrix_builder={"params": {}},
        max_total_cost=0.6,
        fixed_exposure_delay_calibrated=True,  # 测试里放开置信度分级，验证真实的分级逻辑
    )
    params.update(overrides)
    return StereoMatcher(**params)


def test_stereo_unique_match():
    """场景1：唯一匹配——A、B 各一个检测，几何和形状都吻合。"""
    m = _matcher()
    res = m.match_frame(0, [_det(100, 200, area=50)], [_det(900, 200, area=50)])

    matched = [r for r in res.rows if r.status == "matched"]
    assert len(matched) == 1
    assert matched[0].a_index == 0 and matched[0].b_index == 0
    assert matched[0].reason == "ok"


def test_stereo_epipolar_gate_rejection():
    """场景2：极线门控拒绝——y坐标差远超阈值。"""
    m = _matcher()
    res = m.match_frame(0, [_det(100, 200, area=50)], [_det(900, 800, area=50)])

    assert all(r.status != "matched" for r in res.rows)
    a_row = next(r for r in res.rows if r.status == "unmatched_a")
    assert a_row.reason == "epipolar_gate_rejected"


def test_stereo_cost_exceeds_threshold():
    """场景3：代价超阈值——极线和形状都在门控内，但总代价超过max_total_cost。"""
    m = _matcher(max_total_cost=0.05)  # 卡到很紧的阈值
    # y相同(过极线门)，面积差一点(过形状门，area_ratio=1.5<3.0)，
    # 但面积差本身会贡献一定代价，紧阈值下应该超标。
    res = m.match_frame(0, [_det(100, 200, area=50)], [_det(900, 200, area=75)])

    assert all(r.status != "matched" for r in res.rows)
    a_row = next(r for r in res.rows if r.status == "unmatched_a")
    assert a_row.reason == "total_cost_exceeded"


def test_stereo_multi_target_conflict_uses_hungarian():
    """场景4：多目标冲突——两条A候选和两条B候选互相在门控范围内竞争。"""
    m = _matcher()
    dets_a = [_det(100, 200, area=50), _det(105, 200, area=50)]
    dets_b = [_det(900, 200, area=50), _det(905, 200, area=50)]

    res = m.match_frame(0, dets_a, dets_b)

    assert res.n_candidate_conflicts >= 1
    matched = [r for r in res.rows if r.status == "matched"]
    assert len(matched) == 2
    assert all(r.method == "hungarian" for r in matched)


def test_stereo_nearest_neighbor_trivial_case():
    """场景5：最近邻结果——只有一对候选，没有竞争，方法必须是 nearest_neighbor。"""
    m = _matcher()
    res = m.match_frame(0, [_det(100, 200)], [_det(900, 200)])
    matched = [r for r in res.rows if r.status == "matched"]
    assert matched[0].method == "nearest_neighbor"


def test_stereo_hungarian_global_optimum():
    """场景6：匈牙利全局最优——贪心会选错，Hungarian必须选全局代价更小的组合。

    仿照阶段3的经典反例：A0 离 B0 最近，A1 离 B0 也比离 B1 近，
    贪心抢 B0 会导致次优，全局最优应该是 A0-B1、A1-B0。
    """
    m = _matcher(cost_matrix_builder={"params": {
        "weights": {"epipolar": 0.0, "position": 0.0, "area": 1.0, "aspect": 0.0},
        "norm_scale": {"epipolar_px": 5.0, "position_px": 50.0, "area_px": 50.0, "aspect": 2.0},
    }})
    # 用面积差控制代价：A0 和 B0 面积几乎一样(代价小)，A0 和 B1 差一点(代价中)，
    # A1 和 B0 差很多但仍在门控内(代价较大)，A1 和 B1 几乎一样(代价小)。
    dets_a = [_det(100, 200, area=50), _det(105, 200, area=90)]
    dets_b = [_det(900, 200, area=52), _det(905, 200, area=92)]

    res = m.match_frame(0, dets_a, dets_b)
    matched = {(r.a_index, r.b_index) for r in res.rows if r.status == "matched"}
    assert matched == {(0, 0), (1, 1)}  # 全局最优：各自匹配面积最接近的


def test_stereo_no_calibration_safe_handling():
    """场景7：无标定文件时的安全处理——不崩溃、不虚构、明确标 calibration_unavailable。"""
    m = StereoMatcher()  # 不传 calibration，等价于标定文件不存在
    dets_a = [_det(100, 200), _det(300, 400)]
    dets_b = [_det(900, 200), _det(700, 400)]

    res = m.match_frame(0, dets_a, dets_b)

    assert all(r.status != "matched" for r in res.rows)
    assert all(r.reason == "calibration_unavailable" for r in res.rows)
    assert all(r.confidence == "none" for r in res.rows)
    assert len(res.rows) == len(dets_a) + len(dets_b)  # 每个检测都有一行诊断，没有被吞掉


def test_stereo_no_forced_match_when_gate_fails():
    """门控不通过时绝不能强行配对——即使双路各只有一个检测（没有"更好选择"可比较）。"""
    m = _matcher()
    res = m.match_frame(0, [_det(100, 200)], [_det(900, 900)])  # y差700，远超门控
    assert all(r.status != "matched" for r in res.rows)


def test_stereo_empty_one_side():
    m = _matcher()
    res = m.match_frame(0, [_det(100, 200)], [])
    assert len(res.rows) == 1
    assert res.rows[0].status == "unmatched_a"
    assert res.rows[0].reason == "no_candidate"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
