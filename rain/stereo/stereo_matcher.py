"""StereoMatcher 实现：把 EpipolarGate、MatchGate、CostMatrixBuilder、
NearestNeighborSolver、HungarianSolver 按固定顺序串起来，产出一帧的
双路匹配诊断。

这是阶段4六个必需模块里"组合"的那一个——本身不包含任何几何/代价计算，
只负责按顺序调用其它模块，和 ModularPipelineDetector（阶段2）、
TrackManager（阶段3）是同一种设计，满足"不得把算法逻辑写入Pipeline"的
要求（这个类本身就是"阶段4的算法"，pipeline/stage4_stereo_matching.py
只是按 pair_index 循环调用它，不做任何几何或代价计算）。

流程：
    1. 对每个 (A检测, B检测) 候选：先过 EpipolarGate，再过 MatchGate。
    2. 通过两道门的候选，用 CostMatrixBuilder 算总代价；总代价超过
       max_total_cost 的候选也判不通过（不强行配对）。
    3. 剩下通过的候选看成二分图，按连通分量拆分：
       1条A、1条B的分量直接用 NearestNeighborSolver；
       有竞争的分量交给 HungarianSolver（复用阶段3已验证实现）。
    4. 汇总成每个候选的完整诊断行，包括匹配成功的和各种原因失败的。
"""

from __future__ import annotations

import time

import numpy as np

from ..registry import (
    build_cost_matrix_builder,
    build_epipolar_gate,
    build_hungarian_solver,
    build_match_gate,
    build_nn_solver,
    register_stereo_matcher,
)
from ..tracking.track_associator import _connected_components  # 复用阶段3已验证的连通分量算法
from ..types import Detection
from .types import CostBreakdown, StereoCalibration, StereoFrameResult, StereoMatchRow

# 汇报未匹配原因时，按"信息量从高到低"排序；同一个候选可能同时撞上多种
# 失败原因（比如既超出面积比又超出极线距离），优先报告更具体的几何/形状
# 原因，而不是笼统的"总代价超阈值"。
_REASON_PRIORITY = (
    "epipolar_gate_rejected",
    "degenerate_epipolar_line",
    "area_ratio_exceeded",
    "aspect_diff_exceeded",
    "total_cost_exceeded",
)


def _resolve_cfg(cfg: dict | None, default_name: str) -> dict:
    """把调用方传来的"可能只给了params、没给name"的局部配置，补全成
    {"name":..., "params": {...}} 的完整形状，供 registry._build 使用。

    这样调用方（config.yaml 或者测试代码）既可以只写要覆盖的参数
    （{"params": {"thresh": 1}}），也可以完整指定 name 换一个实现，
    两种写法都不会因为"忘了填 name"而报错。
    """
    cfg = dict(cfg or {})
    return {"name": cfg.get("name", default_name), "params": dict(cfg.get("params") or {})}


@register_stereo_matcher("gated_cost_stereo_matcher")
class StereoMatcher:
    """双路匹配的组合模块。参数全部来自 config/pipeline.yaml 的 stereo 段。"""

    def __init__(
        self,
        epipolar_gate: dict | None = None,
        match_gate: dict | None = None,
        cost_matrix_builder: dict | None = None,
        nn_solver: dict | None = None,
        hungarian_solver: dict | None = None,
        max_total_cost: float = 0.6,
        fixed_exposure_delay_calibrated: bool = False,
        calibration: StereoCalibration | None = None,
    ):
        # calibration 是一个 Python 对象（由调用方先用 CalibrationLoader
        # 读出来，或者测试里直接构造），不是能写进 YAML 的标量参数，
        # 所以单独作为构造参数接收，再并入 epipolar_gate 的嵌套配置里，
        # 不走 registry._build 的 **params 展开（那条路径只能传
        # JSON/YAML 能表示的基本类型）。
        eg_cfg = _resolve_cfg(epipolar_gate, "fundamental_matrix_gate")
        eg_cfg["params"]["calibration"] = calibration
        self.epipolar_gate = build_epipolar_gate(eg_cfg)

        self.match_gate = build_match_gate(_resolve_cfg(match_gate, "size_shape_gate"))
        self.cost_builder = build_cost_matrix_builder(
            _resolve_cfg(cost_matrix_builder, "weighted_normalized_cost")
        )
        self.nn_solver = build_nn_solver(_resolve_cfg(nn_solver, "greedy_nearest_neighbor"))
        self.hungarian_solver = build_hungarian_solver(_resolve_cfg(hungarian_solver, "kuhn_munkres"))
        self.max_total_cost = float(max_total_cost)
        # 固定曝光延迟未标定：任务书要求的全局降级——即使某天有了双相机
        # 标定，只要这一项还没标定，所有匹配都必须标"低置信度"，
        # 不能因为总代价低就自动升级成高置信度。
        self.fixed_exposure_delay_calibrated = bool(fixed_exposure_delay_calibrated)

    def match_frame(
        self, pair_index: int, dets_a: list[Detection], dets_b: list[Detection]
    ) -> StereoFrameResult:
        n_a, n_b = len(dets_a), len(dets_b)
        calibration_available = self.epipolar_gate.is_available

        # --- 1. 极线门控 + 匹配门控 ------------------------------------------
        t0 = time.perf_counter()
        gated = np.zeros((n_a, n_b), dtype=bool)
        epi_dist = np.full((n_a, n_b), np.nan)
        gate_fail_reason: dict[tuple[int, int], str] = {}

        for i, da in enumerate(dets_a):
            for j, db in enumerate(dets_b):
                ok, d, reason = self.epipolar_gate.passes((da.cx, da.cy), (db.cx, db.cy))
                if d is not None:
                    epi_dist[i, j] = d
                if not ok:
                    gate_fail_reason[(i, j)] = reason
                    continue
                mg_ok, mg_reason = self.match_gate.passes(da, db)
                if not mg_ok:
                    gate_fail_reason[(i, j)] = mg_reason
                    continue
                gated[i, j] = True
        gate_ms = (time.perf_counter() - t0) * 1000.0

        # --- 2. 代价矩阵（只对通过两道门的候选算）------------------------------
        t0 = time.perf_counter()
        cost = np.full((n_a, n_b), np.inf)
        breakdowns: dict[tuple[int, int], CostBreakdown] = {}

        for i in range(n_a):
            for j in range(n_b):
                if not gated[i, j]:
                    continue
                bd = self.cost_builder.compute(
                    dets_a[i], dets_b[j],
                    epipolar_distance_px=None if np.isnan(epi_dist[i, j]) else float(epi_dist[i, j]),
                    calibration=self.epipolar_gate.calibration,
                )
                breakdowns[(i, j)] = bd
                if bd.total_cost > self.max_total_cost:
                    gated[i, j] = False
                    gate_fail_reason[(i, j)] = "total_cost_exceeded"
                    continue
                cost[i, j] = bd.total_cost
        cost_ms = (time.perf_counter() - t0) * 1000.0

        # --- 3. 连通分量拆分 + 最近邻/匈牙利求解 -------------------------------
        t0 = time.perf_counter()
        matches: list[tuple[int, int]] = []
        methods: dict[tuple[int, int], str] = {}
        n_conflicts = 0

        for a_idxs, b_idxs in _connected_components(gated):
            if len(a_idxs) == 1 and len(b_idxs) == 1:
                i, j = a_idxs[0], b_idxs[0]
                for li, lj in self.nn_solver.solve(cost[np.ix_([i], [j])]):
                    matches.append((i, j))
                    methods[(i, j)] = "nearest_neighbor"
                continue

            n_conflicts += 1
            sub_cost = cost[np.ix_(a_idxs, b_idxs)]
            for li, lj in self.hungarian_solver.solve(sub_cost):
                i, j = a_idxs[li], b_idxs[lj]
                if not np.isfinite(cost[i, j]):
                    continue
                matches.append((i, j))
                methods[(i, j)] = "hungarian"
        solve_ms = (time.perf_counter() - t0) * 1000.0

        rows = self._build_rows(
            pair_index, dets_a, dets_b, matches, methods, breakdowns,
            epi_dist, gated, gate_fail_reason, calibration_available,
        )
        return StereoFrameResult(
            rows=rows, n_candidate_conflicts=n_conflicts,
            gate_ms=gate_ms, cost_ms=cost_ms, solve_ms=solve_ms,
        )

    # -------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------

    def _confidence(self, total_cost: float, calibration_available: bool) -> str:
        if not calibration_available or not self.fixed_exposure_delay_calibrated:
            return "low"
        if total_cost < 0.3:
            return "high"
        if total_cost < 0.7:
            return "medium"
        return "low"

    def _build_rows(
        self, pair_index, dets_a, dets_b, matches, methods, breakdowns,
        epi_dist, gated, gate_fail_reason, calibration_available,
    ) -> list[StereoMatchRow]:
        matched_a = {i for i, _ in matches}
        matched_b = {j for _, j in matches}
        rows: list[StereoMatchRow] = []

        for i, j in matches:
            bd = breakdowns[(i, j)]
            da, db = dets_a[i], dets_b[j]
            rows.append(StereoMatchRow(
                pair_index=pair_index, a_index=i, b_index=j,
                a_cx=da.cx, a_cy=da.cy, b_cx=db.cx, b_cy=db.cy,
                epipolar_distance_px=None if np.isnan(epi_dist[i, j]) else float(epi_dist[i, j]),
                cost_epipolar=bd.cost_epipolar, cost_position=bd.cost_position,
                cost_area=bd.cost_area, cost_aspect=bd.cost_aspect, total_cost=bd.total_cost,
                gate_epipolar_passed=True, gate_match_passed=True,
                method=methods[(i, j)],
                confidence=self._confidence(bd.total_cost, calibration_available),
                status="matched", reason="ok",
            ))

        for i, da in enumerate(dets_a):
            if i in matched_a:
                continue
            reason = self._unmatched_reason(
                gated[i, :], [gate_fail_reason.get((i, k)) for k in range(len(dets_b))],
                calibration_available,
            )
            rows.append(StereoMatchRow(
                pair_index=pair_index, a_index=i, b_index=None,
                a_cx=da.cx, a_cy=da.cy, b_cx=None, b_cy=None,
                epipolar_distance_px=None, cost_epipolar=None, cost_position=None,
                cost_area=None, cost_aspect=None, total_cost=None,
                gate_epipolar_passed=False, gate_match_passed=False,
                method="none", confidence="none", status="unmatched_a", reason=reason,
            ))

        for j, db in enumerate(dets_b):
            if j in matched_b:
                continue
            reason = self._unmatched_reason(
                gated[:, j], [gate_fail_reason.get((k, j)) for k in range(len(dets_a))],
                calibration_available,
            )
            rows.append(StereoMatchRow(
                pair_index=pair_index, a_index=None, b_index=j,
                a_cx=None, a_cy=None, b_cx=db.cx, b_cy=db.cy,
                epipolar_distance_px=None, cost_epipolar=None, cost_position=None,
                cost_area=None, cost_aspect=None, total_cost=None,
                gate_epipolar_passed=False, gate_match_passed=False,
                method="none", confidence="none", status="unmatched_b", reason=reason,
            ))

        return rows

    @staticmethod
    def _unmatched_reason(gated_row_or_col, fail_reasons: list, calibration_available: bool) -> str:
        if not calibration_available:
            return "calibration_unavailable"
        if np.any(gated_row_or_col):
            # 门控和代价都通过了，只是在冲突消解里没被选中——不是几何/形状不合理。
            return "lost_in_conflict_resolution"

        reasons = {r for r in fail_reasons if r is not None}
        if not reasons:
            return "no_candidate"  # 另一路当帧检测数为 0，压根没有候选可比
        for priority in _REASON_PRIORITY:
            if priority in reasons:
                return priority
        return sorted(reasons)[0]
