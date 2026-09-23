"""TrackAssociator 实现：门控 + 最近邻，候选冲突时用 HungarianSolver。

流程：
    1. 用一个以卡尔曼预测位置为中心的矩形门（dx/dy 独立限幅）筛出
       "轨迹-检测"候选对，门外的代价记为不可选。
    2. 把候选对看成一个二分图，按连通分量拆成若干组：
        - 组里只有 1 条轨迹、1 个检测 -> 没有竞争，直接关联（方法标
          "gated_nn"）。
        - 组里有多条轨迹或多个检测互相竞争 -> 调用 HungarianSolver
          在这个小组内求全局最优匹配（方法标"hungarian"）。
    3. 任何代价超过 max_match_cost 的候选，即使在门控范围内也不关联，
       保持未关联——不强行连接轨迹。
"""

from __future__ import annotations

import numpy as np

from ..registry import build_hungarian_solver, register_track_associator
from ..types import Detection
from .types import AssociationResult


@register_track_associator("gated_nearest_neighbor")
class GatedNearestNeighborAssociator:
    """矩形门控 + 最近邻，冲突时匈牙利求解。

    参数（全部来自 config/pipeline.yaml 的 tracking.track_associator.params）：
        gate_dx_max     : 门控半宽（水平方向，像素）。检测与预测位置的
                          水平距离超过这个值就不算候选。
        gate_dy_max     : 门控半高（垂直方向，像素）。
        cost_dx_weight  : 代价函数里水平偏移的权重。雨滴下落以垂直运动
                          为主，水平偏移权重调高可以让代价对"斜着连"更敏感。
        cost_dy_weight  : 代价函数里垂直偏移的权重。
        max_match_cost  : 代价超过这个值即使在门控内也不关联。
        hungarian_solver: 用哪个 HungarianSolver 实现（嵌套配置，格式同
                          顶层的 detector/tracker 段），默认 kuhn_munkres。
    """

    def __init__(
        self,
        gate_dx_max: float = 25.0,
        gate_dy_max: float = 180.0,
        cost_dx_weight: float = 3.0,
        cost_dy_weight: float = 1.0,
        max_match_cost: float = 250.0,
        hungarian_solver: dict | None = None,
    ):
        self.gate_dx_max = float(gate_dx_max)
        self.gate_dy_max = float(gate_dy_max)
        self.cost_dx_weight = float(cost_dx_weight)
        self.cost_dy_weight = float(cost_dy_weight)
        self.max_match_cost = float(max_match_cost)
        self.solver = build_hungarian_solver(
            hungarian_solver or {"name": "kuhn_munkres", "params": {}}
        )

    def associate(
        self,
        predicted_positions: list[tuple[float, float]],
        detections: list[Detection],
    ) -> AssociationResult:
        n_tracks = len(predicted_positions)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return AssociationResult(
                matches=[], match_costs={}, match_methods={},
                unmatched_tracks=list(range(n_tracks)),
                unmatched_detections=list(range(n_dets)),
                n_candidate_conflicts=0,
            )

        gated = np.zeros((n_tracks, n_dets), dtype=bool)
        cost = np.full((n_tracks, n_dets), np.inf)

        for i, (px, py) in enumerate(predicted_positions):
            for j, det in enumerate(detections):
                dx = det.cx - px
                dy = det.cy - py
                if abs(dx) > self.gate_dx_max or abs(dy) > self.gate_dy_max:
                    continue
                c = float(np.hypot(dx * self.cost_dx_weight, dy * self.cost_dy_weight))
                if c > self.max_match_cost:
                    continue
                gated[i, j] = True
                cost[i, j] = c

        matches: list[tuple[int, int]] = []
        match_costs: dict[tuple[int, int], float] = {}
        match_methods: dict[tuple[int, int], str] = {}
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        n_conflicts = 0

        for track_idxs, det_idxs in _connected_components(gated):
            if len(track_idxs) == 1 and len(det_idxs) == 1:
                i, j = track_idxs[0], det_idxs[0]
                matches.append((i, j))
                match_costs[(i, j)] = cost[i, j]
                match_methods[(i, j)] = "gated_nn"
                matched_tracks.add(i)
                matched_dets.add(j)
                continue

            # 冲突组：多条轨迹和/或多个检测互相竞争，交给 HungarianSolver
            # 在这个小组内求全局最优，而不是谁先谁得的贪心规则。
            n_conflicts += 1
            sub_cost = cost[np.ix_(track_idxs, det_idxs)]
            for local_i, local_j in self.solver.solve(sub_cost):
                i, j = track_idxs[local_i], det_idxs[local_j]
                if not np.isfinite(cost[i, j]):
                    # Hungarian 要求方阵/矩形阵都填满数字，组内没有真正被
                    # 门控允许的组合会被替换成哨兵大数（见 HungarianSolver），
                    # 这里按真实代价矩阵二次核实，不是门控内的组合一律丢弃。
                    continue
                matches.append((i, j))
                match_costs[(i, j)] = cost[i, j]
                match_methods[(i, j)] = "hungarian"
                matched_tracks.add(i)
                matched_dets.add(j)

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

        return AssociationResult(
            matches=matches,
            match_costs=match_costs,
            match_methods=match_methods,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
            n_candidate_conflicts=n_conflicts,
        )


def _connected_components(gated: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """把"轨迹-检测"门控矩阵看成二分图，找出所有连通分量。

    没有任何候选的轨迹/检测天然不出现在任何分量里，调用方按"没被
    matched_tracks/matched_dets 收录"直接判定为未关联，不需要这里
    额外处理孤立点。
    """
    n_tracks, n_dets = gated.shape
    visited_t = [False] * n_tracks
    visited_d = [False] * n_dets
    components: list[tuple[list[int], list[int]]] = []

    for start in range(n_tracks):
        if visited_t[start] or not gated[start].any():
            continue

        comp_t, comp_d = [start], []
        visited_t[start] = True
        stack: list[tuple[str, int]] = [("t", start)]

        while stack:
            kind, idx = stack.pop()
            if kind == "t":
                for j in range(n_dets):
                    if gated[idx, j] and not visited_d[j]:
                        visited_d[j] = True
                        comp_d.append(j)
                        stack.append(("d", j))
            else:
                for i in range(n_tracks):
                    if gated[i, idx] and not visited_t[i]:
                        visited_t[i] = True
                        comp_t.append(i)
                        stack.append(("t", i))

        components.append((comp_t, comp_d))

    return components
