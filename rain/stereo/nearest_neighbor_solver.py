"""NearestNeighborSolver 实现：简单场景（没有候选冲突）下的贪心最近邻分配。

"简单场景"指二分图连通分量里只有 1 条 A 候选、1 条 B 候选——这种情况
最近邻和最优分配（Hungarian）结果必然相同，用最近邻更省一次矩阵求解。
真正存在竞争的分量交给 HungarianSolver（复用阶段3已实现的
rain/tracking/hungarian_solver.py，见 StereoMatcher 的组合逻辑）。
"""

from __future__ import annotations

import numpy as np

from ..registry import register_nn_solver


@register_nn_solver("greedy_nearest_neighbor")
class NearestNeighborSolver:
    """贪心最近邻：每次在剩余候选里选全局代价最小的一对，直到没有可选的。"""

    def solve(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        if cost_matrix.size == 0:
            return []

        cost = cost_matrix.copy()
        n_rows, n_cols = cost.shape
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        matches: list[tuple[int, int]] = []

        n_possible = min(n_rows, n_cols)
        for _ in range(n_possible):
            # 在所有还没用过的行列组合里找当前最小代价（忽略 inf/nan）
            best = None
            best_cost = np.inf
            for i in range(n_rows):
                if i in used_rows:
                    continue
                for j in range(n_cols):
                    if j in used_cols:
                        continue
                    c = cost[i, j]
                    if np.isfinite(c) and c < best_cost:
                        best_cost = c
                        best = (i, j)
            if best is None:
                break  # 剩下的全是 inf/nan，没有可选的候选了
            matches.append(best)
            used_rows.add(best[0])
            used_cols.add(best[1])

        matches.sort()
        return matches
