"""HungarianSolver 实现：Kuhn-Munkres 最优分配算法（O(n^3)）。

用于 TrackAssociator 在"候选冲突"（一个检测同时落进多条轨迹的门控范围，
或一条轨迹同时有多个候选检测）时求全局代价最小的一一匹配——而不是简单
按"谁先谁得"的贪心规则，贪心在多个候选互相竞争时容易连出次优甚至
明显错误的配对。

这是一个经典算法，如果自己现场推导容易出边界条件的 bug，这里用的是
业界广泛验证过的"势 (potential) + 增广路"实现（也叫 Jonker-Volgenant
风格的 O(n^3) 版本），只是重新用带中文注释的方式抄写一遍，逻辑与
标准实现完全一致，正确性由 tests/test_tracking_stage3.py 里"和暴力
穷举结果对拍"的测试保证。
"""

from __future__ import annotations

import numpy as np

from ..registry import register_hungarian_solver

_INF = 1e12
# 矩阵里的"不可选"代价必须是一个很大的有限数，不能是 inf——
# 算法内部要做减法（cur = cost - u - v），inf 参与减法会变成 nan，
# 直接让整个求解过程失效。1e6 比业务里任何真实代价（几十到几百像素）
# 大几个数量级，效果等同于"绝对不会被选中，除非没有其它选择"。
_SENTINEL_COST = 1e6


@register_hungarian_solver("kuhn_munkres")
class HungarianSolver:
    """最小代价二分图最优匹配（Kuhn-Munkres / 匈牙利算法）。"""

    def solve(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        if cost_matrix.size == 0:
            return []

        n_rows, n_cols = cost_matrix.shape

        # 算法要求"行数 <= 列数"，行多于列时转置求解、结果再转置回来。
        if n_rows > n_cols:
            transposed = self._solve_square_or_wide(cost_matrix.T)
            return [(j, i) for (i, j) in transposed]
        return self._solve_square_or_wide(cost_matrix)

    @staticmethod
    def _solve_square_or_wide(cost: np.ndarray) -> list[tuple[int, int]]:
        """要求 n_rows <= n_cols。返回每一行匹配到的列（(row, col) 列表）。

        把无穷大 / 非有限值先替换成一个足够大的哨兵值，避免参与数值运算时
        出现 nan（见模块开头的说明）。
        """
        cost = np.where(np.isfinite(cost), cost, _SENTINEL_COST).astype(np.float64)
        n, m = cost.shape  # n <= m

        # 势（potential）数组，1-indexed 少一次 -1 的心智负担，和经典实现一致。
        u = np.zeros(n + 1)
        v = np.zeros(m + 1)
        # p[j] = 当前分配给列 j 的行号（1-indexed），0 代表"还没分配"。
        p = np.zeros(m + 1, dtype=int)
        way = np.zeros(m + 1, dtype=int)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, _INF)
            used = np.zeros(m + 1, dtype=bool)

            # 不断寻找增广路径，直到找到一个当前未分配的列。
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = _INF
                j1 = -1
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

                # 用这一轮找到的最小松弛量更新势，保持"已访问列"对应的匹配
                # 代价不变、其余列的可行性同步收紧——这是 O(n^3) 复杂度的关键。
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            # 沿着 way[] 记录的路径，把这条增广路上的分配都翻转一遍。
            while j0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1

        result = []
        for j in range(1, m + 1):
            if p[j] != 0:
                result.append((p[j] - 1, j - 1))
        result.sort()
        return result
