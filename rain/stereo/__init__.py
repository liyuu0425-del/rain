"""阶段4双路匹配包：极线约束、匹配门控、代价矩阵、最近邻/匈牙利分配。

HungarianSolver 复用 rain/tracking/hungarian_solver.py 里已经实现并
验证过的 Kuhn-Munkres 算法（同一张 HUNGARIAN_SOLVERS 注册表），
这里不重新实现一遍。
"""

from __future__ import annotations

from . import cost_matrix_builder, epipolar_gate, match_gate, nearest_neighbor_solver, stereo_matcher

__all__ = [
    "epipolar_gate",
    "match_gate",
    "cost_matrix_builder",
    "nearest_neighbor_solver",
    "stereo_matcher",
]
