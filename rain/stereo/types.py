"""阶段4数据契约：双路标定、候选匹配诊断、匹配结果。

和 rain/data/types.py、rain/tracking/types.py 是同一类东西，单独一份
服务于阶段4，不牵动其它阶段已经跑通的代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StereoCalibration:
    """双相机标定数据（从标定文件加载）。

    fundamental_matrix : 基础矩阵 F，3x3。用于把 A 图像里的一个点映射成
                          B 图像里的一条极线：line = F @ [ax, ay, 1]。
                          这是真正做极线约束所需的最小标定信息。
    rectify_a_to_b      : 可选的 3x3 变换，把 A 图像坐标近似映射到 B 图像
                          坐标系，供"质心位置差"这个代价项使用。
                          正交双目（A、B 视角相差约90度）不存在单应矩阵
                          意义上的严格几何对应，这里只是一个可标定的
                          经验映射占位——没有标定文件时这一项就不可用，
                          不能瞎猜一个矩阵出来。
    source_path         : 标定文件路径，写进报告方便追溯。
    """

    fundamental_matrix: np.ndarray
    rectify_a_to_b: np.ndarray | None
    source_path: str


@dataclass
class StereoMatchRow:
    """stereo_matches.csv 的一行：一个候选对（可能匹配成功，也可能失败）。"""

    pair_index: int
    a_index: int | None
    b_index: int | None
    a_cx: float | None
    a_cy: float | None
    b_cx: float | None
    b_cy: float | None
    epipolar_distance_px: float | None
    cost_epipolar: float | None
    cost_position: float | None
    cost_area: float | None
    cost_aspect: float | None
    total_cost: float | None
    gate_epipolar_passed: bool
    gate_match_passed: bool
    method: str  # nearest_neighbor | hungarian | none
    confidence: str  # high | medium | low | none
    status: str  # matched | unmatched_a | unmatched_b
    reason: str


@dataclass
class CostBreakdown:
    """CostMatrixBuilder 对单个候选对算出的总代价 + 各子代价（均已归一化）。

    某个子代价不可用时（比如没有 rectify_a_to_b，位置差算不出来），
    对应字段是 None，权重会在其余可用项之间重新归一化，不是当成 0 参与计算
    ——当成 0 等于"这一项完美匹配"，是错误的默认行为。
    """

    cost_epipolar: float | None
    cost_position: float | None
    cost_area: float | None
    cost_aspect: float | None
    total_cost: float


@dataclass
class StereoFrameResult:
    """StereoMatcher.match_frame() 的返回值：一帧的全部候选诊断 + 计时。"""

    rows: list[StereoMatchRow] = field(default_factory=list)
    n_candidate_conflicts: int = 0
    gate_ms: float = 0.0
    cost_ms: float = 0.0
    solve_ms: float = 0.0
