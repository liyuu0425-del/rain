"""阶段3跨帧跟踪包：卡尔曼预测、门控关联、匈牙利求解、轨迹生命周期管理。

和 `rain/trackers/`（gravity_gate、nn_gate，老 Demo 里"下一帧位移门控"的
简单实现）是两套并存、互不影响的东西：老实现继续留着给 tools/demo.py 用
（任务要求"不得修改原Demo"）；这里是阶段3要求的、基于真实卡尔曼状态估计
的新跟踪系统，服务于 pipeline/stage3_tracking.py。
"""

from __future__ import annotations

from . import hungarian_solver, kalman_predictor, track_associator, track_manager

__all__ = [
    "kalman_predictor",
    "hungarian_solver",
    "track_associator",
    "track_manager",
]
