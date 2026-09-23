"""阶段3三个跟踪子模块 + HungarianSolver 的接口定义。

和 rain/interfaces.py、rain/detection/interfaces.py 是同一类东西，
用 Protocol 定义"这个模块必须提供哪个方法、输入输出是什么"。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import Detection
from .types import AssociationResult, FrameContext, KalmanState, StepResult, TrackRecord


@runtime_checkable
class KalmanPredictor(Protocol):
    """卡尔曼滤波：给轨迹的位置状态做预测和量测更新。

    只负责"数学"，不知道什么是轨迹、什么是帧——这一层只有三个纯函数式方法。
    """

    def init_state(self, cx: float, cy: float) -> KalmanState:
        """用第一个检测点初始化一条新轨迹的状态（速度未知，设为0，
        协方差给一个很大的初始不确定度）。"""
        ...

    def predict(self, state: KalmanState, dt: float) -> KalmanState:
        """把状态外推 dt 秒（dt 来自真实设备时间差，不是固定帧间隔）。

        返回新状态，不修改传入的 state（其它轨迹或线程可能还持有它）。
        """
        ...

    def update(self, state: KalmanState, cx: float, cy: float) -> KalmanState:
        """用一个真实检测位置修正预测状态（卡尔曼量测更新）。

        返回新状态，不修改传入的 state。
        """
        ...


@runtime_checkable
class HungarianSolver(Protocol):
    """最优分配：给一个代价矩阵，求全局代价最小的一一匹配。"""

    def solve(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        """返回 (行下标, 列下标) 匹配对列表。

        矩阵可以不是方阵；返回的匹配数量等于 min(行数, 列数)。
        代价用一个很大的有限数表示"不可选"，不要传 inf（数值算法可能因此出错）。
        """
        ...


@runtime_checkable
class TrackAssociator(Protocol):
    """门控 + 最近邻/匈牙利关联：把"现有轨迹的预测位置"和"当前帧的检测"配对。"""

    def associate(
        self,
        predicted_positions: list[tuple[float, float]],
        detections: list[Detection],
    ) -> AssociationResult:
        """
        参数：
            predicted_positions : 每条活跃轨迹的预测位置 (cx, cy)，
                                   下标即该轨迹在这次调用里的编号
            detections           : 当前帧的检测列表，下标即该检测在这次
                                   调用里的编号
        约定：
            超过代价阈值的候选不得强行配对，必须落在 unmatched_* 里。
        """
        ...


@runtime_checkable
class TrackManager(Protocol):
    """轨迹生命周期管理：创建、更新、暂时丢失、结束，并保证 track_id 稳定。

    内部持有一个 KalmanPredictor 和一个 TrackAssociator（构造时通过配置注入，
    两者都可独立替换），按帧驱动。这是阶段3三个模块里唯一"有状态"的一个——
    它记着所有轨迹的当前生死状态，前两个模块都是无状态的纯计算。
    """

    def step(self, frame: FrameContext) -> StepResult:
        """处理一帧：预测所有活跃轨迹 -> 关联 -> 更新/创建/丢失/结束。

        必须按 frame.pair_index 递增的顺序依次调用——这个方法本身
        不做排序，排序是调用方（pipeline/stage3_tracking.py）的职责。
        """
        ...

    def finalize(self) -> list[TrackRecord]:
        """处理完所有帧后调用一次，把仍处于丢失宽限期的轨迹也结算掉，
        返回全部轨迹（含正常结束的和被强制结算的）。"""
        ...

    def get_summary_counters(self) -> dict:
        """返回运行至今的汇总计数：丢失次数、候选冲突次数、未关联检测数量等，
        供阶段3验收报告直接使用，不用外部重新统计一遍。"""
        ...
