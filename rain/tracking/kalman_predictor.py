"""KalmanPredictor 实现：等速运动模型的线性卡尔曼滤波（2D位置+速度）。

状态向量 x = [cx, cy, vx, vy]，运动模型假设短时间内速度近似恒定
（雨滴下落在相邻两帧之间、3200微秒的时间尺度上，加速度的影响可以忽略，
这是经典的"等速模型 constant velocity"，多目标跟踪里最常用的基线模型）。

只做预测和缩小候选搜索范围，不产出、不替代真实检测——这是任务书的硬性
约束，代码层面的体现是：这个类完全不知道"检测"是什么，只认位置和时间，
真正"用检测更新状态"的调用由 TrackManager 在关联成功后才触发。
"""

from __future__ import annotations

import numpy as np

from ..registry import register_kalman_predictor
from .types import KalmanState


@register_kalman_predictor("constant_velocity_kalman")
class ConstantVelocityKalmanPredictor:
    """等速模型卡尔曼滤波器。

    参数（全部来自 config/pipeline.yaml 的 tracking.kalman_predictor.params，
    不在代码里写死）：
        process_noise_pos  : 过程噪声里位置分量的强度。越大代表越不信任
                              "匀速"这个假设，允许位置有更多随机漂移。
        process_noise_vel  : 过程噪声里速度分量的强度。越大代表越允许
                              速度本身发生变化（比如雨滴受气流扰动）。
        measurement_noise  : 量测噪声，即"检测出来的质心坐标本身有多准"。
                              越大代表越不信任单次检测，更依赖历史轨迹平滑。
        init_velocity_var  : 新建轨迹时，速度分量的初始方差。给一个很大的
                              数，代表"第一个点还完全不知道往哪个方向飞"，
                              第二次关联上之后卡尔曼会很快把它修正过来。
    """

    def __init__(
        self,
        process_noise_pos: float = 1.0,
        process_noise_vel: float = 50.0,
        measurement_noise: float = 4.0,
        init_velocity_var: float = 10000.0,
    ):
        self.process_noise_pos = float(process_noise_pos)
        self.process_noise_vel = float(process_noise_vel)
        self.measurement_noise = float(measurement_noise)
        self.init_velocity_var = float(init_velocity_var)

        # 量测矩阵 H：状态是 [cx,cy,vx,vy]，量测只看得到位置 [cx,cy]，
        # 所以 H 只取状态向量的前两个分量。常量，构造一次复用。
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # 量测噪声协方差 R：假设 x、y 方向的检测误差独立同分布。
        self.R = np.eye(2) * self.measurement_noise

    def init_state(self, cx: float, cy: float) -> KalmanState:
        x = np.array([cx, cy, 0.0, 0.0])
        P = np.diag([
            self.measurement_noise, self.measurement_noise,
            self.init_velocity_var, self.init_velocity_var,
        ])
        return KalmanState(x=x, P=P)

    def predict(self, state: KalmanState, dt: float) -> KalmanState:
        if dt < 0:
            raise ValueError(f"dt 不能为负数（时间倒退？）: {dt}")

        # 状态转移矩阵 F：匀速模型下，dt 秒后 位置' = 位置 + 速度*dt，速度不变。
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # 过程噪声协方差 Q：dt 越大，累积的不确定度越大，所以整体随 dt 缩放。
        # 这是简化写法（标准做法是按运动学积分算出精确的Q(dt)），
        # 对本项目 dt 恒定在 3.2ms 量级的场景足够，且参数含义直观、易于标定。
        q_pos = self.process_noise_pos * dt
        q_vel = self.process_noise_vel * dt
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        x_new = F @ state.x
        P_new = F @ state.P @ F.T + Q
        return KalmanState(x=x_new, P=P_new)

    def update(self, state: KalmanState, cx: float, cy: float) -> KalmanState:
        z = np.array([cx, cy])

        # 标准卡尔曼量测更新公式：
        #   y = z - Hx          （新息/创新：量测和预测的差）
        #   S = H P H^T + R     （新息协方差）
        #   K = P H^T S^-1      （卡尔曼增益）
        #   x' = x + K y
        #   P' = (I - K H) P
        y = z - self.H @ state.x
        S = self.H @ state.P @ self.H.T + self.R
        K = state.P @ self.H.T @ np.linalg.inv(S)

        x_new = state.x + K @ y
        P_new = (np.eye(4) - K @ self.H) @ state.P
        return KalmanState(x=x_new, P=P_new)
