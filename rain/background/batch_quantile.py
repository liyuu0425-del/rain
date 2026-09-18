"""BackgroundModel 实现：用批次自己的帧算逐像素分位数。

原理：
    背光成像下雨滴是**暗目标**（挡住背光）。实测一帧里
    比中位数暗 15 灰阶以上的像素有 627 个，比中位数亮 15 以上的有 0 个——
    偏离完全是单向的。
    所以对每个像素取时间轴上的高分位数，就能得到"没有雨滴挡住时"的亮度。

内存注意：
    一批 180 帧 × 3.19 MB = 573 MB，全部堆进内存再算分位数太重，
    而且每个并行 worker 都要占这么多。
    默认用 stride 抽样最多 32 帧（约 102 MB），对缓慢变化的背光足够。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..registry import register_background


@register_background("batch_quantile")
class BatchQuantileBackground:
    """从一批帧里按分位数合成背景图。

    参数：
        q            : 分位数，0..1。雨滴是暗目标，所以取偏高的分位数。
                       0.5 = 中位数，稀疏降雨够用；
                       0.75~0.9 在密集降雨下更稳（雨滴更难把分位数拉暗）。
        max_frames   : 最多用多少帧参与计算，抽样均匀铺满整批。
                       调大更稳但更吃内存和时间。
    """

    def __init__(self, q: float = 0.5, max_frames: int = 32):
        if not 0.0 < q < 1.0:
            raise ValueError(f"q 必须在 (0,1) 之间，收到 {q}")
        self.q = float(q)
        self.max_frames = int(max_frames)

    def build(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        # frames 可能是生成器，只能遍历一次，先收成列表。
        # 这里收的是"引用"，不拷贝像素，所以本身不额外占内存。
        all_frames = list(frames)
        if not all_frames:
            raise ValueError("背景建模收到空的帧列表")

        # 均匀抽样：比如 180 帧抽 32 帧，取 step=5，覆盖整批而不是只看开头。
        n = len(all_frames)
        if n > self.max_frames:
            idx = np.linspace(0, n - 1, self.max_frames).astype(int)
            sampled = [all_frames[i] for i in idx]
        else:
            sampled = all_frames

        stack = np.stack(sampled)  # shape = (n_sampled, H, W)

        if abs(self.q - 0.5) < 1e-9:
            # 中位数走专用函数，比通用 quantile 快
            bg = np.median(stack, axis=0)
        else:
            bg = np.quantile(stack, self.q, axis=0)

        # 回到 uint8：后续所有算子都吃 uint8，用 float 会让内存带宽翻 4 倍
        return bg.astype(np.uint8)
