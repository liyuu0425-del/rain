"""AdaptiveThresholdSegmenter 实现：阈值分割（阶段2拆分自 simple_threshold.py 第2步）。

支持两类方法，用 `method` 参数切换：
    "fixed"              : 全局固定阈值，`cv2.threshold`。和原 Demo 完全
                            一致的默认行为，用于回归对比。
    "adaptive_mean"      : `cv2.adaptiveThreshold`，邻域均值法。
    "adaptive_gaussian"  : `cv2.adaptiveThreshold`，邻域高斯加权法。

真正的自适应方法是这次阶段2新增的能力——原 Demo 出于性能考虑（"比
adaptiveThreshold 快 28 倍"）只实现了固定阈值，但模块名字明确要求
"自适应阈值分割器"，所以这里把两类都做成同一个类的可选行为，默认仍是
固定阈值，保证"拆分不改变原结果"这条回归测试成立，同时留出自适应阈值
的路径供后续调参对比（会不会更适应局部背光不均匀，需要真实数据验证，
不在阶段2下结论）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..registry import register_segmenter

_ADAPTIVE_METHODS = {
    "adaptive_mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "adaptive_gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


@register_segmenter("adaptive_threshold_segmenter")
class AdaptiveThresholdSegmenter:
    """阈值分割：固定阈值或自适应阈值二选一。

    参数：
        method     : "fixed"（默认） | "adaptive_mean" | "adaptive_gaussian"
        thresh     : method="fixed" 时用，差值阈值（灰阶）
        block_size : method 是自适应两种之一时用，邻域窗口边长，必须是
                     大于 1 的奇数（cv2.adaptiveThreshold 的硬性要求）
        c          : method 是自适应两种之一时用，从邻域统计量里减去的常数，
                     越大分割越保守（前景越少）
    """

    def __init__(
        self,
        method: str = "fixed",
        thresh: int = 20,
        block_size: int = 31,
        c: float = -5.0,
    ):
        if method not in ("fixed",) and method not in _ADAPTIVE_METHODS:
            raise ValueError(
                f"未知的分割方法: {method!r}，只支持 fixed | adaptive_mean | adaptive_gaussian"
            )
        if method in _ADAPTIVE_METHODS and (block_size < 3 or block_size % 2 == 0):
            raise ValueError(f"block_size 必须是 >=3 的奇数，收到 {block_size}")

        self.method = method
        self.thresh = int(thresh)
        self.block_size = int(block_size)
        self.c = float(c)

    def segment(self, diff: np.ndarray) -> np.ndarray:
        if self.method == "fixed":
            _, binary = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
            return binary

        adaptive_flag = _ADAPTIVE_METHODS[self.method]
        return cv2.adaptiveThreshold(
            diff, 255, adaptive_flag, cv2.THRESH_BINARY, self.block_size, self.c
        )
