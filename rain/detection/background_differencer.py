"""BackgroundDifferencer 实现：背景差分（阶段2拆分自 simple_threshold.py 第1步）。

和原 Demo 完全一致的默认行为：`cv2.subtract(bg, img)`（饱和减法，单向，
只检测"比背景暗"的目标，因为背光成像下雨滴是暗目标）。额外提供 "abs"
模式（`cv2.absdiff`）留作对比，默认不用——原 Demo docstring 明确解释过
为什么不用 absdiff（会把背光的双向噪声也当前景）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..registry import register_differencer


@register_differencer("background_differencer")
class BackgroundDifferencer:
    """背景差分。

    参数：
        mode : "bg_minus_img"（默认，与原 Demo 一致）| "abs"
    """

    def __init__(self, mode: str = "bg_minus_img"):
        if mode not in ("bg_minus_img", "abs"):
            raise ValueError(f"未知的差分模式: {mode!r}，只支持 bg_minus_img | abs")
        self.mode = mode

    def diff(self, img: np.ndarray, bg: np.ndarray) -> np.ndarray:
        if self.mode == "bg_minus_img":
            return cv2.subtract(bg, img)
        return cv2.absdiff(bg, img)
