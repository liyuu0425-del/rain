"""MorphologyProcessor 实现：形态学处理（阶段2拆分自 simple_threshold.py 第3步）。

默认行为和原 Demo 完全一致：椭圆核开运算（先腐蚀后膨胀），去掉 1~2 像素
的孤立噪点，同时轻微填充雨滴内部的小孔。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..registry import register_morphology


@register_morphology("morphology_processor")
class MorphologyProcessor:
    """形态学开运算。

    参数：
        open_ksize : 核尺寸（正方形边长）。3 能去 1~2 像素噪点；
                     5 会开始抹掉小雨滴，需要用真实数据验证再调大。
    """

    def __init__(self, open_ksize: int = 3):
        self.open_ksize = int(open_ksize)
        # 核只建一次，跨帧复用，避免每帧都重新分配。
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.open_ksize, self.open_ksize)
        )

    def process(self, binary: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
