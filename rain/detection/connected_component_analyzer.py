"""ConnectedComponentAnalyzer 实现：连通域分析（阶段2拆分自 simple_threshold.py 第4步）。

和原 Demo 一样用 `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`：
只要最外层轮廓（忽略雨滴内部的孔，那不是独立目标），轮廓点做直线段压缩
（省内存，不影响面积/周长这类整体特征的精度）。

为什么不用 `cv2.connectedComponentsWithStats`：
    原 Demo 的 docstring 已经记录了实测结论——findContours 快 7.5 倍，
    而且直接给出轮廓点，下一步 ContourExtractor 要拟合椭圆、算周长，
    正好需要轮廓点而不是 label 图。这里延续同样的选择，输出格式也
    保持和 `cv2.findContours` 完全一致（`list[np.ndarray]`，每个
    shape=(n,1,2) int32），方便直接喂给 ContourExtractor。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..registry import register_cc_analyzer


@register_cc_analyzer("connected_component_analyzer")
class ConnectedComponentAnalyzer:
    """从二值图里找连通域，返回每个连通域的外轮廓点集。"""

    def analyze(self, binary: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)
