"""BrightnessNormalizer 实现：逐帧亮度归一化（阶段2新增，原 Demo 没有这一步）。

方法只有两种，用 `method` 参数切换：
    "none"       : 原样返回（不做任何处理）。这是默认值，目的是让
                   ModularPipelineDetector 在这个参数下和原
                   SimpleThresholdDetector 逐帧结果完全一致，用来做
                   "拆分后结果不变"的回归测试。
    "mean_match" : 把整帧的平均灰度线性缩放到目标值 target_mean，
                   补偿逐帧亮度波动（背光/曝光的小幅漂移）。
"""

from __future__ import annotations

import numpy as np

from ..registry import register_normalizer


@register_normalizer("brightness_normalizer")
class BrightnessNormalizer:
    """逐帧亮度归一化。

    参数：
        method      : "none" | "mean_match"
        target_mean : method="mean_match" 时必填。整帧目标平均灰度，
                      通常取该批数据背景图的平均灰度（离线算好填进配置，
                      不在这里现算——现算需要看到整批帧，超出"单帧输入"
                      这个接口的职责范围）。
    """

    def __init__(self, method: str = "none", target_mean: float | None = None):
        if method not in ("none", "mean_match"):
            raise ValueError(f"未知的归一化方法: {method!r}，只支持 none | mean_match")
        if method == "mean_match" and target_mean is None:
            raise ValueError("method='mean_match' 时必须提供 target_mean")
        self.method = method
        self.target_mean = target_mean

    def normalize(self, img: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return img

        # method == "mean_match"
        current_mean = float(img.mean())
        if current_mean < 1e-6:
            # 整帧接近全黑，缩放会放大到无意义的数字，原样返回更安全。
            return img

        ratio = self.target_mean / current_mean
        # 先转 float32 做乘法再截断到 [0,255]，避免 uint8 乘法溢出后回绕
        # （比如 200 * 1.5 = 300，uint8 会绕成 44，而不是我们想要的 255）。
        scaled = img.astype(np.float32) * ratio
        return np.clip(scaled, 0, 255).astype(np.uint8)
