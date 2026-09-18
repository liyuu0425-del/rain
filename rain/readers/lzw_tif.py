"""FrameReader 实现：LZW 压缩 TIF（历史数据格式）。

实测性能（2064x1544 uint8，单核）：
    读盘 928 KB   0.16 ms
    LZW 解码     20.93 ms
    ------------------
    合计         21.1 ms/帧

这是整条流水线最贵的一环，占精简流水线总耗时约 70%。
新数据建议改存未压缩格式，见 raw_bin.py。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imageio import imread_unicode
from ..registry import register_reader


@register_reader("lzw_tif")
class LzwTifReader:
    """读 LZW 压缩的灰度 TIF。

    参数：
        expect_shape: 可选的 (H, W)。填了就会校验每帧尺寸，
                      尺寸不符直接报错，避免坏文件悄悄污染统计结果。
    """

    def __init__(self, expect_shape: tuple[int, int] | None = None):
        self.expect_shape = tuple(expect_shape) if expect_shape else None

    def read(self, path: str) -> np.ndarray:
        # 为什么不用 cv2.imread？它在 Windows 上打不开中文路径，
        # 而且只返回 None 不抛异常。详见 rain/imageio.py 的说明。
        img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)

        if self.expect_shape is not None and img.shape != self.expect_shape:
            raise ValueError(
                f"尺寸不符: {path} 是 {img.shape}，期望 {self.expect_shape}"
            )
        return img
