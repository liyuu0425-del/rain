"""FrameReader 实现：基于 PIL(Pillow) 的 TIF 解码（阶段7B 性能优化新增）。

背景：用 rain/perf/tiff_inspector.py 直接解析这批数据(A/B-Capture_085820)
的 TIFF IFD 标签验证过（不是猜的）——全部是 Compression=LZW(5)、
Predictor=2(水平差分)、RowsPerStrip=3。针对同一批文件，实测 PIL 解码比
`rain/readers/lzw_tif.py` 用的 `cv2.imdecode` 快约15-20%，180帧样本
逐像素比对完全一致（np.array_equal全部为True）。

这是**新增**的一个可选读取器，不改 lzw_tif.py 一个字节——遵循项目
"在对应子包里新增实现类、用注册装饰器登记"的扩展方式（README已有说明），
换 reader 只需要改 config 的 name，不改代码。

对外契约和 LzwTifReader 完全一致：返回 uint8 灰度数组，失败即抛异常，
不返回 None。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..registry import register_reader


@register_reader("fast_tif")
class FastTifReader:
    """PIL(Pillow) 解码的灰度 TIF 读取器。

    参数：
        expect_shape: 可选的 (H, W)，校验用，含义和 LzwTifReader 完全一样。
    """

    def __init__(self, expect_shape: tuple[int, int] | None = None):
        self.expect_shape = tuple(expect_shape) if expect_shape else None

    def read(self, path: str) -> np.ndarray:
        # PIL 的 Image.open 用 Python 内置的 open()，原生支持 Windows 的
        # Unicode 路径（不像 cv2.imread 那样只认 ASCII），所以这里不需要
        # 像 rain/imageio.py 那样绕一圈"读字节再解码"。
        with Image.open(path) as im:
            img = np.asarray(im)

        if img.dtype != np.uint8:
            raise IOError(f"读到的不是 uint8 灰度数组: {path}，dtype={img.dtype}")
        if img.ndim != 2:
            raise IOError(f"读到的不是单通道灰度图: {path}，shape={img.shape}")
        if self.expect_shape is not None and img.shape != self.expect_shape:
            raise ValueError(f"尺寸不符: {path} 是 {img.shape}，期望 {self.expect_shape}")
        return img
