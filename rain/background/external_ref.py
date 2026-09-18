"""BackgroundModel 实现：外部无雨背景参考批次。

用法：专门采集一批无雨背景，用它处理数据批次。好处是背景更干净（没有任何雨滴）。
缺点是如果采集时间相隔较远，背光漂移可能会引入误差。

实测：背景参考 (bja/Background_Ref_20260727_230651) 采于 23:06:51，
      数据批次 (cama/Capture_20260727_232420)        采于 23:24:20，
      相隔 17 分钟，平均灰度只差 0.4，可忽略。

建议：背景参考和批内分位数两条路都保留，配置可选。
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from ..registry import register_background


@register_background("external_ref")
class ExternalRefBackground:
    """从外部目录加载预先采集的无雨背景。

    参数：
        ref_dir       : 背景参考批次的目录，例如
                        D:\\毕设\\数据\\bja\\Background_Ref_20260727_230651
        reader_cfg    : 用什么 reader 读背景帧，格式同 config.yaml 的 reader 段。
                        如果背景也是 LZW TIF，写 {name: lzw_tif}；
                        如果是 raw，写 {name: raw_bin}。
        q             : 多帧背景取哪个分位数融合。0.5 = 中位数。
                        背景批次本身是无雨的，取中位数即可。
        max_frames    : 最多用多少帧参与融合，均匀抽样。
    """

    def __init__(
        self,
        ref_dir: str,
        reader_cfg: dict,
        q: float = 0.5,
        max_frames: int = 32,
    ):
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(f"背景参考目录不存在: {ref_dir}")
        self.ref_dir = ref_dir
        self.reader_cfg = dict(reader_cfg)
        self.q = float(q)
        self.max_frames = int(max_frames)
        self._cached_bg: np.ndarray | None = None

    def build(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        # frames 参数在这个实现里用不上（因为背景来自外部），
        # 但接口要求有这个参数，所以我们保留它。
        # 把 frames 转成 list 是为了满足"只遍历一次"的契约（虽然这里不遍历）。
        _ = list(frames)

        # 背景只建一次，后续复用缓存
        if self._cached_bg is not None:
            return self._cached_bg

        # 延迟 import，避免循环依赖（registry -> interfaces -> types）
        from ..registry import build_reader

        reader = build_reader(self.reader_cfg)

        # 扫描背景目录，找所有 .tif / .raw 文件
        all_files = sorted(
            f
            for f in os.listdir(self.ref_dir)
            if f.lower().endswith((".tif", ".tiff", ".raw"))
        )
        if not all_files:
            raise FileNotFoundError(f"背景参考目录里没有图像: {self.ref_dir}")

        paths = [os.path.join(self.ref_dir, f) for f in all_files]

        # 均匀抽样
        n = len(paths)
        if n > self.max_frames:
            idx = np.linspace(0, n - 1, self.max_frames).astype(int)
            sampled = [paths[i] for i in idx]
        else:
            sampled = paths

        imgs = [reader.read(p) for p in sampled]
        stack = np.stack(imgs)

        if abs(self.q - 0.5) < 1e-9:
            bg = np.median(stack, axis=0)
        else:
            bg = np.quantile(stack, self.q, axis=0)

        self._cached_bg = bg.astype(np.uint8)
        return self._cached_bg
