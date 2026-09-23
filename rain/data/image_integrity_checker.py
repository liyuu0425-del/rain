"""ImageIntegrityChecker：扫描一批 TIF 文件，检查能否正常解码、格式是否一致。

这是"阶段1数据读取"的验收内容，不是检测算法——它只回答"这批文件本身
靠不靠谱"，不看图像内容里有没有雨滴。因此故意不依赖 rain/detectors 或
rain/readers 的任何检测逻辑，只用 OpenCV 做最基础的解码。

用 `cv2.IMREAD_UNCHANGED` 而不是项目里检测器常用的 `IMREAD_GRAYSCALE`：
    检测流程默认按灰度读是因为它明确知道这批数据是 8 位灰度图，可以放心
    简化处理；但"审计"这一步恰恰要反过来验证这个假设本身成不成立
    （真的是单通道 8 位吗？有没有混进 16 位或彩色文件？），
    所以必须用"不做任何猜测的原始解码"。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .file_hash import sha256_of_file
from .types import ImageFileReport, ImageIntegritySummary

_IMAGE_EXT = ".tif"


class ImageIntegrityChecker:
    """扫描一路相机目录下的所有 TIF，逐个尝试解码并记录格式信息。"""

    def scan_camera(self, camera: str, folder: str) -> list[ImageFileReport]:
        """扫描单路目录，返回每个文件的检查结果（不抛异常中断整批）。

        参数：
            camera : "a" 或 "b"，只用于在结果里标注来源，不影响解码逻辑
            folder : 图像所在目录
        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"图像目录不存在: {folder}")

        reports: list[ImageFileReport] = []
        filenames = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith(_IMAGE_EXT))

        for fn in filenames:
            path = os.path.join(folder, fn)
            reports.append(self._check_one(camera, fn, path))

        return reports

    @staticmethod
    def _check_one(camera: str, filename: str, path: str) -> ImageFileReport:
        file_size = os.path.getsize(path)

        if file_size == 0:
            return ImageFileReport(
                camera=camera, filename=filename, path=path,
                ok=False, error="文件为空（0 字节）",
                width=None, height=None, dtype=None, n_channels=None,
                file_size_bytes=0, sha256=sha256_of_file(path),
            )

        # try / except：先假设解码能成功，失败了再在 except 里记录原因，
        # 而不是解码之前先猜"这个文件是不是坏的"——猜不如直接试。
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError("cv2.imdecode 返回 None（文件可能损坏或格式不支持）")
        except Exception as exc:  # noqa: BLE001 - 审计脚本要能扫完整批，不能被一个坏文件打断
            return ImageFileReport(
                camera=camera, filename=filename, path=path,
                ok=False, error=f"{type(exc).__name__}: {exc}",
                width=None, height=None, dtype=None, n_channels=None,
                file_size_bytes=file_size, sha256=sha256_of_file(path),
            )

        # img.shape 是 (H, W) 表示单通道灰度图，(H, W, C) 表示多通道。
        # 用 len(img.shape) 判断，不假设一定是哪一种。
        height, width = img.shape[0], img.shape[1]
        n_channels = img.shape[2] if len(img.shape) == 3 else 1

        return ImageFileReport(
            camera=camera, filename=filename, path=path,
            ok=True, error=None,
            width=int(width), height=int(height),
            dtype=str(img.dtype), n_channels=int(n_channels),
            file_size_bytes=file_size, sha256=sha256_of_file(path),
        )


def summarize(reports_a: list[ImageFileReport], reports_b: list[ImageFileReport]) -> ImageIntegritySummary:
    """把 A、B 两路的逐文件检查结果合并成一份汇总统计。

    这是一个模块级函数（不属于任何 class），因为它不需要保存状态，
    纯粹是"输入一堆结果，算出统计数字"，写成类反而多余。
    """
    all_reports = reports_a + reports_b  # 列表相加 = 拼接成一个更长的列表

    n_total = len(all_reports)
    n_ok = sum(1 for r in all_reports if r.ok)
    n_corrupt_or_empty = n_total - n_ok

    resolution_counts: dict[str, int] = {}
    dtype_channel_counts: dict[str, int] = {}
    for r in all_reports:
        if not r.ok:
            continue
        res_key = f"{r.width}x{r.height}"
        resolution_counts[res_key] = resolution_counts.get(res_key, 0) + 1
        dc_key = f"{r.dtype}_ch{r.n_channels}"
        dtype_channel_counts[dc_key] = dtype_channel_counts.get(dc_key, 0) + 1

    # A/B 同名文件如果 sha256 完全相同，说明两台相机存下来的是同一份字节内容——
    # 物理上不可能（两个不同光路，画面不会逐字节相同），几乎可以肯定是
    # 采集或拷贝流程出了 bug（例如误把 A 帧存了两份）。这里只检测、报告，
    # 不做任何修复。
    hash_a = {r.filename: r.sha256 for r in reports_a if r.ok}
    hash_b = {r.filename: r.sha256 for r in reports_b if r.ok}
    identical_ab_filenames = sorted(
        fn for fn in (hash_a.keys() & hash_b.keys())  # 集合交集：两路都有且都解码成功的文件名
        if hash_a[fn] == hash_b[fn]
    )

    return ImageIntegritySummary(
        n_total=n_total,
        n_ok=n_ok,
        n_corrupt_or_empty=n_corrupt_or_empty,
        resolution_counts=resolution_counts,
        dtype_channel_counts=dtype_channel_counts,
        identical_ab_filenames=identical_ab_filenames,
        per_file=all_reports,
    )
