"""DropSizeDistributionEstimator 实现：按像素等效直径分箱统计。

没有像素-毫米标定，这里的"粒径"只能是像素等效直径（Detection.d_eq_px），
分箱边界的单位也是像素。任何时候都不得把这份结果称为"雨滴谱"（那个词
专指毫米单位的标定后结果）——输出的字段名、note 字段都显式写"像素"。
"""

from __future__ import annotations

import numpy as np

from ..registry import register_size_distribution_estimator
from .types import DropSizeDistributionResult


@register_size_distribution_estimator("pixel_size_histogram")
class DropSizeDistributionEstimator:
    """像素等效直径直方图。

    参数（来自 config/pipeline.yaml 的 statistics.size_distribution.params）：
        bin_edges_px            : 分箱边界列表（像素），长度为 N+1 时产出 N 个分箱。
                                   必须严格递增，否则 np.histogram 会算出无意义的结果。
        spatial_calibration_status : 空间标定可信状态的文本标记，默认
                                   "spatial_calibration_unavailable"。从配置读取，
                                   不是代码里的永久常量——完成像素-毫米标定后，
                                   只需要在配置里改这个值。
    """

    def __init__(
        self,
        bin_edges_px: list[float] | None = None,
        spatial_calibration_status: str = "spatial_calibration_unavailable",
    ):
        edges = bin_edges_px or [0, 5, 10, 15, 20, 30, 50, 100, 99999]
        edges = [float(e) for e in edges]
        if len(edges) < 2:
            raise ValueError(f"bin_edges_px 至少需要2个边界(1个分箱)，收到 {edges}")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"bin_edges_px 必须严格递增: {edges}")
        self.bin_edges_px = edges
        self.spatial_calibration_status = str(spatial_calibration_status)

    def estimate(self, camera: str, sizes_px: list[float]) -> DropSizeDistributionResult:
        if sizes_px:
            counts, _ = np.histogram(np.array(sizes_px), bins=self.bin_edges_px)
            counts_list = [int(c) for c in counts]
        else:
            counts_list = [0] * (len(self.bin_edges_px) - 1)

        return DropSizeDistributionResult(
            camera=camera,
            bin_edges_px=list(self.bin_edges_px),
            counts=counts_list,
            n_total=len(sizes_px),
            spatial_calibration_status=self.spatial_calibration_status,
        )
