"""Rain 算法包。

把子模块的核心对象暴露到顶层，方便导入：
    from rain import Detection, build_detector
而不用
    from rain.types import Detection
    from rain.registry import build_detector
"""

# 数据契约
from .types import (
    BatchPair,
    BatchResult,
    Detection,
    Track,
    QC_MERGED,
    QC_ODD_SHAPE,
    QC_OK,
    QC_SATURATED,
    QC_STATIC,
    QC_TOO_LARGE,
    QC_TOO_SMALL,
    QC_TOUCHES_BORDER,
    qc_to_text,
)

# 注册表与工厂
from .registry import (
    build_background,
    build_detector,
    build_reader,
    build_stereo,
    build_tracker,
    describe,
)

# 显式导入实现模块，让装饰器执行、注册生效
from .readers import lzw_tif, raw_bin
from .background import batch_quantile, external_ref
from .detectors import simple_threshold
from .trackers import gravity_gate, nn_gate

__all__ = [
    # 数据契约
    "Detection",
    "Track",
    "BatchPair",
    "BatchResult",
    "QC_OK",
    "QC_TOUCHES_BORDER",
    "QC_MERGED",
    "QC_TOO_SMALL",
    "QC_TOO_LARGE",
    "QC_ODD_SHAPE",
    "QC_SATURATED",
    "QC_STATIC",
    "qc_to_text",
    # 工厂
    "build_reader",
    "build_background",
    "build_detector",
    "build_tracker",
    "build_stereo",
    "describe",
]
