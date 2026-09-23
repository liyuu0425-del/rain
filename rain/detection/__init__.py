"""阶段2检测子模块包：亮度归一化、背景差分、自适应阈值、形态学、连通域、轮廓提取。

这六个文件把原来揉在 `rain/detectors/simple_threshold.py` 一个 `detect()`
方法里的流程拆成独立、可单独测试、可单独替换的模块。原文件保持不动
（任务要求"不得修改检测算法"），这里是纯新增。

六步的组合方式由 `rain/detectors/modular_pipeline.py` 里的
`ModularPipelineDetector` 负责——那个类本身不包含任何算法实现，
只按顺序调用这六个模块，同时仍然满足 `rain/interfaces.py` 里
`Detector` 协议的 `detect(img, bg, frame_idx, debug)` 接口，
所以能直接注册进现有的 DETECTORS 表，和 `simple_threshold` 并列可选。
"""

# 显式导入，让每个文件里的 @register_xxx 装饰器执行、注册生效。
# 和 rain/detectors/__init__.py、rain/background/__init__.py 是同一套写法。
from . import (
    adaptive_threshold_segmenter,
    background_differencer,
    brightness_normalizer,
    connected_component_analyzer,
    contour_extractor,
    morphology_processor,
)

__all__ = [
    "brightness_normalizer",
    "background_differencer",
    "adaptive_threshold_segmenter",
    "morphology_processor",
    "connected_component_analyzer",
    "contour_extractor",
]
