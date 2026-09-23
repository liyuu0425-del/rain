"""Detector 实现：组合式六步流水线（亮度归一化→背景差分→自适应阈值→
形态学→连通域→轮廓提取）。

这个类本身**不包含任何检测算法**，只按固定顺序调用
`rain/detection/` 下的六个独立模块——这正是任务要求的
"Pipeline只负责调用顺序，不能包含具体算法实现"在检测器层面的体现。

默认参数（normalizer=none、segmenter=fixed 阈值）和
`SimpleThresholdDetector` 完全对齐，两者在相同参数下必须逐帧产出相同的
检测结果——见 tests/test_detection_stage2.py 里的等价性回归测试。
这样"拆分模块"这件事本身不会改变检测算法的行为，只是把实现拆开了。
"""

from __future__ import annotations

import numpy as np

from ..registry import (
    build_cc_analyzer,
    build_contour_extractor,
    build_differencer,
    build_morphology,
    build_normalizer,
    build_segmenter,
    register_detector,
)
from ..types import Detection

# 六个子步骤各自的默认配置：normalizer=none、differencer=bg_minus_img、
# segmenter=fixed，全部对齐 SimpleThresholdDetector 的默认行为。
_DEFAULT_NORMALIZER = {"name": "brightness_normalizer", "params": {"method": "none"}}
_DEFAULT_DIFFERENCER = {"name": "background_differencer", "params": {"mode": "bg_minus_img"}}
_DEFAULT_SEGMENTER = {"name": "adaptive_threshold_segmenter", "params": {"method": "fixed", "thresh": 20}}
_DEFAULT_MORPHOLOGY = {"name": "morphology_processor", "params": {"open_ksize": 3}}
_DEFAULT_CC_ANALYZER = {"name": "connected_component_analyzer", "params": {}}
_DEFAULT_CONTOUR_EXTRACTOR = {
    "name": "contour_extractor",
    "params": {"min_area": 10.0, "max_area": 5000.0, "max_aspect": 3.5},
}


@register_detector("modular_pipeline")
class ModularPipelineDetector:
    """六步模块化检测流水线。

    参数（每一个都是 `{"name": ..., "params": {...}}` 形式的嵌套配置，
    格式和 config.yaml 里顶层的 detector/tracker 段完全一样，只是嵌在
    这一个 detector 的 params 里）：
        normalizer, differencer, segmenter, morphology,
        cc_analyzer, contour_extractor

    不传的段落使用上面 `_DEFAULT_*` 里定义的默认值（=和原 Demo 一致的
    行为），所以最省事的用法是完全不传参数，等价于原 SimpleThresholdDetector
    默认参数下的行为。
    """

    def __init__(
        self,
        normalizer: dict | None = None,
        differencer: dict | None = None,
        segmenter: dict | None = None,
        morphology: dict | None = None,
        cc_analyzer: dict | None = None,
        contour_extractor: dict | None = None,
    ):
        self.normalizer = build_normalizer(normalizer or _DEFAULT_NORMALIZER)
        self.differencer = build_differencer(differencer or _DEFAULT_DIFFERENCER)
        self.segmenter = build_segmenter(segmenter or _DEFAULT_SEGMENTER)
        self.morphology = build_morphology(morphology or _DEFAULT_MORPHOLOGY)
        self.cc_analyzer = build_cc_analyzer(cc_analyzer or _DEFAULT_CC_ANALYZER)
        self.contour_extractor = build_contour_extractor(
            contour_extractor or _DEFAULT_CONTOUR_EXTRACTOR
        )

    def detect(
        self,
        img: np.ndarray,
        bg: np.ndarray,
        frame_idx: int = 0,
        debug: dict | None = None,
    ) -> list[Detection]:
        normalized = self.normalizer.normalize(img)
        diff = self.differencer.diff(normalized, bg)
        binary_raw = self.segmenter.segment(diff)
        binary = self.morphology.process(binary_raw)

        if debug is not None:
            debug["normalized"] = normalized
            debug["diff"] = diff
            debug["binary_raw"] = binary_raw
            debug["binary"] = binary

        contours = self.cc_analyzer.analyze(binary)
        return self.contour_extractor.extract(contours, img.shape, frame_idx)
