"""阶段2模块测试：六个检测子模块 + 组合式 Detector 与原 Demo 的等价性回归。

和 tests/test_contracts.py 的关系：那份文件对"注册表里所有 Detector 实现"
跑同一套黑盒契约（不关心内部怎么实现）；这份文件额外测两件事：
    1. 六个子模块各自的输入输出是否符合预期（白盒，构造小图直接验证数值）；
    2. `ModularPipelineDetector`（拆分后的组合流水线）在默认参数下，是否和
       `SimpleThresholdDetector`（原 Demo 的检测器）对同一张图产出完全相同
       的结果——这是"拆分模块不改变算法行为"的回归证据。
不依赖真实数据（不读 D:\\毕设\\数据），保证在任何机器上都能跑；
真实数据上的对比放在 pipeline/stage2_detection.py 里，跑出来的结果单独
在验收报告里报告。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  触发所有实现的注册
from rain.detection.adaptive_threshold_segmenter import AdaptiveThresholdSegmenter
from rain.detection.background_differencer import BackgroundDifferencer
from rain.detection.brightness_normalizer import BrightnessNormalizer
from rain.detection.connected_component_analyzer import ConnectedComponentAnalyzer
from rain.detection.contour_extractor import ContourExtractor
from rain.detection.morphology_processor import MorphologyProcessor
from rain.detectors.modular_pipeline import ModularPipelineDetector
from rain.detectors.simple_threshold import SimpleThresholdDetector

H, W = 200, 300


def make_background() -> np.ndarray:
    return np.full((H, W), 200, dtype=np.uint8)


def make_frame_with_drops(centers, radius=6, bg_value=200, drop_value=120) -> np.ndarray:
    img = np.full((H, W), bg_value, dtype=np.uint8)
    for (cx, cy) in centers:
        cv2.circle(img, (cx, cy), radius, drop_value, -1)
    return img


# ---------------------------------------------------------------------------
# BrightnessNormalizer
# ---------------------------------------------------------------------------


def test_normalizer_none_returns_same_array():
    img = make_frame_with_drops([(50, 50)])
    out = BrightnessNormalizer(method="none").normalize(img)
    assert np.array_equal(out, img)


def test_normalizer_mean_match_scales_to_target():
    img = np.full((10, 10), 100, dtype=np.uint8)
    normed = BrightnessNormalizer(method="mean_match", target_mean=150.0).normalize(img)
    assert abs(float(normed.mean()) - 150.0) < 1.0


def test_normalizer_mean_match_does_not_overflow():
    """100 * (250/100) = 250，在合理范围；但要确认极端比例下会截断到255而不回绕。"""
    img = np.full((10, 10), 200, dtype=np.uint8)
    normed = BrightnessNormalizer(method="mean_match", target_mean=400.0).normalize(img)
    assert normed.max() <= 255
    assert normed.dtype == np.uint8


def test_normalizer_invalid_method_raises():
    with pytest.raises(ValueError):
        BrightnessNormalizer(method="不存在的方法")


def test_normalizer_mean_match_requires_target_mean():
    with pytest.raises(ValueError):
        BrightnessNormalizer(method="mean_match")


def test_normalizer_does_not_modify_input():
    img = make_frame_with_drops([(50, 50)])
    img_copy = img.copy()
    BrightnessNormalizer(method="mean_match", target_mean=180.0).normalize(img)
    assert np.array_equal(img, img_copy)


# ---------------------------------------------------------------------------
# BackgroundDifferencer
# ---------------------------------------------------------------------------


def test_differencer_bg_minus_img_matches_cv2_subtract():
    bg = make_background()
    img = make_frame_with_drops([(50, 50)])
    out = BackgroundDifferencer(mode="bg_minus_img").diff(img, bg)
    expected = cv2.subtract(bg, img)
    assert np.array_equal(out, expected)


def test_differencer_bg_minus_img_is_saturated_not_negative():
    """雨滴比背景暗，diff 只应在雨滴处为正；饱和减法不会出现"负数回绕"。"""
    bg = np.full((10, 10), 50, dtype=np.uint8)   # 背景比"雨滴"还暗的极端情况
    img = np.full((10, 10), 200, dtype=np.uint8)
    out = BackgroundDifferencer(mode="bg_minus_img").diff(img, bg)
    assert out.min() == 0  # 50-200 应截断到 0，不是回绕成一个大正数


def test_differencer_abs_mode_matches_absdiff():
    bg = make_background()
    img = make_frame_with_drops([(50, 50)])
    out = BackgroundDifferencer(mode="abs").diff(img, bg)
    expected = cv2.absdiff(bg, img)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# AdaptiveThresholdSegmenter
# ---------------------------------------------------------------------------


def test_segmenter_fixed_matches_cv2_threshold():
    diff = np.array([[0, 10, 30], [25, 5, 100]], dtype=np.uint8)
    out = AdaptiveThresholdSegmenter(method="fixed", thresh=20).segment(diff)
    _, expected = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    assert np.array_equal(out, expected)


def test_segmenter_adaptive_mean_runs_and_returns_binary():
    diff = make_frame_with_drops([(50, 50), (150, 100)], drop_value=80)
    out = AdaptiveThresholdSegmenter(method="adaptive_mean", block_size=31, c=-5).segment(diff)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_segmenter_adaptive_gaussian_runs_and_returns_binary():
    diff = make_frame_with_drops([(50, 50), (150, 100)], drop_value=80)
    out = AdaptiveThresholdSegmenter(method="adaptive_gaussian", block_size=31, c=-5).segment(diff)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_segmenter_rejects_even_block_size():
    with pytest.raises(ValueError):
        AdaptiveThresholdSegmenter(method="adaptive_mean", block_size=30)


def test_segmenter_rejects_unknown_method():
    with pytest.raises(ValueError):
        AdaptiveThresholdSegmenter(method="不存在的方法")


# ---------------------------------------------------------------------------
# MorphologyProcessor
# ---------------------------------------------------------------------------


def test_morphology_matches_cv2_morphologyex():
    binary = make_frame_with_drops([(50, 50)], drop_value=255, bg_value=0)
    proc = MorphologyProcessor(open_ksize=3)
    out = proc.process(binary)
    expected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, proc.kernel)
    assert np.array_equal(out, expected)


def test_morphology_removes_small_noise():
    """开运算应该去掉孤立的单像素噪点，保留大目标。"""
    binary = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(binary, (25, 25), 8, 255, -1)  # 大目标
    binary[5, 5] = 255  # 孤立单像素噪点

    out = MorphologyProcessor(open_ksize=3).process(binary)

    assert out[5, 5] == 0          # 噪点被去掉
    assert out[25, 25] == 255      # 大目标保留


# ---------------------------------------------------------------------------
# ConnectedComponentAnalyzer
# ---------------------------------------------------------------------------


def test_cc_analyzer_matches_cv2_findcontours():
    binary = make_frame_with_drops([(50, 50), (150, 100)], drop_value=255, bg_value=0)
    out = ConnectedComponentAnalyzer().analyze(binary)
    expected, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(out) == len(expected)
    for a, b in zip(out, expected):
        assert np.array_equal(a, b)


def test_cc_analyzer_counts_separate_components():
    binary = make_frame_with_drops([(30, 30), (150, 100), (250, 150)], radius=5,
                                    drop_value=255, bg_value=0)
    out = ConnectedComponentAnalyzer().analyze(binary)
    assert len(out) == 3


def test_cc_analyzer_empty_image_returns_empty_list():
    binary = np.zeros((50, 50), dtype=np.uint8)
    out = ConnectedComponentAnalyzer().analyze(binary)
    assert out == []


# ---------------------------------------------------------------------------
# ContourExtractor
# ---------------------------------------------------------------------------


def test_contour_extractor_basic_fields():
    binary = make_frame_with_drops([(50, 50)], radius=6, drop_value=255, bg_value=0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = ContourExtractor(min_area=10, max_area=5000, max_aspect=3.5).extract(
        contours, (H, W), frame_idx=7
    )

    assert len(dets) == 1
    d = dets[0]
    assert d.frame_idx == 7
    assert abs(d.cx - 50) < 1.0 and abs(d.cy - 50) < 1.0
    assert d.d_eq_px == pytest.approx(2.0 * np.sqrt(d.area_px / np.pi), abs=1e-3)
    assert d.contour is not None
    assert d.perimeter_px is not None


def test_contour_extractor_drops_below_min_area():
    binary = make_frame_with_drops([(50, 50)], radius=1, drop_value=255, bg_value=0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = ContourExtractor(min_area=1000, max_area=5000, max_aspect=3.5).extract(
        contours, (H, W), frame_idx=0
    )
    assert dets == []


def test_contour_extractor_flags_touches_border():
    binary = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(binary, (0, 25), 8, 255, -1)  # 圆心在左边界上
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = ContourExtractor(min_area=1, max_area=5000, max_aspect=10).extract(
        contours, (50, 50), frame_idx=0
    )
    assert dets and dets[0].qc_flags & 1  # QC_TOUCHES_BORDER = 1 << 0


# ---------------------------------------------------------------------------
# 等价性回归测试：ModularPipelineDetector 默认参数 vs SimpleThresholdDetector
# ---------------------------------------------------------------------------


def _assert_detections_equal(dets_a, dets_b):
    assert len(dets_a) == len(dets_b), f"数量不同: {len(dets_a)} vs {len(dets_b)}"
    for da, db in zip(dets_a, dets_b):
        assert da.frame_idx == db.frame_idx
        assert abs(da.cx - db.cx) < 1e-6
        assert abs(da.cy - db.cy) < 1e-6
        assert abs(da.area_px - db.area_px) < 1e-6
        assert da.bbox == db.bbox
        assert abs(da.d_eq_px - db.d_eq_px) < 1e-6
        assert da.qc_flags == db.qc_flags
        assert abs((da.perimeter_px or 0) - (db.perimeter_px or 0)) < 1e-6
        maj_a = da.major_axis_px if da.major_axis_px is not None else -1
        maj_b = db.major_axis_px if db.major_axis_px is not None else -1
        assert abs(maj_a - maj_b) < 1e-3


def test_modular_pipeline_matches_simple_threshold_on_synthetic_image():
    bg = make_background()
    img = make_frame_with_drops([(50, 50), (150, 100), (220, 60), (30, 170)])

    simple = SimpleThresholdDetector(thresh=20, open_ksize=3, min_area=10, max_area=5000, max_aspect=3.5)
    modular = ModularPipelineDetector()  # 全部用默认值，理论上等价于上面这行

    dets_simple = simple.detect(img, bg, frame_idx=3)
    dets_modular = modular.detect(img, bg, frame_idx=3)

    assert len(dets_simple) > 0, "合成图应该能检出目标，测试用例本身有问题"
    _assert_detections_equal(dets_simple, dets_modular)


def test_modular_pipeline_matches_simple_threshold_on_empty_frame():
    bg = make_background()
    img = make_background()  # 无目标

    simple = SimpleThresholdDetector()
    modular = ModularPipelineDetector()

    assert simple.detect(img, bg) == modular.detect(img, bg) == []


def test_modular_pipeline_matches_simple_threshold_with_custom_params():
    bg = make_background()
    img = make_frame_with_drops([(80, 80), (200, 120)], radius=4)

    custom = dict(thresh=15, open_ksize=5, min_area=5, max_area=3000, max_aspect=2.0)
    simple = SimpleThresholdDetector(**custom)
    modular = ModularPipelineDetector(
        segmenter={"name": "adaptive_threshold_segmenter", "params": {"method": "fixed", "thresh": 15}},
        morphology={"name": "morphology_processor", "params": {"open_ksize": 5}},
        contour_extractor={
            "name": "contour_extractor",
            "params": {"min_area": 5, "max_area": 3000, "max_aspect": 2.0},
        },
    )

    _assert_detections_equal(simple.detect(img, bg), modular.detect(img, bg))


def test_modular_pipeline_debug_dict_has_all_intermediate_stages():
    bg = make_background()
    img = make_frame_with_drops([(50, 50)])

    dbg: dict = {}
    ModularPipelineDetector().detect(img, bg, debug=dbg)

    for key in ("normalized", "diff", "binary_raw", "binary"):
        assert key in dbg, f"debug dict 缺少中间产物: {key}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
