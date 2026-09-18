"""契约测试：保证任何实现都能安全替换。

## 为什么需要这个文件

"换算法不动主程序"不能靠口头约定。只写一个实现的时候，
任何接口看起来都是对的；等到加第三、第四个实现时才发现契约被破坏，
返工代价就很大了。

这里的测试对**注册表里的每一个实现**跑同一组检查。
新加的算法自动被覆盖，不需要额外写测试。

## 运行方式

    cd D:\\Codex\\毕设相关\\rain
    python -m pytest tests/ -v

没装 pytest 的话直接跑也行（脚本末尾有兜底的 main）：

    python tests/test_contracts.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  —— import 它是为了触发所有实现的注册
from rain.registry import BACKGROUNDS, DETECTORS, READERS, TRACKERS
from rain.types import Detection, Track


# ---------------------------------------------------------------------------
# 测试夹具：造一张合成图，不依赖真实数据
#
# 用合成图而不是真实雨滴图，是因为契约测试要能在任何机器上跑，
# 不能依赖 D:\毕设\数据 存在。真实数据的算法精度评估是另一回事，
# 属于人工标注验证集的工作。
# ---------------------------------------------------------------------------

H, W = 200, 300


def make_background() -> np.ndarray:
    """均匀亮背景，模拟背光板。"""
    return np.full((H, W), 200, dtype=np.uint8)


def make_frame_with_drops(centers, radius=6) -> np.ndarray:
    """在亮背景上画几个暗圆，模拟被雨滴挡住的背光。"""
    import cv2

    img = make_background()
    for (cx, cy) in centers:
        cv2.circle(img, (cx, cy), radius, 120, -1)
    return img


def make_empty_frame() -> np.ndarray:
    """没有雨滴的帧。"""
    return make_background()


# ---------------------------------------------------------------------------
# Detector 契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_returns_list_of_detection(name):
    """必须返回 list[Detection]，不是 tuple、不是 ndarray、不是 dict。"""
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50), (150, 100)])

    out = det.detect(img, bg, frame_idx=7)

    assert isinstance(out, list), f"{name} 返回了 {type(out).__name__}，应为 list"
    for d in out:
        assert isinstance(d, Detection), f"{name} 返回的元素是 {type(d).__name__}，应为 Detection"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_empty_frame_returns_empty_list(name):
    """空图必须返回 []，不能返回 None —— 否则下游 for 循环直接崩。"""
    det = DETECTORS[name]()
    bg = make_background()

    out = det.detect(make_empty_frame(), bg, frame_idx=0)

    assert out is not None, f"{name} 空图返回了 None，应为 []"
    assert isinstance(out, list)
    assert len(out) == 0, f"{name} 在无目标的图上检出了 {len(out)} 个目标（误检）"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_propagates_frame_idx(name):
    """frame_idx 必须原样写进每个 Detection，实现不许自己编号。"""
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50)])

    out = det.detect(img, bg, frame_idx=42)

    for d in out:
        assert d.frame_idx == 42, f"{name} 把 frame_idx 改成了 {d.frame_idx}，应为 42"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_d_eq_matches_area(name):
    """等效直径必须等于 2*sqrt(A/pi)，各实现不许自己另算一套。"""
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50), (150, 100), (250, 150)])

    out = det.detect(img, bg, frame_idx=0)
    assert out, f"{name} 在有 3 个明显目标的图上没检出任何东西"

    for d in out:
        expected = 2.0 * math.sqrt(d.area_px / math.pi)
        assert abs(d.d_eq_px - expected) < 1e-3, (
            f"{name} 的 d_eq_px={d.d_eq_px:.4f}，"
            f"但按 area_px={d.area_px} 算应为 {expected:.4f}"
        )


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_coords_inside_image(name):
    """坐标必须在图像范围内（如果实现内部裁了 ROI，要把偏移加回去）。"""
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50), (150, 100)])

    out = det.detect(img, bg, frame_idx=0)

    for d in out:
        assert 0 <= d.cx <= W, f"{name} 的 cx={d.cx} 超出图宽 {W}"
        assert 0 <= d.cy <= H, f"{name} 的 cy={d.cy} 超出图高 {H}"
        x, y, w, h = d.bbox
        assert x >= 0 and y >= 0, f"{name} 的 bbox 起点为负: {d.bbox}"
        assert x + w <= W and y + h <= H, f"{name} 的 bbox 超出图像: {d.bbox}"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_does_not_modify_inputs(name):
    """不得修改传入的 img 和 bg —— 其他线程可能在共享它们。

    这一条直接关系到并行安全。如果某个实现图省事就地改了输入数组，
    单线程跑没问题，并行跑就会随机出错。
    """
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50), (150, 100)])

    bg_copy = bg.copy()
    img_copy = img.copy()

    det.detect(img, bg, frame_idx=0)

    assert np.array_equal(img, img_copy), f"{name} 修改了传入的 img"
    assert np.array_equal(bg, bg_copy), f"{name} 修改了传入的 bg"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_debug_none_is_default(name):
    """debug=None 时不产出中间图；传 dict 时才填。

    中间图不能进返回值 —— 否则每个实现都被迫产出，
    而且跨线程传 3 MB 数组会让内存和队列都爆掉。
    """
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50)])

    # 不传 debug，应该正常工作
    out1 = det.detect(img, bg, frame_idx=0)
    assert isinstance(out1, list)

    # 传 dict，应该被填入中间结果
    dbg: dict = {}
    out2 = det.detect(img, bg, frame_idx=0, debug=dbg)
    assert isinstance(out2, list)
    assert len(dbg) > 0, f"{name} 收到 debug dict 但没往里放任何中间图"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_is_deterministic(name):
    """同样输入必须给出同样输出 —— 否则并行与单线程结果对不上。"""
    det = DETECTORS[name]()
    bg = make_background()
    img = make_frame_with_drops([(50, 50), (150, 100), (250, 150)])

    a = det.detect(img, bg, frame_idx=0)
    b = det.detect(img, bg, frame_idx=0)

    assert len(a) == len(b), f"{name} 两次调用检出数量不同: {len(a)} vs {len(b)}"
    for d1, d2 in zip(a, b):
        assert abs(d1.cx - d2.cx) < 1e-9 and abs(d1.cy - d2.cy) < 1e-9, (
            f"{name} 两次调用坐标不同"
        )


# ---------------------------------------------------------------------------
# BackgroundModel 契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BACKGROUNDS))
def test_background_returns_uint8_same_shape(name):
    """背景图必须是 uint8、与输入帧同尺寸。

    external_ref 需要构造参数，这里跳过 —— 它的行为由集成测试覆盖。
    """
    cls = BACKGROUNDS[name]
    try:
        bg_model = cls()
    except TypeError:
        pytest.skip(f"{name} 需要构造参数，跳过无参契约测试")

    frames = [make_frame_with_drops([(i * 10 + 20, 50)]) for i in range(10)]
    bg = bg_model.build(frames)

    assert isinstance(bg, np.ndarray), f"{name} 返回了 {type(bg).__name__}"
    assert bg.dtype == np.uint8, f"{name} 返回 dtype={bg.dtype}，应为 uint8"
    assert bg.shape == frames[0].shape, f"{name} 返回 shape={bg.shape}，应为 {frames[0].shape}"


@pytest.mark.parametrize("name", sorted(BACKGROUNDS))
def test_background_accepts_generator(name):
    """frames 可能是生成器，实现里只允许遍历一次。"""
    cls = BACKGROUNDS[name]
    try:
        bg_model = cls()
    except TypeError:
        pytest.skip(f"{name} 需要构造参数，跳过")

    gen = (make_frame_with_drops([(i * 10 + 20, 50)]) for i in range(10))
    bg = bg_model.build(gen)

    assert bg.shape == (H, W)


# ---------------------------------------------------------------------------
# Tracker 契约
# ---------------------------------------------------------------------------


def _fake_falling_track(n_frames=8, dy=80, dx=2, x0=100, y0=20):
    """造一条匀速下落的合成轨迹（逐帧检测列表）。"""
    per_frame = []
    for i in range(n_frames):
        d = Detection(
            frame_idx=i,
            cx=x0 + dx * i,
            cy=y0 + dy * i,
            area_px=100.0,
            bbox=(int(x0 + dx * i) - 5, int(y0 + dy * i) - 5, 10, 10),
            d_eq_px=Detection.d_eq_from_area(100.0),
        )
        per_frame.append([d])
    return per_frame


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_tracker_returns_list_of_track(name):
    tk = TRACKERS[name]()
    out = tk.track(_fake_falling_track(), cam="a")

    assert isinstance(out, list), f"{name} 返回了 {type(out).__name__}"
    for t in out:
        assert isinstance(t, Track), f"{name} 返回的元素是 {type(t).__name__}"
        assert t.cam == "a", f"{name} 没有把 cam 写进 Track"


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_tracker_empty_input(name):
    """全空的逐帧列表必须返回 []，不能崩。"""
    tk = TRACKERS[name]()
    assert tk.track([], cam="a") == []
    assert tk.track([[], [], []], cam="a") == []


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_tracker_links_a_clean_falling_drop(name):
    """一条干净的匀速下落轨迹必须被连成一条，不能碎成多条。

    这是跟踪器最基本的能力。连不上说明门控参数和实际位移不匹配 ——
    本项目实测踩过这个坑：圆形门控 40px 只连上了 15% 的检测。
    """
    tk = TRACKERS[name]()
    per_frame = _fake_falling_track(n_frames=8, dy=80, dx=2)

    out = tk.track(per_frame, cam="a")

    assert len(out) == 1, (
        f"{name} 把一条 8 帧的匀速下落轨迹连成了 {len(out)} 条。"
        f"检查门控参数是否与 dy=80 px/帧 匹配。"
    )
    assert out[0].n_frames == 8, f"{name} 只连上了 {out[0].n_frames}/8 帧"


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_tracker_is_deterministic(name):
    """必须是纯函数，包括 track_id 的编号顺序。

    这一条直接关系到 run_offline 的 --workers 1 vs --workers 8 回归测试。
    """
    tk = TRACKERS[name]()
    per_frame = _fake_falling_track(n_frames=6)

    a = tk.track(per_frame, cam="a")
    # 注意要重新造数据：track() 会往 Track.detections 里 append，
    # 如果复用同一份 per_frame，第二次的输入已经不同了。
    per_frame2 = _fake_falling_track(n_frames=6)
    b = tk.track(per_frame2, cam="a")

    assert [t.track_id for t in a] == [t.track_id for t in b], f"{name} 的 track_id 不稳定"
    assert [t.n_frames for t in a] == [t.n_frames for t in b], f"{name} 的轨迹长度不稳定"


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_tracker_outputs_sorted_by_id(name):
    """输出必须按 track_id 排序，保证结果顺序确定。"""
    tk = TRACKERS[name]()
    per_frame = _fake_falling_track(n_frames=8)
    # 再加一条轨迹
    for i, dets in enumerate(per_frame):
        dets.append(
            Detection(
                frame_idx=i,
                cx=250 - 1 * i,
                cy=15 + 75 * i,
                area_px=80.0,
                bbox=(240, 10, 10, 10),
                d_eq_px=Detection.d_eq_from_area(80.0),
            )
        )

    out = tk.track(per_frame, cam="a")
    ids = [t.track_id for t in out]
    assert ids == sorted(ids), f"{name} 的输出没有按 track_id 排序: {ids}"


# ---------------------------------------------------------------------------
# Reader 契约（只测不需要真实文件的部分）
# ---------------------------------------------------------------------------


def test_all_registries_are_non_empty():
    """确认实现模块真的被 import 了、装饰器真的执行了。

    如果哪张表是空的，说明 rain/__init__.py 里漏了 import，
    那么 config.yaml 里写的名字会查不到。
    """
    assert READERS, "READERS 注册表为空 —— 检查 rain/__init__.py 的 import"
    assert BACKGROUNDS, "BACKGROUNDS 注册表为空"
    assert DETECTORS, "DETECTORS 注册表为空"
    assert TRACKERS, "TRACKERS 注册表为空"


def test_registry_rejects_duplicate_names():
    """重名注册必须报错，而不是静默覆盖。"""
    from rain.registry import register_detector

    with pytest.raises(ValueError, match="重复注册"):

        @register_detector(sorted(DETECTORS)[0])
        class _Dup:
            def detect(self, img, bg, frame_idx=0, debug=None):
                return []


def test_build_unknown_name_gives_helpful_error():
    """配置里写了不存在的名字，报错信息要列出可用选项。"""
    from rain.registry import build_detector

    with pytest.raises(KeyError) as exc:
        build_detector({"name": "这个不存在"})
    msg = str(exc.value)
    assert "当前已注册" in msg, f"报错信息没列出可用选项: {msg}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
