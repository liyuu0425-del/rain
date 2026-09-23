"""阶段7B性能优化基础设施测试：TiffInspector / FastTifReader / BoundedPipelineRunner。

用 PIL 现场生成合成 TIFF（LZW压缩 / 不压缩两种）做测试夹具，不依赖
D:\\毕设\\数据 真实文件——和项目里其它契约测试一致的原则：功能测试要能
在任何机器上跑。真实数据上的验证（确认这批数据确实是LZW、性能优化
效果如何）放在 pipeline/stage7b_optimization.py 的运行报告里。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import rain  # noqa: F401  触发所有实现的注册
from rain.perf.bounded_pipeline import BoundedPipelineRunner
from rain.perf.tiff_inspector import inspect_tiff
from rain.readers.fast_tif import FastTifReader
from rain.readers.lzw_tif import LzwTifReader


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


def _make_tiff(tmp_path: Path, name: str, compression: str, arr: np.ndarray | None = None) -> Path:
    if arr is None:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(20, 30), dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(str(p), compression=compression)
    return p


# ---------------------------------------------------------------------------
# TiffInspector —— "先检查压缩方式，不能假设"
# ---------------------------------------------------------------------------


def test_tiff_inspector_detects_lzw_compression(tmp_path):
    p = _make_tiff(tmp_path, "lzw.tif", "tiff_lzw")
    info = inspect_tiff(str(p))
    assert info.compression_id == 5
    assert info.compression_name == "LZW"


def test_tiff_inspector_detects_uncompressed(tmp_path):
    p = _make_tiff(tmp_path, "none.tif", "raw")
    info = inspect_tiff(str(p))
    assert info.compression_id == 1
    assert info.compression_name == "none (uncompressed)"


def test_tiff_inspector_reports_dimensions_and_ratio(tmp_path):
    arr = np.zeros((20, 30), dtype=np.uint8)
    p = _make_tiff(tmp_path, "dims.tif", "tiff_lzw", arr=arr)
    info = inspect_tiff(str(p))
    assert info.width == 30 and info.height == 20
    assert info.raw_uncompressed_bytes == 30 * 20
    assert info.compression_ratio is not None


def test_tiff_inspector_rejects_non_tiff_file(tmp_path):
    p = tmp_path / "not_a_tiff.tif"
    p.write_bytes(b"this is definitely not a tiff file")
    with pytest.raises(ValueError):
        inspect_tiff(str(p))


def test_tiff_inspector_rejects_truncated_file(tmp_path):
    p = tmp_path / "truncated.tif"
    p.write_bytes(b"II")
    with pytest.raises(ValueError):
        inspect_tiff(str(p))


# ---------------------------------------------------------------------------
# FastTifReader —— 和原 LzwTifReader(cv2) 像素级一致
# ---------------------------------------------------------------------------


def test_fast_tif_reader_pixel_identical_to_lzw_tif_reader(tmp_path):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(50, 60), dtype=np.uint8)
    p = _make_tiff(tmp_path, "cmp.tif", "tiff_lzw", arr=arr)

    img_fast = FastTifReader().read(str(p))
    img_cv2 = LzwTifReader().read(str(p))

    assert img_fast.dtype == img_cv2.dtype == np.uint8
    assert img_fast.shape == img_cv2.shape
    assert np.array_equal(img_fast, img_cv2)
    assert np.array_equal(img_fast, arr)  # 和写入前的原始数组也完全一致


def test_fast_tif_reader_validates_expected_shape(tmp_path):
    p = _make_tiff(tmp_path, "shape.tif", "tiff_lzw", arr=np.zeros((20, 30), dtype=np.uint8))
    reader = FastTifReader(expect_shape=(999, 999))
    with pytest.raises(ValueError):
        reader.read(str(p))


def test_fast_tif_reader_missing_file_raises():
    with pytest.raises(Exception):
        FastTifReader().read("这个文件不存在.tif")


def test_fast_tif_reader_registered_in_registry():
    from rain.registry import READERS

    assert "fast_tif" in READERS


# ---------------------------------------------------------------------------
# BoundedPipelineRunner —— 有界队列 + 多线程读取/处理 + 单写出线程
# ---------------------------------------------------------------------------


def test_bounded_pipeline_processes_all_items_exactly_once():
    items = list(range(50))
    results = []
    lock = __import__("threading").Lock()

    def read_fn(x):
        return x * 2

    def process_fn(x):
        return x + 1

    def write_fn(x):
        with lock:
            results.append(x)

    runner = BoundedPipelineRunner(
        items=items, read_fn=read_fn, process_fn=process_fn, write_fn=write_fn,
        n_reader_threads=4, n_processor_threads=2, queue_maxsize=4,
    )
    result = runner.run()

    assert result.n_items == 50
    assert result.n_errors == 0
    assert sorted(results) == sorted(x * 2 + 1 for x in items)  # 每个都处理了、恰好一次


def test_bounded_pipeline_queue_never_exceeds_maxsize():
    items = list(range(100))

    def read_fn(x):
        time.sleep(0.001)  # 让读取比写出慢一点，制造队列真正有内容堆积的机会
        return x

    results = []

    def write_fn(x):
        results.append(x)

    runner = BoundedPipelineRunner(
        items=items, read_fn=read_fn, write_fn=write_fn,
        n_reader_threads=4, n_processor_threads=2, queue_maxsize=5,
    )
    result = runner.run()

    assert result.read_queue_stats.max_size_observed <= 5
    assert result.write_queue_stats.max_size_observed <= 5
    assert len(results) == 100


def test_bounded_pipeline_process_fn_none_is_passthrough():
    items = [1, 2, 3, 4, 5]
    results = []

    runner = BoundedPipelineRunner(
        items=items, read_fn=lambda x: x * 10, process_fn=None,
        write_fn=lambda x: results.append(x), n_reader_threads=2, queue_maxsize=2,
    )
    runner.run()

    assert sorted(results) == [10, 20, 30, 40, 50]


def test_bounded_pipeline_propagates_read_errors():
    def read_fn(x):
        if x == 3:
            raise ValueError("模拟读取失败")
        return x

    runner = BoundedPipelineRunner(
        items=[1, 2, 3, 4, 5], read_fn=read_fn, write_fn=lambda x: None, queue_maxsize=2,
    )
    with pytest.raises(ValueError):
        runner.run()


def test_bounded_pipeline_propagates_process_errors():
    def process_fn(x):
        if x == 2:
            raise RuntimeError("模拟处理失败")
        return x

    runner = BoundedPipelineRunner(
        items=[1, 2, 3], read_fn=lambda x: x, process_fn=process_fn,
        write_fn=lambda x: None, queue_maxsize=2,
    )
    with pytest.raises(RuntimeError):
        runner.run()


def test_bounded_pipeline_rejects_zero_maxsize():
    with pytest.raises(ValueError):
        BoundedPipelineRunner(items=[1], read_fn=lambda x: x, write_fn=lambda x: None, queue_maxsize=0)


def test_bounded_pipeline_empty_items():
    results = []
    runner = BoundedPipelineRunner(
        items=[], read_fn=lambda x: x, write_fn=lambda x: results.append(x), queue_maxsize=2,
    )
    result = runner.run()
    assert result.n_items == 0
    assert results == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
