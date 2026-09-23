"""阶段1模块测试：MetadataReader / TimingReader / TimeQualityChecker / FramePairer。

和 tests/test_contracts.py 的定位不同：那份测试面向"注册表里所有实现的
通用契约"（换算法也要过），这份测试面向阶段1这四个具体模块各自的行为，
用构造出来的小数据（不依赖真实的 D:\\毕设\\数据），保证在任何机器上都能跑。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rain.data.frame_pairer import FramePairer
from rain.data.metadata_reader import MetadataReader
from rain.data.time_quality_checker import TimeQualityChecker
from rain.data.timing_reader import TimingReader
from rain.data.types import CaptureMetadata, TimingRecord

# ---------------------------------------------------------------------------
# 测试夹具：构造最小可用的 capture_metadata.json / capture_timing.csv，
# 不依赖真实数据目录。
# ---------------------------------------------------------------------------

_METADATA_TEMPLATE = {
    "capture_name": "Capture_TEST",
    "camera_a_sn": "SN_A",
    "camera_b_sn": "SN_B",
    "requested_frame_pairs": 5,
    "requested_width": 2064,
    "requested_height": 1544,
    "requested_exposure_us": 175.0,
    "requested_analog_gain": 15,
    "a_trigger_mode": "software_trigger_burst",
    "b_trigger_mode": "hardware_edge_trigger",
    "captured_complete_frame_pairs": 5,
    "capture_complete": True,
    "missing_frame_pairs": 0,
    "valid_frame_pairs": 5,
    "median_frame_period_a_us": 3200.0,
    "median_frame_period_b_us": 3200.0,
    "max_frame_period_a_us": 3300.0,
    "max_frame_period_b_us": 3300.0,
    "frame_gap_threshold_a_us": 4800.0,
    "frame_gap_threshold_b_us": 4800.0,
    "median_relative_drift_us": 0.0,
    "p95_abs_relative_drift_us": 100.0,
    "max_abs_relative_drift_us": 100.0,
    "timing_model": "A软件启动；A每帧STROBE硬件触发B",
    "alignment_note": "A/B设备时钟零点独立",
}

_TIMING_COLUMNS = [
    "pair_index", "pair_filename", "a_device_relative_us", "b_device_relative_us",
    "b_minus_a_relative_drift_us", "a_exposure_us", "b_exposure_us",
    "a_analog_gain", "b_analog_gain",
    "a_host_receive_wall_ns", "b_host_receive_wall_ns",
    "a_is_trigger_frame", "b_is_trigger_frame",
]


def _write_metadata(path: Path, overrides: dict | None = None) -> None:
    data = dict(_METADATA_TEMPLATE)
    if overrides:
        data.update(overrides)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_timing_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_TIMING_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _make_clean_rows(n=5, period_us=3200.0, filename_prefix="frame") -> list[dict]:
    """造 n 帧完全正常、匀速的时间记录（无倒退、无重复、无漂移异常）。"""
    rows = []
    for i in range(n):
        t = i * period_us
        rows.append({
            "pair_index": i,
            "pair_filename": f"{filename_prefix}_{i:03d}.tif",
            "a_device_relative_us": t,
            "b_device_relative_us": t,
            "b_minus_a_relative_drift_us": 0.0,
            "a_exposure_us": 175, "b_exposure_us": 175,
            "a_analog_gain": 3.75, "b_analog_gain": 3.75,
            "a_host_receive_wall_ns": 1000 + i, "b_host_receive_wall_ns": 1000 + i,
            "a_is_trigger_frame": 1, "b_is_trigger_frame": 2,
        })
    return rows


# ---------------------------------------------------------------------------
# MetadataReader
# ---------------------------------------------------------------------------


def test_metadata_reader_reads_all_fields(tmp_path):
    p = tmp_path / "capture_metadata.json"
    _write_metadata(p)

    meta = MetadataReader().read(str(p))

    assert meta.capture_name == "Capture_TEST"
    assert meta.camera_a_sn == "SN_A"
    assert meta.requested_frame_pairs == 5
    assert meta.requested_exposure_us == 175.0
    assert meta.capture_complete is True
    assert meta.raw["capture_name"] == "Capture_TEST"  # 原始 JSON 全文也保留着


def test_metadata_reader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        MetadataReader().read("这个文件不存在.json")


def test_metadata_reader_compare_detects_mismatch(tmp_path):
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    _write_metadata(pa)
    _write_metadata(pb, overrides={"requested_frame_pairs": 6})  # 故意造一个不一致

    reader = MetadataReader()
    meta_a = reader.read(str(pa))
    meta_b = reader.read(str(pb))

    mismatches = reader.compare(meta_a, meta_b)
    assert any("requested_frame_pairs" in m for m in mismatches)


def test_metadata_reader_compare_identical_returns_empty(tmp_path):
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    _write_metadata(pa)
    _write_metadata(pb)

    reader = MetadataReader()
    mismatches = reader.compare(reader.read(str(pa)), reader.read(str(pb)))
    assert mismatches == []


# ---------------------------------------------------------------------------
# TimingReader
# ---------------------------------------------------------------------------


def test_timing_reader_reads_rows_sorted_by_pair_index(tmp_path):
    p = tmp_path / "capture_timing.csv"
    rows = _make_clean_rows(5)
    rows_shuffled = [rows[3], rows[0], rows[4], rows[1], rows[2]]  # 故意打乱行序
    _write_timing_csv(p, rows_shuffled)

    records = TimingReader().read(str(p))

    assert [r.pair_index for r in records] == [0, 1, 2, 3, 4]
    assert records[1].a_device_relative_us == 3200.0
    assert isinstance(records[0], TimingRecord)


def test_timing_reader_types_are_numeric(tmp_path):
    p = tmp_path / "capture_timing.csv"
    _write_timing_csv(p, _make_clean_rows(3))

    records = TimingReader().read(str(p))

    rec = records[0]
    assert isinstance(rec.pair_index, int)
    assert isinstance(rec.a_device_relative_us, float)
    assert isinstance(rec.a_host_receive_wall_ns, int)


# ---------------------------------------------------------------------------
# TimeQualityChecker
# ---------------------------------------------------------------------------


def test_quality_checker_clean_data_has_no_flags(tmp_path):
    p = tmp_path / "capture_timing.csv"
    _write_timing_csv(p, _make_clean_rows(10, period_us=3200.0))
    records = TimingReader().read(str(p))

    report = TimeQualityChecker().check(records)

    assert report.n_records == 10
    assert report.n_time_backwards == 0
    assert report.n_duplicate_time == 0
    assert report.n_interval_outliers == 0
    assert report.n_drift_outliers == 0
    assert report.n_missing_pair_index == 0
    assert abs(report.measured_frame_period_median_us - 3200.0) < 1e-6


def test_quality_checker_detects_time_backwards():
    rows = _make_clean_rows(5, period_us=3200.0)
    rows[3]["a_device_relative_us"] = rows[2]["a_device_relative_us"] - 100.0  # 人为造一次倒退
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    report = TimeQualityChecker().check(records)

    assert report.n_time_backwards >= 1
    flagged = [p for p in report.per_pair if p.is_time_backwards_a]
    assert flagged and flagged[0].pair_index == 3


def test_quality_checker_detects_duplicate_time():
    rows = _make_clean_rows(5, period_us=3200.0)
    rows[2]["a_device_relative_us"] = rows[1]["a_device_relative_us"]  # 造一次重复时间戳
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    report = TimeQualityChecker(duplicate_time_tolerance_us=0.0).check(records)

    assert report.n_duplicate_time >= 1


def test_quality_checker_detects_interval_outlier():
    rows = _make_clean_rows(5, period_us=3200.0)
    # 第 4 帧的时间戳突然跳变一大截，制造一次异常间隔
    rows[4]["a_device_relative_us"] += 50000.0
    rows[4]["b_device_relative_us"] += 50000.0
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    # 不传 metadata，走"本批中位数 * 倍数"的兜底阈值
    report = TimeQualityChecker(interval_outlier_multiplier=1.5).check(records, metadata=None)

    assert report.n_interval_outliers >= 1


def test_quality_checker_detects_drift_outlier():
    rows = _make_clean_rows(5, period_us=3200.0)
    rows[2]["b_minus_a_relative_drift_us"] = 5000.0  # 远超正常漂移范围
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    report = TimeQualityChecker().check(records)

    assert report.n_drift_outliers >= 1


def test_quality_checker_detects_missing_pair_index():
    rows = _make_clean_rows(6, period_us=3200.0)
    del rows[3]  # 删掉 pair_index=3，制造一次缺帧
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    report = TimeQualityChecker().check(records)

    assert report.missing_pair_indices == [3]


def test_quality_checker_uses_metadata_threshold(tmp_path):
    """传入 metadata 时，间隔阈值应使用 metadata 里的实测值而不是兜底倍数。"""
    meta_path = tmp_path / "capture_metadata.json"
    _write_metadata(meta_path, overrides={"frame_gap_threshold_a_us": 1000.0})
    metadata = MetadataReader().read(str(meta_path))

    rows = _make_clean_rows(5, period_us=3200.0)
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    # 正常帧间隔是 3200us，阈值给到很小的 1000us，所有帧间隔都应被判定异常。
    report = TimeQualityChecker().check(records, metadata=metadata)
    assert report.n_interval_outliers == 4  # 5 帧有 4 个间隔


def _stringify(row: dict) -> dict:
    """csv.DictReader 读出来的值全是字符串，TimingReader._row_to_record 期望
    收到这种"全字符串"的字典；这个小工具把测试里手写的 dict（数值是 int/float）
    转成同样的字符串形式，避免测试绕过真实的类型转换代码路径。"""
    return {k: str(v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# FramePairer
# ---------------------------------------------------------------------------


def _touch_tif_files(folder: Path, filenames: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for fn in filenames:
        (folder / fn).write_bytes(b"\x00")


def test_frame_pairer_all_ok(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    rows = _make_clean_rows(4)
    filenames = [r["pair_filename"] for r in rows]
    _touch_tif_files(dir_a, filenames)
    _touch_tif_files(dir_b, filenames)
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    pairs = FramePairer().pair(str(dir_a), str(dir_b), records)

    assert len(pairs) == 4
    assert all(p.status == "ok" for p in pairs)


def test_frame_pairer_detects_missing_b(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    rows = _make_clean_rows(4)
    filenames = [r["pair_filename"] for r in rows]
    _touch_tif_files(dir_a, filenames)
    _touch_tif_files(dir_b, filenames[:3])  # B 路少一帧
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    pairs = FramePairer().pair(str(dir_a), str(dir_b), records)

    missing = [p for p in pairs if p.status == "missing_b"]
    assert len(missing) == 1
    assert missing[0].filename == filenames[3]


def test_frame_pairer_detects_extra_file(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    rows = _make_clean_rows(3)
    filenames = [r["pair_filename"] for r in rows]
    _touch_tif_files(dir_a, filenames + ["frame_999.tif"])  # A 路多一个 timing 没记录的文件
    _touch_tif_files(dir_b, filenames)
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    pairs = FramePairer().pair(str(dir_a), str(dir_b), records)

    extras = [p for p in pairs if p.status == "extra_a"]
    assert len(extras) == 1
    assert extras[0].filename == "frame_999.tif"


def test_frame_pairer_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FramePairer().pair(str(tmp_path / "不存在的目录"), str(tmp_path), [])


# ---------------------------------------------------------------------------
# 新增字段：requested_analog_gain / a_analog_gain / b_analog_gain
# ---------------------------------------------------------------------------


def test_metadata_reader_reads_requested_analog_gain(tmp_path):
    p = tmp_path / "capture_metadata.json"
    _write_metadata(p)
    meta = MetadataReader().read(str(p))
    assert meta.requested_analog_gain == 15


def test_timing_reader_reads_analog_gain(tmp_path):
    p = tmp_path / "capture_timing.csv"
    _write_timing_csv(p, _make_clean_rows(3))
    records = TimingReader().read(str(p))
    assert records[0].a_analog_gain == 3.75
    assert records[0].b_analog_gain == 3.75


# ---------------------------------------------------------------------------
# sha256_of_file
# ---------------------------------------------------------------------------


def test_sha256_of_file_matches_hashlib(tmp_path):
    import hashlib

    from rain.data.file_hash import sha256_of_file

    p = tmp_path / "x.bin"
    p.write_bytes(b"hello rain")

    expected = hashlib.sha256(b"hello rain").hexdigest()
    assert sha256_of_file(str(p)) == expected


def test_sha256_of_file_identical_content_same_hash(tmp_path):
    from rain.data.file_hash import sha256_of_file

    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"same content")
    p2.write_bytes(b"same content")

    assert sha256_of_file(str(p1)) == sha256_of_file(str(p2))


def test_sha256_of_file_different_content_different_hash(tmp_path):
    from rain.data.file_hash import sha256_of_file

    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"content one")
    p2.write_bytes(b"content two")

    assert sha256_of_file(str(p1)) != sha256_of_file(str(p2))


# ---------------------------------------------------------------------------
# ImageIntegrityChecker
# ---------------------------------------------------------------------------


def _write_fake_tif(path: Path, height=20, width=30, value=100) -> None:
    """用 OpenCV 编码一张真正能被解码的小灰度 TIF，供完整性检查测试用。"""
    import cv2
    import numpy as np

    img = np.full((height, width), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".tif", img)
    assert ok
    path.write_bytes(buf.tobytes())


def test_image_integrity_checker_decodes_valid_tif(tmp_path):
    from rain.data.image_integrity_checker import ImageIntegrityChecker

    folder = tmp_path / "a"
    folder.mkdir()
    _write_fake_tif(folder / "frame_000.tif", height=20, width=30)

    reports = ImageIntegrityChecker().scan_camera("a", str(folder))

    assert len(reports) == 1
    r = reports[0]
    assert r.ok is True
    assert r.width == 30 and r.height == 20
    assert r.dtype == "uint8"
    assert r.n_channels == 1
    assert r.error is None


def test_image_integrity_checker_flags_empty_file(tmp_path):
    from rain.data.image_integrity_checker import ImageIntegrityChecker

    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "frame_000.tif").write_bytes(b"")  # 0 字节空文件

    reports = ImageIntegrityChecker().scan_camera("a", str(folder))

    assert reports[0].ok is False
    assert "空" in reports[0].error


def test_image_integrity_checker_flags_corrupt_file(tmp_path):
    from rain.data.image_integrity_checker import ImageIntegrityChecker

    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "frame_000.tif").write_bytes(b"\x00\x01\x02not a real tif")  # 非法内容

    reports = ImageIntegrityChecker().scan_camera("a", str(folder))

    assert reports[0].ok is False
    assert reports[0].error is not None


def test_image_integrity_summarize_counts_resolution_and_dtype(tmp_path):
    from rain.data.image_integrity_checker import ImageIntegrityChecker, summarize

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_fake_tif(dir_a / "frame_000.tif", height=20, width=30)
    _write_fake_tif(dir_b / "frame_000.tif", height=20, width=30, value=50)  # 内容不同

    checker = ImageIntegrityChecker()
    reports_a = checker.scan_camera("a", str(dir_a))
    reports_b = checker.scan_camera("b", str(dir_b))
    summary = summarize(reports_a, reports_b)

    assert summary.n_total == 2
    assert summary.n_ok == 2
    assert summary.n_corrupt_or_empty == 0
    assert summary.resolution_counts == {"30x20": 2}
    assert summary.dtype_channel_counts == {"uint8_ch1": 2}
    assert summary.n_identical_ab_pairs == 0  # 内容不同，不应被判定为"异常相同"


def test_image_integrity_summarize_detects_identical_ab_content(tmp_path):
    """A/B 同名文件如果内容逐字节相同，必须被识别为异常（正常不可能发生）。"""
    from rain.data.image_integrity_checker import ImageIntegrityChecker, summarize

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_fake_tif(dir_a / "frame_000.tif", height=20, width=30, value=77)
    _write_fake_tif(dir_b / "frame_000.tif", height=20, width=30, value=77)  # 内容故意造成一样

    checker = ImageIntegrityChecker()
    reports_a = checker.scan_camera("a", str(dir_a))
    reports_b = checker.scan_camera("b", str(dir_b))
    summary = summarize(reports_a, reports_b)

    assert summary.n_identical_ab_pairs == 1
    assert summary.identical_ab_filenames == ["frame_000.tif"]


def test_image_integrity_checker_missing_dir_raises(tmp_path):
    from rain.data.image_integrity_checker import ImageIntegrityChecker

    with pytest.raises(FileNotFoundError):
        ImageIntegrityChecker().scan_camera("a", str(tmp_path / "不存在"))


# ---------------------------------------------------------------------------
# output_integrity
# ---------------------------------------------------------------------------


def test_check_frame_pairing_uniqueness_passes_on_clean_data(tmp_path):
    from rain.data.output_integrity import check_frame_pairing_uniqueness

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    rows = _make_clean_rows(4)
    filenames = [r["pair_filename"] for r in rows]
    _touch_tif_files(dir_a, filenames)
    _touch_tif_files(dir_b, filenames)
    records = [TimingReader._row_to_record(_stringify(r)) for r in rows]

    pairs = FramePairer().pair(str(dir_a), str(dir_b), records)
    problems = check_frame_pairing_uniqueness(pairs, expected_n=4)

    assert problems == []


def test_check_frame_pairing_uniqueness_detects_duplicate():
    from rain.data.output_integrity import check_frame_pairing_uniqueness
    from rain.data.types import FramePairRecord

    dup_pairs = [
        FramePairRecord(pair_index=0, filename="a.tif", path_a="a", path_b="b", status="ok"),
        FramePairRecord(pair_index=0, filename="a.tif", path_a="a", path_b="b", status="ok"),
        FramePairRecord(pair_index=1, filename="b.tif", path_a="a", path_b="b", status="ok"),
    ]

    problems = check_frame_pairing_uniqueness(dup_pairs, expected_n=2)

    assert any("重复" in p for p in problems)


def test_check_frame_pairing_uniqueness_detects_count_mismatch():
    from rain.data.output_integrity import check_frame_pairing_uniqueness
    from rain.data.types import FramePairRecord

    pairs = [
        FramePairRecord(pair_index=0, filename="a.tif", path_a="a", path_b="b", status="ok"),
    ]

    problems = check_frame_pairing_uniqueness(pairs, expected_n=5)

    assert any("数量" in p for p in problems)


def test_check_timing_quality_coverage_passes_on_clean_data(tmp_path):
    from rain.data.output_integrity import check_timing_quality_coverage

    p = tmp_path / "capture_timing.csv"
    _write_timing_csv(p, _make_clean_rows(6, period_us=3200.0))
    records = TimingReader().read(str(p))
    report = TimeQualityChecker().check(records)

    problems = check_timing_quality_coverage(report, expected_n=6)

    assert problems == []


def test_check_timing_quality_coverage_detects_count_mismatch(tmp_path):
    from rain.data.output_integrity import check_timing_quality_coverage

    p = tmp_path / "capture_timing.csv"
    _write_timing_csv(p, _make_clean_rows(6, period_us=3200.0))
    records = TimingReader().read(str(p))
    report = TimeQualityChecker().check(records)

    problems = check_timing_quality_coverage(report, expected_n=180)

    assert any("数量" in p or "不符" in p for p in problems)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
