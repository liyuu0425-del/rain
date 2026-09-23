#!/usr/bin/env python3
"""阶段1：数据读取 —— 元数据/时间戳读取、双路帧配对、时间质量检查、
图像完整性扫描、输出自检。

## 用途

读 capture_metadata.json + capture_timing.csv + 全部 TIF 图像，产出审计
文件，供人工复核这批数据是否可以进入阶段2（检测拆分）。这一步不做任何
检测算法计算，只回答"这批数据本身靠不靠谱、我们自己的配对/统计代码有没
有 bug"。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage1_data_audit.py
    python pipeline\\stage1_data_audit.py --config config\\pipeline.yaml

## 输出（默认写到 config/pipeline.yaml 里 data.output_dir 指定的目录）

    data_audit.json       整体审计摘要（元数据一致性、配对统计、时间质量、
                           图像完整性、模拟增益差异、输出自检、总体 PASS/FAIL）
    frame_pairing.csv     每个 pair_index 的 A/B 路径配对结果
    timing_quality.csv    每个 pair_index 的时间质量诊断（间隔/倒退/重复/漂移）
    image_integrity.csv   360 张 TIF 逐个文件的解码结果和格式信息

Python 语法说明（这个脚本里第一次出现的写法）：

- `import argparse`：
  标准库，用来解析命令行参数（就是 `--config xxx` 这种写法）。
  比自己写 `sys.argv[1]` 之类的解析更规范，还能自动生成 `--help` 说明。

- `if __name__ == "__main__":`：
  每个 Python 文件被执行时，Python 会给一个内置变量 `__name__` 赋值：
  直接运行这个文件（`python xxx.py`）时 `__name__` 等于字符串 `"__main__"`；
  被其它文件 `import` 时，`__name__` 等于模块名（比如 `"stage1_data_audit"`）。
  这行判断的效果是："只有直接运行这个文件时才执行 main()，被 import
  时不执行"——这样以后阶段2、阶段3 的脚本也能安全地
  `from pipeline.stage1_data_audit import xxx` 而不会意外触发一整套流程。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# 把项目根目录塞进模块搜索路径，这样才能 `import rain.xxx`。
# 和 tools/demo.py、tests/test_contracts.py 用的是同一套写法。
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from rain.data.file_hash import sha256_of_file
from rain.data.frame_pairer import FramePairer
from rain.data.image_integrity_checker import ImageIntegrityChecker, summarize as summarize_images
from rain.data.metadata_reader import MetadataReader
from rain.data.output_integrity import check_frame_pairing_uniqueness, check_timing_quality_coverage
from rain.data.time_quality_checker import TimeQualityChecker
from rain.data.timing_reader import TimingReader
from rain.data.types import (
    FramePairRecord,
    ImageIntegritySummary,
    PairTimingQC,
    TimingQualityReport,
)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段1：数据读取与审计")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dir_a = cfg["data"]["dir_a"]
    dir_b = cfg["data"]["dir_b"]
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段1：数据读取（含最终验收扩展项）")
    print("=" * 70)
    print(f"A 路目录 : {dir_a}")
    print(f"B 路目录 : {dir_b}")
    print(f"输出目录 : {output_dir}")
    print()

    # --- 1. 元数据 + A/B 副本哈希核对 --------------------------------------
    print("[1/6] 读取采集元数据 ...")
    meta_reader = MetadataReader()
    meta_a_path = str(Path(dir_a) / "capture_metadata.json")
    meta_b_path = str(Path(dir_b) / "capture_metadata.json")
    meta_a = meta_reader.read(meta_a_path)
    meta_b = meta_reader.read(meta_b_path)
    metadata_mismatches = meta_reader.compare(meta_a, meta_b)
    metadata_hash_check = _check_redundant_copy(meta_a_path, meta_b_path)

    print(f"      采集名称: {meta_a.capture_name}")
    print(f"      相机SN  : A={meta_a.camera_a_sn}  B={meta_a.camera_b_sn}")
    print(f"      触发模式: A={meta_a.a_trigger_mode}  B={meta_a.b_trigger_mode}")
    print(f"      请求帧对: {meta_a.requested_frame_pairs}  "
          f"实采完整帧对: {meta_a.captured_complete_frame_pairs}  "
          f"缺帧对: {meta_a.missing_frame_pairs}")
    print(f"      曝光(us): {meta_a.requested_exposure_us}")
    print(f"      A/B 两路 capture_metadata.json 字段级一致: {len(metadata_mismatches) == 0}")
    print(f"      A/B 两路 capture_metadata.json 字节级(sha256)一致: "
          f"{metadata_hash_check['byte_identical']}")
    if metadata_mismatches:
        print(f"      !! 字段不一致，共 {len(metadata_mismatches)} 项：")
        for m in metadata_mismatches:
            print(f"         {m}")

    # --- 2. 时间戳 + A/B 副本哈希核对 --------------------------------------
    print("\n[2/6] 读取逐帧设备时间戳 ...")
    timing_reader = TimingReader()
    timing_a_path = str(Path(dir_a) / "capture_timing.csv")
    timing_b_path = str(Path(dir_b) / "capture_timing.csv")
    timing_a = timing_reader.read(timing_a_path)
    timing_b = timing_reader.read(timing_b_path)
    timing_hash_check = _check_redundant_copy(timing_a_path, timing_b_path)

    print(f"      A 路 capture_timing.csv: {len(timing_a)} 行")
    print(f"      B 路 capture_timing.csv: {len(timing_b)} 行")
    print(f"      A/B 两路 capture_timing.csv 字节级(sha256)一致: "
          f"{timing_hash_check['byte_identical']}")
    print("      主记录来源(master_source): dir_a"
          "（A 是软件触发发起方，STROBE 硬件触发 B；哈希核对通过后，"
          "阶段2起只应读取 dir_a 下的这两份文件，不重复解析 B 路副本）")

    # 时间字段使用策略：写死在这里、也写进 data_audit.json，双重固定。
    timing_policy = {
        "a_time_field": "a_device_relative_us",
        "b_time_field": "b_device_relative_us",
        "forbidden_for_velocity": [
            "pair_filename 中的毫秒",
            "a_host_receive_wall_ns / b_host_receive_wall_ns",
            "frame_idx 除以固定帧率",
        ],
        "enforced_by": (
            "TimingRecord 只暴露 a_device_relative_us/b_device_relative_us 供测速使用；"
            "host_receive_wall_ns 仅保留用于审计交叉核对，测速阶段（阶段5）不得读取它。"
        ),
    }
    print(f"      A 测速字段: {timing_policy['a_time_field']}")
    print(f"      B 测速字段: {timing_policy['b_time_field']}")
    print("      禁止用于测速: " + "；".join(timing_policy["forbidden_for_velocity"]))

    # A、B 两路目录下的 capture_timing.csv 内容应完全相同；哈希一致时，
    # 后续步骤统一用 A 路解析结果，不重复处理 B 路那一份。
    timing = timing_a

    # --- 3. 双路帧配对 -------------------------------------------------
    print("\n[3/6] 双路帧配对 ...")
    pairer = FramePairer()
    frame_pairs = pairer.pair(dir_a, dir_b, timing)
    pairing_summary = _summarize_pairing(frame_pairs)
    print(f"      pair_index 总数: {pairing_summary['n_pairs_total']}")
    print(f"      配对成功(ok)   : {pairing_summary['n_ok']}")
    for status in ("missing_a", "missing_b", "missing_both", "extra_a", "extra_b"):
        n = pairing_summary["by_status"].get(status, 0)
        if n:
            print(f"      {status:13s}: {n}")

    # --- 4. 时间质量检查 -------------------------------------------------
    print("\n[4/6] 时间质量检查 ...")
    checker = TimeQualityChecker(**cfg["time_quality"])
    quality = checker.check(timing, metadata=meta_a)
    print(f"      实测帧周期 中位 {quality.measured_frame_period_median_us:.1f} us  "
          f"P95 {quality.measured_frame_period_p95_us:.1f} us  "
          f"最大 {quality.measured_frame_period_max_us:.1f} us")
    print(f"      对应帧率(中位) : {1e6 / quality.measured_frame_period_median_us:.2f} 帧/秒")
    print(f"      pair_index 缺号: {quality.n_missing_pair_index}")
    print(f"      时间倒退       : {quality.n_time_backwards}")
    print(f"      重复时间戳     : {quality.n_duplicate_time}")
    print(f"      间隔异常       : {quality.n_interval_outliers}")
    print(f"      双路漂移异常   : {quality.n_drift_outliers}")
    print(f"      标记异常的 pair 总数: {quality.n_flagged_total} / {quality.n_records}")

    # --- 4.5 模拟增益字段核对（只记录矛盾，不自行换算/修正）----------------
    gains_a = sorted({r.a_analog_gain for r in timing})
    gains_b = sorted({r.b_analog_gain for r in timing})
    analog_gain_check = {
        "requested_analog_gain": meta_a.requested_analog_gain,
        "observed_a_analog_gain_values": gains_a,
        "observed_b_analog_gain_values": gains_b,
        "requested_matches_observed": (
            len(gains_a) == 1 and len(gains_b) == 1
            and float(gains_a[0]) == float(meta_a.requested_analog_gain)
            and float(gains_b[0]) == float(meta_a.requested_analog_gain)
        ),
        "status": "unit_or_meaning_unconfirmed",
        "note": (
            f"capture_metadata.json 的 requested_analog_gain="
            f"{meta_a.requested_analog_gain}，但 capture_timing.csv 里实测的 "
            f"a/b_analog_gain={gains_a[0] if len(gains_a) == 1 else gains_a}。"
            "两者数值不一致，可能是不同量纲（如档位序号 vs 线性倍数/dB）或不同"
            "字段含义，尚未确认。此处如实记录两个数值，不自行换算或判定哪个正确。"
        ),
    }
    print("\n      [增益字段核对] requested_analog_gain="
          f"{analog_gain_check['requested_analog_gain']}  "
          f"实测 a/b_analog_gain={gains_a}/{gains_b}  "
          f"一致: {analog_gain_check['requested_matches_observed']}  "
          "-> 标记为 unit_or_meaning_unconfirmed，不自行修正")

    # --- 5. 图像完整性扫描（360 张 TIF）------------------------------------
    print("\n[5/6] 扫描 A/B 全部 TIF 完整性 ...")
    img_checker = ImageIntegrityChecker()
    reports_a = img_checker.scan_camera("a", dir_a)
    reports_b = img_checker.scan_camera("b", dir_b)
    image_summary = summarize_images(reports_a, reports_b)

    print(f"      文件总数: {image_summary.n_total}"
          f"（A={len(reports_a)}  B={len(reports_b)}）")
    print(f"      成功解码: {image_summary.n_ok}")
    print(f"      损坏/空文件: {image_summary.n_corrupt_or_empty}")
    print(f"      分辨率分布: {image_summary.resolution_counts}")
    print(f"      位深/通道分布: {image_summary.dtype_channel_counts}")
    print(f"      A/B 同名文件内容完全相同(异常)的数量: {image_summary.n_identical_ab_pairs}")
    if image_summary.identical_ab_filenames:
        print(f"         文件名: {image_summary.identical_ab_filenames}")
    if image_summary.n_corrupt_or_empty:
        for r in image_summary.per_file:
            if not r.ok:
                print(f"      !! 解码失败: [{r.camera}] {r.filename} -> {r.error}")

    # --- 6. 输出自检（防止我们自己的代码产出重复/缺行）----------------------
    print("\n[6/6] 输出文件自检 ...")
    output_problems: list[str] = []
    output_problems += check_frame_pairing_uniqueness(
        frame_pairs, expected_n=meta_a.requested_frame_pairs
    )
    output_problems += check_timing_quality_coverage(
        quality, expected_n=meta_a.requested_frame_pairs
    )
    if output_problems:
        print(f"      !! 发现 {len(output_problems)} 个问题：")
        for p in output_problems:
            print(f"         {p}")
    else:
        print(f"      frame_pairing: {pairing_summary['n_pairs_total']} 条，"
              f"pair_index 全部唯一，与预期帧对数 {meta_a.requested_frame_pairs} 一致")
        print(f"      timing_quality: {len(quality.per_pair)} 条，覆盖全部 "
              f"{meta_a.requested_frame_pairs} 个图像对（每对一行，A/B 诊断同列，"
              "不采用逐相机拆行格式），无重复记录")

    # --- 写输出文件 -----------------------------------------------------
    print("\n写输出文件 ...")
    frame_pairing_csv = output_dir / "frame_pairing.csv"
    timing_quality_csv = output_dir / "timing_quality.csv"
    image_integrity_csv = output_dir / "image_integrity.csv"
    data_audit_json = output_dir / "data_audit.json"

    _write_frame_pairing_csv(frame_pairs, frame_pairing_csv)
    _write_timing_quality_csv(quality, timing_quality_csv)
    _write_image_integrity_csv(image_summary, image_integrity_csv)

    overall_pass, gate_results = _evaluate_pass_fail(
        metadata_mismatches=metadata_mismatches,
        metadata_hash_check=metadata_hash_check,
        timing_hash_check=timing_hash_check,
        pairing_summary=pairing_summary,
        image_summary=image_summary,
        output_problems=output_problems,
        expected_n=meta_a.requested_frame_pairs,
    )

    _write_data_audit_json(
        meta_a=meta_a,
        metadata_mismatches=metadata_mismatches,
        metadata_hash_check=metadata_hash_check,
        timing_hash_check=timing_hash_check,
        timing_policy=timing_policy,
        pairing_summary=pairing_summary,
        quality=quality,
        analog_gain_check=analog_gain_check,
        image_summary=image_summary,
        output_problems=output_problems,
        overall_pass=overall_pass,
        gate_results=gate_results,
        out_path=data_audit_json,
    )

    print(f"  {frame_pairing_csv}")
    print(f"  {timing_quality_csv}")
    print(f"  {image_integrity_csv}")
    print(f"  {data_audit_json}")
    print()
    print("-" * 70)
    for name, ok in gate_results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 70)
    print(f"阶段1最终验收：{'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        print("存在未通过项，见上方列表和 data_audit.json，禁止进入阶段2。")
    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _check_redundant_copy(path_a: str, path_b: str) -> dict:
    """核对 A、B 两路目录下"理应完全相同"的一份文件是否真的逐字节相同。"""
    hash_a = sha256_of_file(path_a)
    hash_b = sha256_of_file(path_b)
    return {
        "path_a": path_a,
        "path_b": path_b,
        "sha256_a": hash_a,
        "sha256_b": hash_b,
        "byte_identical": hash_a == hash_b,
    }


def _summarize_pairing(frame_pairs: list[FramePairRecord]) -> dict:
    """统计配对结果里各种 status 各出现多少次。

    Python 语法说明：
    - `dict` 花括号 `{}` 造一个空字典，`d[key] = d.get(key, 0) + 1`
      是"计数"的标准写法：`dict.get(key, 默认值)` 取值时，key 不存在就
      返回默认值（这里是 0），而不是像 `d[key]` 那样直接报错 KeyError。
    """
    by_status: dict[str, int] = {}
    for rec in frame_pairs:
        by_status[rec.status] = by_status.get(rec.status, 0) + 1

    return {
        "n_pairs_total": len(frame_pairs),
        "n_ok": by_status.get("ok", 0),
        "by_status": by_status,
    }


def _write_frame_pairing_csv(frame_pairs: list[FramePairRecord], out_path: Path) -> None:
    columns = ["pair_index", "filename", "path_a", "path_b", "status", "note"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for rec in frame_pairs:
            w.writerow([
                rec.pair_index, rec.filename,
                rec.path_a or "", rec.path_b or "",
                rec.status, rec.note,
            ])


def _write_timing_quality_csv(report: TimingQualityReport, out_path: Path) -> None:
    columns = [
        "pair_index", "a_device_relative_us", "b_device_relative_us",
        "interval_a_us", "interval_b_us", "drift_us",
        "is_time_backwards_a", "is_time_backwards_b",
        "is_duplicate_time_a", "is_duplicate_time_b",
        "is_interval_outlier_a", "is_interval_outlier_b",
        "is_drift_outlier", "flags_text",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for qc in report.per_pair:
            w.writerow(_qc_row(qc))


def _qc_row(qc: PairTimingQC) -> list:
    return [
        qc.pair_index,
        f"{qc.a_device_relative_us:.1f}",
        f"{qc.b_device_relative_us:.1f}",
        "" if qc.interval_a_us is None else f"{qc.interval_a_us:.1f}",
        "" if qc.interval_b_us is None else f"{qc.interval_b_us:.1f}",
        f"{qc.drift_us:.1f}",
        int(qc.is_time_backwards_a), int(qc.is_time_backwards_b),
        int(qc.is_duplicate_time_a), int(qc.is_duplicate_time_b),
        int(qc.is_interval_outlier_a), int(qc.is_interval_outlier_b),
        int(qc.is_drift_outlier),
        qc.flags_text,
    ]


def _write_image_integrity_csv(summary: ImageIntegritySummary, out_path: Path) -> None:
    columns = [
        "camera", "filename", "ok", "error", "width", "height",
        "dtype", "n_channels", "file_size_bytes", "sha256",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in summary.per_file:
            w.writerow([
                r.camera, r.filename, int(r.ok), r.error or "",
                r.width if r.width is not None else "",
                r.height if r.height is not None else "",
                r.dtype or "",
                r.n_channels if r.n_channels is not None else "",
                r.file_size_bytes, r.sha256,
            ])


def _evaluate_pass_fail(
    metadata_mismatches: list[str],
    metadata_hash_check: dict,
    timing_hash_check: dict,
    pairing_summary: dict,
    image_summary: ImageIntegritySummary,
    output_problems: list[str],
    expected_n: int,
) -> tuple[bool, dict]:
    """汇总阶段1最终验收的各个硬性门槛，返回 (总体是否通过, 每项门槛的结果)。

    只对"数据本身或代码本身有没有 bug"的项目做硬性门槛；已经明确标记为
    "低置信度/待确认"的项（比如增益字段矛盾、时间质量里的少量异常帧）
    不在这里拦截——它们的处理方式是打标记而不是拒绝整批数据。
    """
    gate_results = {
        "元数据A/B字段一致": len(metadata_mismatches) == 0,
        "元数据A/B字节级(sha256)一致": metadata_hash_check["byte_identical"],
        "时间表A/B字节级(sha256)一致": timing_hash_check["byte_identical"],
        "双路帧配对全部成功(无缺帧/多余文件)": (
            pairing_summary["n_ok"] == expected_n
            and pairing_summary["n_pairs_total"] == expected_n
        ),
        "图像全部成功解码(360/360)": image_summary.n_corrupt_or_empty == 0,
        "无A/B同名文件内容异常相同": image_summary.n_identical_ab_pairs == 0,
        "输出文件无重复记录且行数达标": len(output_problems) == 0,
    }
    overall = all(gate_results.values())
    return overall, gate_results


def _write_data_audit_json(
    meta_a,
    metadata_mismatches: list[str],
    metadata_hash_check: dict,
    timing_hash_check: dict,
    timing_policy: dict,
    pairing_summary: dict,
    quality: TimingQualityReport,
    analog_gain_check: dict,
    image_summary: ImageIntegritySummary,
    output_problems: list[str],
    overall_pass: bool,
    gate_results: dict,
    out_path: Path,
) -> None:
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_name": meta_a.capture_name,
        "camera_a_sn": meta_a.camera_a_sn,
        "camera_b_sn": meta_a.camera_b_sn,
        "a_trigger_mode": meta_a.a_trigger_mode,
        "b_trigger_mode": meta_a.b_trigger_mode,
        "timing_model": meta_a.timing_model,
        "requested_frame_pairs": meta_a.requested_frame_pairs,
        "requested_width": meta_a.requested_width,
        "requested_height": meta_a.requested_height,
        "requested_exposure_us": meta_a.requested_exposure_us,
        "captured_complete_frame_pairs": meta_a.captured_complete_frame_pairs,
        "capture_complete": meta_a.capture_complete,
        "missing_frame_pairs_reported": meta_a.missing_frame_pairs,
        "metadata_a_b_consistent": len(metadata_mismatches) == 0,
        "metadata_mismatches": metadata_mismatches,
        "redundant_copies_policy": {
            "capture_metadata_json": metadata_hash_check,
            "capture_timing_csv": timing_hash_check,
            "master_source": "dir_a",
            "rationale": (
                "A、B 两路目录下各存一份内容相同的 capture_metadata.json / "
                "capture_timing.csv（同一次采集程序统一写出、双路各留一份归档）。"
                "A 是软件触发发起方（timing_model: A每帧STROBE硬件触发B），"
                "选 dir_a 作为主记录来源。"
            ),
            "downstream_policy": (
                "阶段1需要读取两份副本一次以完成字节级(sha256)一致性核对；"
                "哈希核对通过后，阶段2起的所有模块只应读取 dir_a 下的这两份文件，"
                "不再重复解析 dir_b 的副本。若未来某次采集两份哈希不一致，"
                "必须停止自动选择、交由人工确认后再继续。"
            ),
        },
        "timing_policy": timing_policy,
        "frame_pairing": pairing_summary,
        "timing_quality": {
            "n_records": quality.n_records,
            "measured_frame_period_median_us": quality.measured_frame_period_median_us,
            "measured_frame_period_p95_us": quality.measured_frame_period_p95_us,
            "measured_frame_period_max_us": quality.measured_frame_period_max_us,
            "measured_frame_rate_median_hz": 1e6 / quality.measured_frame_period_median_us,
            "n_missing_pair_index": quality.n_missing_pair_index,
            "missing_pair_indices": quality.missing_pair_indices,
            "n_time_backwards": quality.n_time_backwards,
            "n_duplicate_time": quality.n_duplicate_time,
            "n_interval_outliers": quality.n_interval_outliers,
            "n_drift_outliers": quality.n_drift_outliers,
            "n_flagged_total": quality.n_flagged_total,
        },
        "analog_gain_check": analog_gain_check,
        "image_integrity": {
            "n_total": image_summary.n_total,
            "n_ok": image_summary.n_ok,
            "n_corrupt_or_empty": image_summary.n_corrupt_or_empty,
            "resolution_counts": image_summary.resolution_counts,
            "dtype_channel_counts": image_summary.dtype_channel_counts,
            "n_identical_ab_pairs": image_summary.n_identical_ab_pairs,
            "identical_ab_filenames": image_summary.identical_ab_filenames,
        },
        "output_file_checks": {
            "frame_pairing_csv_rows": pairing_summary["n_pairs_total"],
            "timing_quality_csv_rows": len(quality.per_pair),
            "format_note": (
                "timing_quality.csv 采用「每个图像对一行、A/B 诊断在同一行不同列」"
                "的格式，覆盖全部图像对；未采用逐相机拆分成两行的格式。"
            ),
            "problems": output_problems,
            "no_duplicates_confirmed": len(output_problems) == 0,
        },
        "source_data_untouched": {
            "note": (
                "本次运行对 dir_a / dir_b 下的所有文件只做只读操作"
                "（json.loads / csv.DictReader / sha256 / cv2.imdecode 均为只读），"
                "代码路径中没有任何对这两个目录的写入或删除调用。"
            ),
        },
        "overall_pass": overall_pass,
        "gate_results": gate_results,
        "known_constraints": [
            "测速只能用 a_device_relative_us / b_device_relative_us，"
            "禁止用文件名毫秒、host 接收时间或帧号/固定帧率。",
            "A/B 设备时钟零点独立，双路固定曝光延迟尚未标定。",
            "观测体积 590.4 cm^3 是名义值，尚未实验标定。",
            "requested_analog_gain 与实测 a/b_analog_gain 数值不一致，"
            "含义待确认，见 analog_gain_check，未自行修正。",
        ],
    }

    # json.dump(对象, 文件句柄, ...)：把 Python 对象（这里是 dict）写成
    # JSON 文本。ensure_ascii=False 让中文直接以中文字符写入（不转义成
    # \uXXXX），indent=2 让输出带缩进、方便人眼查看。
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
