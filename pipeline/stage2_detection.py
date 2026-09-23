#!/usr/bin/env python3
"""阶段2：检测拆分 —— 用真实数据验证六个检测子模块与原 Demo 结果一致。

## 用途

用阶段1已验收通过的真实数据（A-Capture_085820 / B-Capture_085820），
对 A、B 两路各自用同一套检测配置，分别跑：
    - 原 Demo 的 `SimpleThresholdDetector`（rain/detectors/simple_threshold.py，
      完全未改动）
    - 拆分后的 `ModularPipelineDetector`（六个独立子模块按顺序调用）
逐帧比较两者的检测结果，验证"拆分模块不改变检测算法行为"。同时产出中间图、
检测叠加图和 detections.csv，供人工复核。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage2_detection.py
    python pipeline\\stage2_detection.py --config config\\pipeline.yaml

## 输出（默认写到 config/pipeline.yaml 里 detection.output_dir 指定的目录）

    a_01_原图.png ... a_07_检测叠加.png   A路中间产物（取某一帧）
    b_01_原图.png ... b_07_检测叠加.png   B路中间产物（取某一帧）
    overlay/a_frame_XXXX.png / overlay/b_frame_XXXX.png   均匀抽样的检测叠加图
    detections.csv                         A、B两路全部帧的候选雨滴（cam列区分）
    stage2_report.json                     与原Demo的逐帧比较结果 + 真实数据现状说明

## 重要提醒（写进 stage2_report.json，这里也重复一遍）

本次真实数据运行使用**固定阈值(fixed)**；自适应阈值(adaptive_mean/
adaptive_gaussian)只完成了实现和单元测试，尚未用真实标注数据比较过。
阈值、面积门限沿用旧数据集(cama/camb)的经验参数，未针对这批新数据标定。
检出数量不等于真实雨滴总数。"拆分后与原Demo结果一致"只证明重构正确，
不代表检测精度（召回率/误检率）已经过验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

import rain  # noqa: F401  触发所有实现的注册
from rain.detectors.simple_threshold import SimpleThresholdDetector
from rain.imageio import imwrite_unicode
from rain.io_index import parse_frame_id
from rain.registry import build_background, build_detector, build_reader
from rain.sink import DETECTION_COLUMNS, _detection_row, draw_overlay
from rain.types import Detection, qc_to_text

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段2：检测拆分，A/B两路与原Demo等价性验证")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    det_cfg = cfg["detection"]
    output_dir = Path(det_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段2：检测拆分（A/B两路，与原Demo逐帧等价性验证）")
    print("=" * 70)
    print(f"A 路目录 : {cfg['data']['dir_a']}")
    print(f"B 路目录 : {cfg['data']['dir_b']}")
    print(f"输出目录 : {output_dir}")

    reader = build_reader(cfg["reader"])

    camera_results: dict[str, dict] = {}
    all_dets_for_csv: list[tuple[str, Detection]] = []  # (cam, Detection)，写进同一份 detections.csv

    for cam, dir_key in (("a", "dir_a"), ("b", "dir_b")):
        print(f"\n{'-'*70}\n处理 {cam.upper()} 路\n{'-'*70}")
        try:
            result = _process_camera(
                cam=cam,
                folder=cfg["data"][dir_key],
                cfg=cfg,
                det_cfg=det_cfg,
                reader=reader,
                output_dir=output_dir,
            )
            camera_results[cam] = result
            all_dets_for_csv.extend((cam, d) for d in result["all_dets_modular"])
            print(f"  [{cam.upper()}] 处理完成，无异常退出")
        except Exception as exc:  # noqa: BLE001 - 一路失败不能连累另一路，也不能静默吞掉
            tb = traceback.format_exc(limit=5)
            camera_results[cam] = {"error": f"{type(exc).__name__}: {exc}", "traceback": tb}
            print(f"  !! [{cam.upper()}] 处理失败: {exc}")
            print(tb)

    # --- 合并写 detections.csv（A、B 两路都在里面，用 cam 列区分）-----------
    detections_csv = output_dir / "detections.csv"
    _write_detections_csv(all_dets_for_csv, detections_csv)
    print(f"\n{detections_csv} 共 {len(all_dets_for_csv)} 行"
          f"（A={sum(1 for c,_ in all_dets_for_csv if c=='a')}，"
          f"B={sum(1 for c,_ in all_dets_for_csv if c=='b')}）")

    # --- 写 stage2_report.json（含真实数据现状的明确说明）--------------------
    report_path = output_dir / "stage2_report.json"
    _write_report(report_path, cfg, det_cfg, camera_results)
    print(f"{report_path}")

    # --- 总体判定 -------------------------------------------------------
    both_ok = all("error" not in r for r in camera_results.values())
    both_180 = all(r.get("n_frames") == 180 for r in camera_results.values() if "error" not in r)
    both_equiv = all(
        r.get("directly_comparable") and r.get("n_frames_different", 1) == 0
        for r in camera_results.values() if "error" not in r
    )
    module_refactor_pass = both_ok and both_180 and both_equiv

    print()
    print("=" * 70)
    print(f"[{'PASS' if module_refactor_pass else 'FAIL'}] 模块化重构与原Demo一致性")
    print("[LIMITED] 真实数据检测参数尚未标定（阈值/面积门限沿用旧数据集经验值，"
          "检出数量不代表真实雨滴总数，详见 stage2_report.json）")
    print("=" * 70)
    return 0 if module_refactor_pass else 1


# ---------------------------------------------------------------------------
# 单路处理
# ---------------------------------------------------------------------------


def _process_camera(cam: str, folder: str, cfg: dict, det_cfg: dict, reader, output_dir: Path) -> dict:
    """处理单路相机的全部帧：读取、背景建模、双检测器逐帧比较、写中间图/叠加图。

    返回一个字典，包含这一路的全部统计数据，供 main() 汇总进
    stage2_report.json，以及供 detections.csv 合并使用。
    """
    all_frames = _list_camera_frames(folder)
    n = min(int(det_cfg["n_frames"]), len(all_frames))
    frame_ids = [fid for fid, _ in all_frames[:n]]
    paths = [p for _, p in all_frames[:n]]

    t0 = time.perf_counter()
    frames = [reader.read(p) for p in paths]
    ms_read = (time.perf_counter() - t0) * 1000
    print(f"  [1/5] 读取 {n}/{len(all_frames)} 帧: {ms_read:.0f} ms ({ms_read/n:.1f} ms/帧)")

    bgmodel = build_background(cfg["background"])
    t0 = time.perf_counter()
    bg = bgmodel.build(frames)
    ms_bg = (time.perf_counter() - t0) * 1000
    print(f"  [2/5] 背景建模: {ms_bg:.0f} ms  背景平均灰度 {bg.mean():.1f}")

    seg_p = det_cfg["segmenter"]["params"]
    morph_p = det_cfg["morphology"]["params"]
    ce_p = det_cfg["contour_extractor"]["params"]
    norm_p = det_cfg["normalizer"]["params"]
    diff_p = det_cfg["differencer"]["params"]

    directly_comparable = (
        norm_p.get("method", "none") == "none"
        and diff_p.get("mode", "bg_minus_img") == "bg_minus_img"
        and seg_p.get("method", "fixed") == "fixed"
    )

    simple_detector = SimpleThresholdDetector(
        thresh=seg_p.get("thresh", 20),
        open_ksize=morph_p.get("open_ksize", 3),
        min_area=ce_p.get("min_area", 10.0),
        max_area=ce_p.get("max_area", 5000.0),
        max_aspect=ce_p.get("max_aspect", 3.5),
    )
    modular_detector = build_detector({
        "name": "modular_pipeline",
        "params": {
            "normalizer": det_cfg["normalizer"],
            "differencer": det_cfg["differencer"],
            "segmenter": det_cfg["segmenter"],
            "morphology": det_cfg["morphology"],
            "cc_analyzer": det_cfg["cc_analyzer"],
            "contour_extractor": det_cfg["contour_extractor"],
        },
    })
    print(f"  [3/5] 检测器就绪。可直接逐帧比较(参数与原Demo默认行为对齐): {directly_comparable}")

    per_frame_modular: list[list[Detection]] = []
    per_frame_simple: list[list[Detection]] = []
    frame_diffs: list[dict] = []

    t0 = time.perf_counter()
    for img, fid in zip(frames, frame_ids):
        dets_simple = simple_detector.detect(img, bg, frame_idx=fid)
        dets_modular = modular_detector.detect(img, bg, frame_idx=fid)
        per_frame_simple.append(dets_simple)
        per_frame_modular.append(dets_modular)

        diff = _compare_detections(dets_simple, dets_modular)
        if diff is not None:
            frame_diffs.append({"frame_idx": fid, "reason": diff})
    ms_detect = (time.perf_counter() - t0) * 1000

    all_dets_modular = [d for lst in per_frame_modular for d in lst]
    all_dets_simple = [d for lst in per_frame_simple for d in lst]
    n_identical_frames = n - len(frame_diffs)
    qc_dist = _qc_distribution(all_dets_modular)

    print(f"  [4/5] 逐帧检测完成: {ms_detect:.0f} ms ({ms_detect/n:.2f} ms/帧，两个检测器合计)")
    print(f"        原Demo检出 {len(all_dets_simple)} 个，拆分流水线检出 {len(all_dets_modular)} 个，"
          f"逐帧一致 {n_identical_frames}/{n} 帧")
    print(f"        QC标记分布: {qc_dist}")
    if frame_diffs:
        print(f"        !! {len(frame_diffs)} 帧不一致，前5条: {frame_diffs[:5]}")

    print("  [5/5] 写中间图 + 叠加图 ...")
    show_idx = min(int(det_cfg.get("show_frame_index", 0)), n - 1)
    _save_intermediate_images(
        cam, frames[show_idx], bg, frame_ids[show_idx], modular_detector,
        per_frame_modular[show_idx], output_dir,
    )
    _save_overlay_samples(
        cam, frames, per_frame_modular, frame_ids, output_dir,
        n_overlay=int(det_cfg.get("overlay_samples", 10)),
    )

    areas = np.array([d.area_px for d in all_dets_modular]) if all_dets_modular else np.array([])
    deqs = np.array([d.d_eq_px for d in all_dets_modular]) if all_dets_modular else np.array([])

    return {
        "n_frames": n,
        "n_frames_available": len(all_frames),
        "directly_comparable": directly_comparable,
        "n_frames_different": len(frame_diffs),
        "frame_diffs": frame_diffs,
        "n_detections_simple_threshold": len(all_dets_simple),
        "n_detections_modular_pipeline": len(all_dets_modular),
        "qc_flag_distribution": qc_dist,
        "area_px_median": float(np.median(areas)) if areas.size else None,
        "area_px_min": float(areas.min()) if areas.size else None,
        "area_px_max": float(areas.max()) if areas.size else None,
        "d_eq_px_median": float(np.median(deqs)) if deqs.size else None,
        "timings_ms": {"read_ms": ms_read, "background_ms": ms_bg, "detect_ms_total": ms_detect},
        "background_mean_gray": float(bg.mean()),
        "all_dets_modular": all_dets_modular,  # 供 main() 合并写 CSV，不进最终 JSON 报告
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _list_camera_frames(folder: str) -> list[tuple[int, str]]:
    """列出目录下全部 TIF，按文件名里的帧号排序。

    复用 rain/io_index.py 里已有的 parse_frame_id（正则提取 frame_XXX 里的数字），
    不重新写一遍解析逻辑。
    """
    items = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith((".tif", ".tiff")):
            continue
        fid = parse_frame_id(fn)
        if fid is None:
            continue
        items.append((fid, os.path.join(folder, fn)))
    items.sort(key=lambda t: t[0])
    return items


def _compare_detections(a: list[Detection], b: list[Detection]) -> str | None:
    """比较同一帧两个检测器的结果，返回差异描述；完全一致返回 None。"""
    if len(a) != len(b):
        return f"检出数量不同: simple={len(a)} modular={len(b)}"
    for i, (da, db) in enumerate(zip(a, b)):
        if (
            abs(da.cx - db.cx) > 1e-6
            or abs(da.cy - db.cy) > 1e-6
            or abs(da.area_px - db.area_px) > 1e-6
            or da.bbox != db.bbox
            or da.qc_flags != db.qc_flags
        ):
            return f"第{i}个检测不同: simple(cx={da.cx:.3f},cy={da.cy:.3f}) vs modular(cx={db.cx:.3f},cy={db.cy:.3f})"
    return None


def _qc_distribution(dets: list[Detection]) -> dict[str, int]:
    """统计每种 QC 标记文本各出现多少次（见 rain/types.py 的 qc_to_text）。"""
    dist: dict[str, int] = {}
    for d in dets:
        label = qc_to_text(d.qc_flags)
        dist[label] = dist.get(label, 0) + 1
    return dist


def _save_intermediate_images(cam, img, bg, frame_idx, modular_detector, dets, output_dir: Path) -> None:
    prefix = f"{cam}_"
    imwrite_unicode(str(output_dir / f"{prefix}01_原图.png"), img)
    imwrite_unicode(str(output_dir / f"{prefix}02_背景图.png"), bg)

    dbg: dict = {}
    modular_detector.detect(img, bg, frame_idx=frame_idx, debug=dbg)

    if "normalized" in dbg:
        imwrite_unicode(str(output_dir / f"{prefix}03_归一化.png"), dbg["normalized"])
    if "diff" in dbg:
        imwrite_unicode(str(output_dir / f"{prefix}04_背景差分.png"), _stretch(dbg["diff"]))
    if "binary_raw" in dbg:
        imwrite_unicode(str(output_dir / f"{prefix}05_二值化.png"), dbg["binary_raw"])
    if "binary" in dbg:
        imwrite_unicode(str(output_dir / f"{prefix}06_形态学.png"), dbg["binary"])

    imwrite_unicode(str(output_dir / f"{prefix}07_检测叠加.png"), draw_overlay(img, dets))


def _stretch(img: np.ndarray) -> np.ndarray:
    """把低对比度的差分图拉伸到 0-255，只为了肉眼看得清，不参与任何计算。"""
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return img
    return ((img.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


def _save_overlay_samples(cam, frames, per_frame, frame_ids, output_dir: Path, n_overlay: int) -> None:
    if n_overlay <= 0 or not frames:
        return
    overlay_dir = output_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, len(frames) // n_overlay)
    for i in range(0, len(frames), step):
        out_path = overlay_dir / f"{cam}_frame_{frame_ids[i]:04d}.png"
        imwrite_unicode(str(out_path), draw_overlay(frames[i], per_frame[i]))


def _write_detections_csv(rows: list[tuple[str, Detection]], detections_csv: Path) -> None:
    """写合并后的 detections.csv：A、B 两路都在同一份文件里，用 cam 列区分。

    复用 rain/sink.py 已有的 DETECTION_COLUMNS / _detection_row，
    保证列结构和原 Demo 的 detections.csv 完全一致——这也是补充验收
    要求的"更换检测参数不需要改动CSV输出结构"的一部分：CSV 结构由
    rain/sink.py 定义，不受阶段2这几个新模块参数变化影响。
    """
    with open(detections_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETECTION_COLUMNS)
        for cam, d in rows:
            w.writerow(_detection_row(cam, cam, d))


def _write_report(report_path: Path, cfg: dict, det_cfg: dict, camera_results: dict) -> None:
    seg_p = det_cfg["segmenter"]["params"]

    # 只挑 JSON 可序列化、且对复核有意义的字段进 report；
    # all_dets_modular（Detection 对象列表，含 numpy 轮廓数组）不放进 JSON。
    per_camera_summary = {}
    for cam, r in camera_results.items():
        if "error" in r:
            per_camera_summary[cam] = {"status": "error", "error": r["error"]}
            continue
        per_camera_summary[cam] = {
            "status": "ok",
            "n_frames_processed": r["n_frames"],
            "n_frames_available": r["n_frames_available"],
            "all_180_frames_processed": r["n_frames"] == 180,
            "directly_comparable_config": r["directly_comparable"],
            "n_frames_identical_to_original_demo": r["n_frames"] - r["n_frames_different"],
            "n_frames_different": r["n_frames_different"],
            "frame_diffs": r["frame_diffs"],
            "n_detections_simple_threshold": r["n_detections_simple_threshold"],
            "n_detections_modular_pipeline": r["n_detections_modular_pipeline"],
            "qc_flag_distribution": r["qc_flag_distribution"],
            "area_px": {
                "median": r["area_px_median"], "min": r["area_px_min"], "max": r["area_px_max"],
            },
            "d_eq_px_median": r["d_eq_px_median"],
            "background_mean_gray": r["background_mean_gray"],
            "timings_ms": r["timings_ms"],
        }

    both_ok = all(r.get("status") == "ok" for r in per_camera_summary.values())
    both_180 = all(
        r.get("all_180_frames_processed") for r in per_camera_summary.values() if r.get("status") == "ok"
    )
    both_equiv = all(
        r.get("directly_comparable_config") and r.get("n_frames_different") == 0
        for r in per_camera_summary.values() if r.get("status") == "ok"
    )
    module_refactor_pass = both_ok and both_180 and both_equiv

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_camera": per_camera_summary,
        "stage2_status": {
            "module_refactor_vs_original_demo": "PASS" if module_refactor_pass else "FAIL",
            "real_data_detection_calibration": "LIMITED",
        },
        # 补充验收要求明确写出的五条现状说明，原样落盘，不做任何美化或弱化。
        "real_data_caveats": [
            {
                "item": "threshold_method_used",
                "detail": (
                    f"本次真实数据运行使用固定阈值 fixed threshold"
                    f"（method={seg_p.get('method','fixed')}, thresh={seg_p.get('thresh')}），"
                    "不是自适应阈值。"
                ),
            },
            {
                "item": "adaptive_threshold_status",
                "detail": (
                    "adaptive_mean 和 adaptive_gaussian 两种自适应阈值方法已经实现"
                    "（rain/detection/adaptive_threshold_segmenter.py）并有单元测试覆盖"
                    "（tests/test_detection_stage2.py），但只验证了'能正常运行、输出合法二值图'，"
                    "尚未用真实标注数据比较过它们和固定阈值在检出率/误检率上的差异。"
                ),
            },
            {
                "item": "parameter_calibration_status",
                "detail": (
                    "当前 thresh、min_area、max_area、max_aspect、open_ksize 等参数"
                    "沿用旧数据集(cama/camb)上摸出的经验值（见 config.yaml 的历史注释），"
                    "尚未针对本批新数据（A/B-Capture_085820）重新标定。"
                ),
            },
            {
                "item": "detection_count_caveat",
                "detail": (
                    f"本次两路共检出 "
                    f"{sum(r.get('n_detections_modular_pipeline', 0) for r in per_camera_summary.values() if r.get('status')=='ok')} "
                    "个候选目标，这是当前(未标定)参数下的检出结果，不等于这批数据里真实的雨滴总数——"
                    "阈值偏保守会漏检，偏松会把噪点当雨滴，两种偏差目前都没有用标注数据量化过。"
                ),
            },
            {
                "item": "equivalence_test_meaning",
                "detail": (
                    "'原Demo与模块化流程结果一致'（per_camera 里的 "
                    "n_frames_identical_to_original_demo）只表示阶段2的模块拆分是纯重构、"
                    "没有意外改变算法行为，不代表检测算法本身的精度（召回率/误检率）已经过验证。"
                    "精度验证需要人工标注的真值数据，不在阶段2范围内。"
                ),
            },
        ],
        "config_swap_note": (
            "阶段2的六个子模块全部通过 rain/registry.py 的注册表 + config/pipeline.yaml 的"
            "detection 段驱动。更换阈值方法(fixed/adaptive_*)、背景模型参数(q/max_frames)"
            "或形态学核尺寸(open_ksize)，只需要改 config/pipeline.yaml，"
            "不需要改动 pipeline/stage2_detection.py、rain/trackers/ 下的任何跟踪代码，"
            "也不改变 detections.csv 的列结构（DETECTION_COLUMNS 定义在 rain/sink.py，"
            "与检测参数无关）。"
        ),
        "known_constraints": [
            "测速只能用 a_device_relative_us / b_device_relative_us，"
            "禁止用文件名毫秒、host 接收时间或帧号/固定帧率。",
            "像素-毫米标定未完成，粒径/面积单位一律是像素。",
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
