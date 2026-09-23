#!/usr/bin/env python3
"""阶段6：雨滴谱（像素）和名义统计 —— 数量浓度 + 像素粒径分布 + 时间窗口统计。

## 用途

复用阶段2检测 + 阶段3跟踪（原地重新跑一遍，不重新实现），对A、B两路
各自独立计算单路调试统计；同时复用阶段4的标定加载逻辑，计算双路共同
体积统计。两条产出线严格分开，互不替代：

    single_camera_proxy   : A、B各自的单路调试统计——检测/跟踪流水线
                             通不通、单路目标量级对不对，看这个。
                             **非正式，不可用于最终浓度**
                             （usable_for_final_concentration恒为False）。
    stereo_common_volume  : 590.4 cm³ 名义体积真正对应的统计——N 只能
                             来自阶段4双路匹配成功的目标数。当前没有
                             标定文件，valid_target_count恒为0，
                             concentration_status恒为"unavailable"
                             （不是"算出来是0"）。

## 明确不做的事（任务书硬性约束）

    - 不把窗口内所有轨迹点相加再除以体积——按轨迹去重，每条轨迹只在它
      首次出现的窗口计数一次。
    - 不产出"正式毫米雨滴谱"——没有像素-毫米标定，只有像素等效直径分布。
    - 不产出雨强——没有可靠物理速度、有效观测体积、毫米粒径，这一步
      没有任何"雨强""rain_rate"字段。
    - 不用A路或B路单独目标数代替双路共同空间目标数。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage6_statistics.py

## 输出（默认写到 config/pipeline.yaml 里 statistics.output_dir 指定的目录）

    window_statistics.csv    single_camera_proxy + stereo_common_volume
                              两类行都在里面，用 category 列区分
    size_distribution.csv    A、B两路各自的像素等效直径分布（仅单路）
    spectrum/{cam}_size_distribution.png   可选的分布图（明确标"非正式雨滴谱"）
    stage6_report.json       模块功能测试引用 + 合成数据结果 + 真实数据名义
                              统计结果（单路/双路分开）+ 当前不能验证的内容
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import rain  # noqa: F401  触发所有实现的注册
from rain.data.metadata_reader import MetadataReader
from rain.data.time_quality_checker import TimeQualityChecker
from rain.data.timing_reader import TimingReader
from rain.io_index import parse_frame_id
from rain.measurement.types import DropSizeDistributionResult, SingleCameraProxyStat, StereoCommonVolumeStat
from rain.registry import (
    build_background,
    build_detector,
    build_reader,
    build_stereo_common_volume_statistics,
    build_track_manager,
    build_window_statistics,
)
from rain.stereo.calibration import CalibrationLoader
from rain.tracking.types import FrameContext, TrackRecord
from rain.types import Detection

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"
_DEVICE_TIME_FIELD = {"a": "a_device_relative_us", "b": "b_device_relative_us"}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段6：雨滴谱(像素)和名义统计")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stats_cfg = cfg["statistics"]
    output_dir = Path(stats_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段6：雨滴谱(像素)和名义统计")
    print("=" * 70)
    print(f"输出目录 : {output_dir}")
    print(f"名义体积 : {stats_cfg['concentration_estimator']['params']['volume_m3']} m^3 "
          "(volume_status=nominal_unvalidated)")

    # --- 1. 时间质量：复用阶段1，标出哪些 pair_index 被判定时间异常 ----------
    # 必须带上 metadata（capture_metadata.json 里实测的 frame_gap_threshold_*、
    # max_abs_relative_drift_us），否则 TimeQualityChecker 会退回"本批中位数*倍数"
    # 的兜底阈值——那套阈值更紧，会把阶段1已经确认正常的漂移值也标成异常，
    # 导致阶段6算出的"时间异常"数量和阶段1的验收结论对不上。
    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    metadata_a = MetadataReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_metadata.json"))
    quality_report = TimeQualityChecker(**cfg["time_quality"]).check(timing_records, metadata=metadata_a)
    flagged_pairs = {qc.pair_index for qc in quality_report.per_pair if qc.flags_text != "ok"}
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}
    print(f"\n阶段1时间质量复用: {len(flagged_pairs)}/{len(timing_records)} 个pair_index被标记时间异常")

    reader = build_reader(cfg["reader"])
    window_stats = build_window_statistics({
        "name": "fixed_length_window_statistics",
        "params": {
            "window_seconds": stats_cfg["window_seconds"],
            "quality_controller": stats_cfg["quality_controller"],
            "concentration_estimator": stats_cfg["concentration_estimator"],
            "size_distribution": stats_cfg["size_distribution"],
            "spatial_calibration_status": stats_cfg["spatial_calibration_status"],
        },
    })

    # --- 2. 双路标定现状：复用阶段4的 CalibrationLoader，不重新判断一遍 -------
    calib_path = cfg["stereo"]["calibration"]["path"]
    calibration = CalibrationLoader().load(calib_path)
    calibration_available = calibration is not None
    print(f"\n双路标定文件: {calib_path}")
    print(f"标定可用    : {calibration_available}"
          + ("" if calibration_available else "  -> stereo_common_volume 全部标记 calibration_unavailable"))

    all_proxy_rows: list[SingleCameraProxyStat] = []
    all_dists: dict[str, DropSizeDistributionResult] = {}
    camera_summaries: dict[str, dict] = {}
    frame_times_a: dict[int, float] | None = None

    for cam in ("a", "b"):
        print(f"\n{'-'*70}\n处理 {cam.upper()} 路（single_camera_proxy）\n{'-'*70}")
        records, frame_times, frame_dets, glue_ms = _detect_and_track_camera(
            cam=cam, folder=cfg["data"][f"dir_{cam}"], cfg=cfg, reader=reader,
            filename_to_pair_index=filename_to_pair_index,
            device_time_us_map={r.pair_index: getattr(r, _DEVICE_TIME_FIELD[cam]) for r in timing_records},
        )
        if cam == "a":
            frame_times_a = frame_times  # A路时间基准，供双路共同体积统计切窗口用
        print(f"  {len(records)} 条轨迹，{sum(len(v) for v in frame_dets.values())} 个原始检测"
              f"（检测+跟踪耗时 {glue_ms:.0f} ms）")

        t0 = time.perf_counter()
        rows, dist = window_stats.compute(cam, records, frame_times, frame_dets, flagged_pairs)
        stats_ms = (time.perf_counter() - t0) * 1000.0

        all_proxy_rows.extend(rows)
        all_dists[cam] = dist
        camera_summaries[cam] = _summarize_camera(cam, rows, dist, stats_ms)
        _print_camera_summary(camera_summaries[cam])

    # --- 3. 双路共同体积统计（590.4cm³真正对应的统计对象）--------------------
    print(f"\n{'-'*70}\nstereo_common_volume（双路共同体积）\n{'-'*70}")
    stereo_stats = build_stereo_common_volume_statistics({
        "name": "gated_stereo_common_volume_statistics",
        "params": {
            "window_seconds": stats_cfg["window_seconds"],
            "concentration_estimator": stats_cfg["concentration_estimator"],
            "spatial_calibration_status": stats_cfg["spatial_calibration_status"],
        },
    })
    t0 = time.perf_counter()
    # calibration_available=False 时 matched_pair_entry_times 必须是空列表——
    # 没有标定就不可能有"双路匹配成功"的目标，不允许伪造任何数据进这个列表。
    # calibration 一旦可用（未来标定完成后），这里需要换成真正调用阶段4
    # StereoMatcher 对每个 pair_index 做匹配、收集 status="matched" 的目标
    # 时间——当前分支未执行到，但接口已经就位，calibration_available=True
    # 时如果传入非空匹配列表，StereoCommonVolumeStatistics 会正常计算。
    matched_pair_entry_times: list[float] = []
    stereo_rows: list[StereoCommonVolumeStat] = stereo_stats.compute(
        calibration_available=calibration_available,
        t_min=min(frame_times_a.values()), t_max=max(frame_times_a.values()),
        matched_pair_entry_times=matched_pair_entry_times,
    )
    stereo_ms = (time.perf_counter() - t0) * 1000.0
    _print_stereo_summary(stereo_rows)

    # --- 输出 CSV --------------------------------------------------------
    window_csv = output_dir / "window_statistics.csv"
    _write_window_statistics_csv(all_proxy_rows, stereo_rows, window_csv)
    print(f"\n{window_csv} 共 {len(all_proxy_rows) + len(stereo_rows)} 行"
          f"（single_camera_proxy={len(all_proxy_rows)}, stereo_common_volume={len(stereo_rows)}）")

    size_csv = output_dir / "size_distribution.csv"
    _write_size_distribution_csv(list(all_dists.values()), size_csv)
    print(f"{size_csv}")

    # --- 可选：像素粒径分布图 ----------------------------------------------
    if stats_cfg.get("plot_spectrum", True):
        spectrum_dir = output_dir / "spectrum"
        spectrum_dir.mkdir(parents=True, exist_ok=True)
        for cam, dist in all_dists.items():
            _save_spectrum_plot(cam, dist, spectrum_dir)
        print(f"{spectrum_dir}/ (每路一张分布图)")

    # --- 合成数据演示（独立于真实数据，证明"轨迹去重不重复计数"等逻辑正确）---
    synthetic_result = _run_synthetic_demo(stats_cfg)

    # --- 报告 -----------------------------------------------------------------
    report_path = output_dir / "stage6_report.json"
    _write_report(
        report_path, stats_cfg, camera_summaries, synthetic_result,
        stereo_rows=stereo_rows, calibration_available=calibration_available,
        calib_path=calib_path, stereo_ms=stereo_ms,
    )
    print(f"{report_path}")

    print()
    print("=" * 70)
    print("阶段6完成。single_camera_proxy=非正式单路调试统计，不可用于最终浓度；")
    print("stereo_common_volume=590.4cm³真正对应的统计，当前因无标定文件"
          "全部为calibration_unavailable。")
    print("没有输出、也不能输出毫米雨滴谱或雨强。")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# 检测 + 跟踪（复用阶段2/阶段3，不重新实现）
# ---------------------------------------------------------------------------


def _detect_and_track_camera(
    cam: str, folder: str, cfg: dict, reader,
    filename_to_pair_index: dict, device_time_us_map: dict,
) -> tuple[list[TrackRecord], dict[int, float], dict[int, list[Detection]], float]:
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]

    frame_files = _list_camera_frame_files(folder)
    n = min(int(det_cfg["n_frames"]), len(frame_files))
    frame_files = frame_files[:n]
    paths = [p for _, p in frame_files]
    frames = [reader.read(p) for p in paths]

    bgmodel = build_background(cfg["background"])
    bg = bgmodel.build(frames)

    detector = build_detector({
        "name": "modular_pipeline",
        "params": {
            "normalizer": det_cfg["normalizer"], "differencer": det_cfg["differencer"],
            "segmenter": det_cfg["segmenter"], "morphology": det_cfg["morphology"],
            "cc_analyzer": det_cfg["cc_analyzer"], "contour_extractor": det_cfg["contour_extractor"],
        },
    })

    t0 = time.perf_counter()
    frame_times: dict[int, float] = {}
    frame_dets: dict[int, list[Detection]] = {}
    frame_contexts: list[FrameContext] = []
    for (fn, _path), img in zip(frame_files, frames):
        if fn not in filename_to_pair_index:
            raise KeyError(f"文件 {fn} 在 capture_timing.csv 里找不到 pair_index，停止处理。")
        pidx = filename_to_pair_index[fn]
        device_time_s = device_time_us_map[pidx] / 1e6
        dets = detector.detect(img, bg, frame_idx=pidx)
        frame_times[pidx] = device_time_s
        frame_dets[pidx] = dets
        frame_contexts.append(FrameContext(pair_index=pidx, device_time_s=device_time_s, detections=dets))
    frame_contexts.sort(key=lambda f: f.pair_index)

    manager = build_track_manager({
        "name": trk_cfg["track_manager"]["name"],
        "params": {
            **trk_cfg["track_manager"]["params"], "camera": cam,
            "kalman_predictor": trk_cfg["kalman_predictor"],
            "track_associator": trk_cfg["track_associator"],
        },
    })
    for fc in frame_contexts:
        manager.step(fc)
    records = manager.finalize()
    ms = (time.perf_counter() - t0) * 1000.0

    return records, frame_times, frame_dets, ms


def _list_camera_frame_files(folder: str) -> list[tuple[str, str]]:
    items = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith((".tif", ".tiff")):
            continue
        fid = parse_frame_id(fn)
        if fid is None:
            continue
        items.append((fid, fn, os.path.join(folder, fn)))
    items.sort(key=lambda t: t[0])
    return [(fn, path) for _, fn, path in items]


# ---------------------------------------------------------------------------
# 汇总 / 输出
# ---------------------------------------------------------------------------


def _summarize_camera(cam: str, rows: list[SingleCameraProxyStat], dist: DropSizeDistributionResult, stats_ms: float) -> dict:
    n_windows = len(rows)
    n_empty = sum(1 for r in rows if r.quality_status == "empty")
    n_low_conf = sum(1 for r in rows if r.quality_status == "low_confidence")
    n_nominal = sum(1 for r in rows if r.quality_status == "nominal")
    total_valid = sum(r.valid_target_count for r in rows)
    total_excluded = sum(r.excluded_target_count for r in rows)
    concentrations = [r.nominal_concentration_m3 for r in rows if r.valid_target_count > 0]

    return {
        "camera": cam,
        "is_single_camera_proxy": True,
        "usable_for_final_concentration": False,
        "n_windows": n_windows,
        "n_windows_empty": n_empty,
        "n_windows_low_confidence": n_low_conf,
        "n_windows_nominal": n_nominal,
        "total_valid_targets": total_valid,
        "total_excluded_targets": total_excluded,
        "concentration_m3_nonzero_windows": {
            "n": len(concentrations),
            "median": (sorted(concentrations)[len(concentrations) // 2] if concentrations else None),
            "min": min(concentrations) if concentrations else None,
            "max": max(concentrations) if concentrations else None,
        },
        "size_distribution_n_total": dist.n_total,
        "size_distribution_bin_edges_px": dist.bin_edges_px,
        "size_distribution_counts": dist.counts,
        "module_ms": stats_ms,
    }


def _print_camera_summary(s: dict) -> None:
    print(f"  窗口数: {s['n_windows']}（空:{s['n_windows_empty']} "
          f"低置信度:{s['n_windows_low_confidence']} 有名义统计:{s['n_windows_nominal']}）")
    print(f"  有效目标(去重后): {s['total_valid_targets']}  被排除目标: {s['total_excluded_targets']}")
    print(f"  像素粒径分布: n={s['size_distribution_n_total']}  counts={s['size_distribution_counts']}")
    print("  !! single_camera_proxy：非正式，不可用于最终浓度 !!")


def _print_stereo_summary(rows: list[StereoCommonVolumeStat]) -> None:
    total_matched = sum(r.n_matched_pairs for r in rows)
    n_unavailable = sum(1 for r in rows if r.concentration_status == "unavailable")
    print(f"  窗口数: {len(rows)}  双路匹配成功目标数合计: {total_matched}  "
          f"concentration_status=unavailable的窗口: {n_unavailable}/{len(rows)}")
    if n_unavailable == len(rows):
        print("  当前无标定文件：全部窗口 stereo_match_status=calibration_unavailable，"
              "concentration_status=unavailable（不是浓度为0，是无法计算）")


def _write_window_statistics_csv(
    proxy_rows: list[SingleCameraProxyStat],
    stereo_rows: list[StereoCommonVolumeStat],
    out_path: Path,
) -> None:
    """single_camera_proxy 和 stereo_common_volume 两类行写进同一份 CSV，
    用 category 列区分——字段并集，某一类不适用的字段留空，不是两边
    互相伪造数据凑成同样的列。
    """
    columns = [
        "category", "camera", "window_index", "window_start_s", "window_end_s",
        "sample_count", "valid_target_count", "excluded_target_count", "n_matched_pairs",
        "size_bin_edges_px", "nominal_concentration_m3", "concentration_status",
        "volume_status", "time_status", "stereo_match_status", "spatial_calibration_status",
        "quality_status", "is_single_camera_proxy", "usable_for_final_concentration",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in proxy_rows:
            w.writerow([
                "single_camera_proxy", r.camera, r.window_index,
                f"{r.window_start_s:.4f}", f"{r.window_end_s:.4f}",
                r.sample_count, r.valid_target_count, r.excluded_target_count, "",
                "|".join(str(e) for e in r.size_bin_edges_px), f"{r.nominal_concentration_m3:.4f}", "",
                r.volume_status, r.time_status, "", r.spatial_calibration_status,
                r.quality_status, int(r.is_single_camera_proxy), int(r.usable_for_final_concentration),
            ])
        for r in stereo_rows:
            usable = r.concentration_status == "computed"
            w.writerow([
                "stereo_common_volume", "", r.window_index,
                f"{r.window_start_s:.4f}", f"{r.window_end_s:.4f}",
                "", r.valid_target_count, "", r.n_matched_pairs,
                "", "" if r.concentration_m3 is None else f"{r.concentration_m3:.4f}", r.concentration_status,
                r.volume_status, "", r.stereo_match_status, r.spatial_calibration_status,
                r.quality_status, 0, int(usable),
            ])


def _write_size_distribution_csv(dists: list[DropSizeDistributionResult], out_path: Path) -> None:
    columns = [
        "camera", "bin_index", "bin_start_px", "bin_end_px", "count",
        "n_total", "spatial_calibration_status", "note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for dist in dists:
            for i, count in enumerate(dist.counts):
                w.writerow([
                    dist.camera, i, dist.bin_edges_px[i], dist.bin_edges_px[i + 1], count,
                    dist.n_total, dist.spatial_calibration_status, dist.note,
                ])


def _save_spectrum_plot(cam: str, dist: DropSizeDistributionResult, spectrum_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    centers = [(a + b) / 2 for a, b in zip(dist.bin_edges_px, dist.bin_edges_px[1:])]
    widths = [b - a for a, b in zip(dist.bin_edges_px, dist.bin_edges_px[1:])]
    ax.bar(centers, dist.counts, width=[w * 0.9 for w in widths], color="tab:blue", edgecolor="black")
    ax.set_xlabel("pixel-equivalent diameter d_eq_px (NOT calibrated mm)")
    ax.set_ylabel("count")
    ax.set_title(f"camera={cam.upper()}  pixel size distribution (n={dist.n_total})\n"
                 "NOT an official drop-size spectrum -- no px-to-mm calibration")
    fig.tight_layout()
    fig.savefig(spectrum_dir / f"{cam}_size_distribution.png", dpi=120)
    plt.close(fig)


def _run_synthetic_demo(stats_cfg: dict) -> dict:
    """独立于真实数据的合成场景：验证"多帧轨迹只计一次""低质量目标被排除"
    等核心逻辑，不依赖真实检测/跟踪结果。"""
    from rain.measurement.window_statistics import WindowStatistics
    from rain.tracking.types import TrackPoint, TrackRecord
    from rain.types import QC_MERGED, QC_OK

    def det(cx, cy, area=50.0, qc_flags=QC_OK):
        return Detection(frame_idx=0, cx=cx, cy=cy, area_px=area,
                          bbox=(int(cx - 5), int(cy - 5), 10, 10),
                          d_eq_px=Detection.d_eq_from_area(area), qc_flags=qc_flags)

    def pt(track_id, pidx, t, d):
        return TrackPoint(track_id=track_id, camera="synthetic", pair_index=pidx, device_time_s=t,
                           detected=d, predicted_cx=d.cx, predicted_cy=d.cy,
                           association_method="gated_nn", quality_flags=d.qc_flags)

    ws = WindowStatistics(
        window_seconds=stats_cfg["window_seconds"],
        quality_controller=stats_cfg["quality_controller"],
        concentration_estimator=stats_cfg["concentration_estimator"],
        size_distribution=stats_cfg["size_distribution"],
    )

    # 场景A：一条5帧轨迹（同一颗雨滴），必须只算1个目标
    dets_a = [det(100, 100 + i * 10) for i in range(5)]
    track_a = TrackRecord(track_id=1, camera="synthetic",
                           points=[pt(1, i, i * 0.01, dets_a[i]) for i in range(5)])
    frame_times_a = {i: i * 0.01 for i in range(5)}
    frame_dets_a = {i: [dets_a[i]] for i in range(5)}
    rows_a, dist_a = ws.compute("synthetic", [track_a], frame_times_a, frame_dets_a, set())

    # 场景B：一个高质量+一个低质量(粘连)目标，低质量必须被排除但不丢失
    good = det(50, 50, area=30, qc_flags=QC_OK)
    bad = det(500, 500, area=80, qc_flags=QC_MERGED)
    track_good = TrackRecord(track_id=1, camera="synthetic", points=[pt(1, 0, 0.0, good)])
    track_bad = TrackRecord(track_id=2, camera="synthetic", points=[pt(2, 0, 0.0, bad)])
    rows_b, dist_b = ws.compute("synthetic", [track_good, track_bad], {0: 0.0}, {0: [good, bad]}, set())

    # 场景C：stereo_common_volume 在"无标定"和"有标定但0个匹配"下的区别
    from rain.measurement.stereo_common_volume_statistics import StereoCommonVolumeStatistics

    stereo = StereoCommonVolumeStatistics(
        window_seconds=stats_cfg["window_seconds"],
        concentration_estimator=stats_cfg["concentration_estimator"],
    )
    rows_no_calib = stereo.compute(calibration_available=False, t_min=0.0, t_max=0.05, matched_pair_entry_times=[])
    rows_with_calib = stereo.compute(calibration_available=True, t_min=0.0, t_max=0.05, matched_pair_entry_times=[0.02])

    return {
        "note": "合成数据，不依赖真实检测/跟踪结果，用于独立验证统计逻辑本身正确。",
        "scenario_multi_frame_track_counted_once": {
            "n_track_points": 5,
            "valid_target_count_summed_across_windows": sum(r.valid_target_count for r in rows_a),
            "expected": 1,
            "size_distribution_n_total": dist_a.n_total,
        },
        "scenario_low_quality_excluded_but_retained": {
            "valid_target_count": rows_b[0].valid_target_count,
            "excluded_target_count": rows_b[0].excluded_target_count,
            "expected_valid": 1,
            "expected_excluded": 1,
        },
        "scenario_stereo_common_volume_unavailable_vs_computed": {
            "no_calibration": {
                "n_matched_pairs": rows_no_calib[0].n_matched_pairs,
                "concentration_status": rows_no_calib[0].concentration_status,
                "concentration_m3": rows_no_calib[0].concentration_m3,
            },
            "with_calibration_and_one_match": {
                "n_matched_pairs": rows_with_calib[0].n_matched_pairs,
                "concentration_status": rows_with_calib[0].concentration_status,
                "concentration_m3": rows_with_calib[0].concentration_m3,
            },
            "expected": "无标定时 n=0/status=unavailable/concentration=None；"
                        "有标定+1个匹配时 n=1/status=computed/concentration=一个具体数值。",
        },
    }


def _write_report(
    report_path: Path, stats_cfg: dict, camera_summaries: dict, synthetic_result: dict,
    stereo_rows: list[StereoCommonVolumeStat], calibration_available: bool,
    calib_path: str, stereo_ms: float,
) -> None:
    total_matched = sum(r.n_matched_pairs for r in stereo_rows)
    stereo_summary = {
        "calibration_path_configured": calib_path,
        "calibration_available": calibration_available,
        "is_stereo_common_volume": True,
        "usable_for_final_concentration": calibration_available,
        "n_windows": len(stereo_rows),
        "total_matched_pairs": total_matched,
        "n_windows_concentration_unavailable": sum(1 for r in stereo_rows if r.concentration_status == "unavailable"),
        "n_windows_concentration_computed": sum(1 for r in stereo_rows if r.concentration_status == "computed"),
        "module_ms": stereo_ms,
        "note": (
            "这才是590.4cm^3名义体积真正对应的统计对象。当前没有双相机标定"
            "文件，n_matched_pairs和concentration恒为不可用状态——不是"
            "'算出来是0'，是'现在没法算'，不得用A路或B路单独目标数替代。"
        ) if not calibration_available else "标定可用，浓度已按双路匹配成功的目标数计算。",
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_test_results": {
            "test_file": "tests/test_statistics_stage6.py",
            "how_to_reproduce": "python -m pytest tests/test_statistics_stage6.py -v",
            "note": "覆盖场景：空窗口、单目标、重复轨迹不重复计数、低质量目标排除、"
                    "体积单位换算、像素粒径统计、未标定时字段结构性检查(不含mm/雨强字段)、"
                    "single_camera_proxy非正式标记、stereo_common_volume无标定时的安全处理。",
        },
        "synthetic_data_results": synthetic_result,
        "real_data_results_180_pairs": {
            "single_camera_proxy": {
                "per_camera": camera_summaries,
                "is_informal": True,
                "usable_for_final_concentration": False,
                "note": (
                    "A/B-Capture_085820，180帧，A、B两路各自独立的调试统计。"
                    "590.4 cm^3 是双路共同观测体积，不是单路视场体积，这里的"
                    "'浓度'数值只用于核对流水线通不通、单路目标量级对不对，"
                    "不能代表真实降雨浓度，不得用于最终结果。"
                ),
            },
            "stereo_common_volume": stereo_summary,
            "note": (
                "single_camera_proxy 和 stereo_common_volume 是两条完全独立的"
                "统计产出线，不能互相替代、不能相加、不能用其中一路凑数补齐"
                "另一路缺失的数据。"
            ),
        },
        "cannot_currently_validate": [
            "590.4 cm^3 共同观测体积是否等于两路相机实际共同视场——需要实验标定"
            "（比如已知尺寸标定物体或双路同步光学事件），当前完全没有做这一步。",
            "双路共同体积统计(stereo_common_volume)本身——依赖阶段4的双路匹配，"
            "当前无标定文件，n_matched_pairs和concentration恒为不可用，"
            "这条产出线目前只有'结构正确'，没有真实数值可验证。",
            "像素等效直径到毫米粒径的换算——没有像素-毫米标定，所有粒径数字"
            "永远停留在像素单位，无法验证是否对应真实的雨滴尺寸范围。",
            "single_camera_proxy 数量级和 stereo_common_volume 数量级之间的"
            "关系——理论上后者应该小于等于前两者中较小的一个（双路都确认的"
            "目标不会比任一单路看到的多），但当前后者恒为0，无法验证这个关系。",
            "数量浓度和粒径分布在真实降雨强度下的合理性——本批数据检出稀疏"
            "（继承自阶段2/3未标定的检测/跟踪参数），样本量太小，无法判断"
            "这套统计流程在中到大雨强度下是否还成立。",
            "时间窗口长度(window_seconds)是否是统计上合理的选择——目前是"
            "工程经验值，没有用真实降雨事件的时间尺度做过验证。",
            "雨强(rain rate)——需要同时具备可靠物理速度(阶段5当前只有1条"
            "低置信度轨迹)、有效观测体积(未标定)、毫米粒径(未标定)三者，"
            "本阶段一个都不满足，因此完全没有计算，也没有任何相关字段。",
        ],
        "known_constraints": [
            "590.4 cm^3 / 0.0005904 m^3 是双路共同观测体积的名义值，不是单路"
            "视场体积。single_camera_proxy 用这个体积算出的'浓度'仅供单路"
            "调试参考，不可用于最终浓度（usable_for_final_concentration恒为False）。",
            "stereo_common_volume 才是这个体积真正对应的统计对象，N 必须来自"
            "阶段4双路匹配成功的目标数，不得用A路或B路单独目标数代替；"
            "当前无标定文件，valid_target_count恒为0，concentration_status"
            "恒为'unavailable'（不是'浓度为0'）。",
            "volume_status 和 spatial_calibration_status 都从"
            "config/pipeline.yaml 的 statistics 段读取，不是代码里的永久"
            "字符串常量——真正完成标定后，改配置文件即可反映新状态。",
            "没有像素-毫米标定，粒径分布只能是像素等效直径分布，不得称为"
            "正式雨滴谱；没有可靠物理速度+有效体积+毫米粒径，不输出雨强。",
            "浓度按轨迹去重计数（每条轨迹只在首次出现的窗口计一次），"
            "不是把窗口内所有轨迹点相加。",
        ],
        "statistics_config_used": {
            "window_seconds": stats_cfg["window_seconds"],
            "volume_m3": stats_cfg["concentration_estimator"]["params"]["volume_m3"],
            "volume_status": stats_cfg["concentration_estimator"]["params"]["volume_status"],
            "spatial_calibration_status": stats_cfg["spatial_calibration_status"],
            "bin_edges_px": stats_cfg["size_distribution"]["params"]["bin_edges_px"],
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
