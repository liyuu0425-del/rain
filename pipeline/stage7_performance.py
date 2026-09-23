#!/usr/bin/env python3
"""阶段7：性能测试 —— 完整流水线（阶段1读取~阶段6统计）在A/B真实数据上的
耗时、吞吐、资源占用。

## 用途

对A、B两路各180帧真实数据，端到端跑一遍"读取->检测->跟踪->双路匹配->
测速->统计"，逐阶段计时，并测整体吞吐、CPU占用、内存峰值、磁盘读写。
复用阶段1~6已经实现、已经验收过的模块（rain.data / rain.detection /
rain.tracking / rain.stereo / rain.measurement），不重新实现任何算法——
这个脚本只是给已有流程加计时和资源监控。

## 明确区分的四件事（任务书要求）

    1. 功能正确性   : 见 tests/ 全量测试（本脚本不重复跑pytest，只引用）。
    2. 短批性能     : 本脚本主体——180对数据单次处理+重复读取压力测试。
    3. 长期实时性能 : **当前180对数据只能证明短批处理性能，不能证明
                       长时间连续运行的稳定性**（内存泄漏、句柄泄漏、
                       降频这类问题需要几小时/几天连续运行才会暴露，
                       几秒钟跑完的180对数据测不出来）。
    4. 物理标定现状 : 双相机内外参、固定曝光延迟、像素-毫米、共同观测
                       体积均未标定——这与性能无关，但阶段7的报告仍要
                       如实重申，不能让"性能测试通过"掩盖"功能仍是名义
                       结果"这件事。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage7_performance.py
    python pipeline\\stage7_performance.py --repeats 5

## 输出（默认写到 config/pipeline.yaml 里 performance.output_dir 指定的目录）

    performance_report.csv   每个阶段的平均/中位/P95/最大耗时 + 调用次数
                              （调试图开启/关闭各一份，用 run 列区分）
    stage7_report.json       吞吐/CPU/内存/磁盘/压力测试/四项区分，一次说清
    debug_images/            调试图开启那一轮实际存下来的叠加图（证据）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

import rain  # noqa: F401  触发所有实现的注册
from rain.data.metadata_reader import MetadataReader
from rain.data.time_quality_checker import TimeQualityChecker
from rain.data.timing_reader import TimingReader
from rain.imageio import imwrite_unicode
from rain.io_index import parse_frame_id
from rain.perf.profiler import StageProfiler
from rain.perf.resource_monitor import ResourceMonitor
from rain.registry import (
    build_background,
    build_detector,
    build_reader,
    build_stereo_common_volume_statistics,
    build_stereo_matcher,
    build_track_manager,
    build_velocity_estimator,
    build_window_statistics,
)
from rain.sink import draw_overlay
from rain.stereo.calibration import CalibrationLoader
from rain.tracking.types import FrameContext

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"
_DEVICE_TIME_FIELD = {"a": "a_device_relative_us", "b": "b_device_relative_us"}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段7：性能测试")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    ap.add_argument("--repeats", type=int, default=None, help="重复读取压力测试跑几轮（覆盖配置文件）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    perf_cfg = cfg["performance"]
    output_dir = Path(perf_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    n_repeats = args.repeats if args.repeats is not None else int(perf_cfg["n_repeats"])
    sample_interval = float(perf_cfg["resource_sample_interval_s"])
    nominal_pps = float(perf_cfg["nominal_input_rate_pps"])
    target_pps = float(perf_cfg["target_rate_with_margin_pps"])

    print("=" * 70)
    print("阶段7：性能测试")
    print("=" * 70)
    print(f"输出目录       : {output_dir}")
    print(f"运行环境       : {platform.system()} {platform.release()}  Python {platform.python_version()}")
    print(f"输入基准       : {nominal_pps} 对/秒（建议目标 {target_pps} 对/秒，留30%余量）")

    # --- 0. 一次性加载共用资源 -----------------------------------------------
    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    metadata_a = MetadataReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_metadata.json"))
    quality_report = TimeQualityChecker(**cfg["time_quality"]).check(timing_records, metadata=metadata_a)
    flagged_pairs = {qc.pair_index for qc in quality_report.per_pair if qc.flags_text != "ok"}
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}
    calibration = CalibrationLoader().load(cfg["stereo"]["calibration"]["path"])
    n_pairs_total = len(timing_records)

    device_time_us_by_field = {
        "a_device_relative_us": {r.pair_index: r.a_device_relative_us for r in timing_records},
        "b_device_relative_us": {r.pair_index: r.b_device_relative_us for r in timing_records},
    }

    # --- 1. 单次180对处理，调试图关闭 -----------------------------------------
    print(f"\n[1/4] 单次{n_pairs_total}对数据处理（调试图关闭）...")
    profiler_off = StageProfiler()
    mon = ResourceMonitor(interval_s=sample_interval)
    mon.start()
    t0 = time.perf_counter()
    summary_off = _run_full_pipeline(
        cfg, profiler_off, timing_records, filename_to_pair_index, flagged_pairs,
        calibration, device_time_us_by_field, save_debug_images=False, debug_dir=None,
    )
    wall_off = time.perf_counter() - t0
    res_off = mon.stop()
    run_off = _build_run_record("debug_off", wall_off, n_pairs_total, res_off, summary_off)
    _print_run_record(run_off)

    # --- 2. 单次180对处理，调试图开启 -----------------------------------------
    print(f"\n[2/4] 单次{n_pairs_total}对数据处理（调试图开启）...")
    debug_dir = output_dir / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)
    profiler_on = StageProfiler()
    mon = ResourceMonitor(interval_s=sample_interval)
    mon.start()
    t0 = time.perf_counter()
    summary_on = _run_full_pipeline(
        cfg, profiler_on, timing_records, filename_to_pair_index, flagged_pairs,
        calibration, device_time_us_by_field, save_debug_images=True, debug_dir=debug_dir,
    )
    wall_on = time.perf_counter() - t0
    res_on = mon.stop()
    run_on = _build_run_record("debug_on", wall_on, n_pairs_total, res_on, summary_on)
    _print_run_record(run_on)

    debug_overhead_s = wall_on - wall_off
    print(f"\n  调试图开销: +{debug_overhead_s:.2f}s "
          f"({100*debug_overhead_s/wall_off:.1f}%)，"
          f"实际存下 {2*n_pairs_total} 张叠加图到 {debug_dir}")

    # --- 3. 重复读取压力测试（调试图关闭）-------------------------------------
    print(f"\n[3/4] 重复读取压力测试（{n_repeats}轮，调试图关闭）...")
    repeat_profiler = StageProfiler()
    repeat_runs = []
    for i in range(n_repeats):
        mon = ResourceMonitor(interval_s=sample_interval)
        mon.start()
        t0 = time.perf_counter()
        summary = _run_full_pipeline(
            cfg, repeat_profiler, timing_records, filename_to_pair_index, flagged_pairs,
            calibration, device_time_us_by_field, save_debug_images=False, debug_dir=None,
        )
        wall = time.perf_counter() - t0
        res = mon.stop()
        run = _build_run_record(f"repeat_{i}", wall, n_pairs_total, res, summary)
        repeat_runs.append(run)
        print(f"  第{i+1}轮: {wall:.2f}s  {run['throughput_pairs_per_s']:.1f}对/秒  "
              f"峰值内存{res['peak_rss_mb']:.1f}MB  CPU{res['cpu_percent']:.0f}%")

    trend = _analyze_repeat_trend(repeat_runs)
    print(f"  重复轮次耗时趋势: {trend['note']}")

    # --- 4. 输出 -----------------------------------------------------------
    print("\n[4/4] 写输出文件 ...")
    perf_csv = output_dir / "performance_report.csv"
    _write_performance_csv(
        {"debug_off": profiler_off, "debug_on": profiler_on, "repeated_stress": repeat_profiler},
        perf_csv,
    )
    print(f"  {perf_csv}")

    report_path = output_dir / "stage7_report.json"
    _write_report(
        report_path=report_path,
        n_pairs_total=n_pairs_total,
        nominal_pps=nominal_pps, target_pps=target_pps,
        run_off=run_off, run_on=run_on, debug_overhead_s=debug_overhead_s,
        repeat_runs=repeat_runs, trend=trend,
        calibration_available=calibration is not None,
        n_repeats=n_repeats,
        bottleneck_analysis=_analyze_bottleneck(profiler_off, n_pairs_total),
    )
    print(f"  {report_path}")

    print()
    print("=" * 70)
    meets_nominal = run_off["throughput_pairs_per_s"] >= nominal_pps
    meets_target = run_off["throughput_pairs_per_s"] >= target_pps
    print(f"端到端吞吐 {run_off['throughput_pairs_per_s']:.1f} 对/秒 "
          f"(基准{nominal_pps}: {'达标' if meets_nominal else '未达标'}；"
          f"目标{target_pps}: {'达标' if meets_target else '未达标'})")
    print("重要提醒：以上只证明180对短批处理性能，不能证明长时间连续运行")
    print("的稳定性；物理标定（双相机内外参/固定曝光延迟/像素-毫米/观测体积）")
    print("仍未完成，与性能测试无关，但结果依然是名义值。")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# 完整流水线（复用阶段1~6已有模块，只加计时，不重新实现算法）
# ---------------------------------------------------------------------------


def _run_full_pipeline(
    cfg: dict,
    profiler: StageProfiler,
    timing_records,
    filename_to_pair_index: dict,
    flagged_pairs: set,
    calibration,
    device_time_us_by_field: dict,
    save_debug_images: bool,
    debug_dir: Path | None,
) -> dict:
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]
    stats_cfg = cfg["statistics"]
    reader = build_reader(cfg["reader"])

    per_camera_records = {}
    per_camera_frame_times = {}
    per_camera_frame_dets = {}

    for cam in ("a", "b"):
        folder = cfg["data"][f"dir_{cam}"]
        frame_files = _list_camera_frame_files(folder)
        n = min(int(det_cfg["n_frames"]), len(frame_files))
        frame_files = frame_files[:n]

        frames = []
        for fn, path in frame_files:
            with profiler.timer("read"):
                img = reader.read(path)
            frames.append(img)

        with profiler.timer("background"):
            bg = build_background(cfg["background"]).build(frames)

        detector = build_detector({
            "name": "modular_pipeline",
            "params": {
                "normalizer": det_cfg["normalizer"], "differencer": det_cfg["differencer"],
                "segmenter": det_cfg["segmenter"], "morphology": det_cfg["morphology"],
                "cc_analyzer": det_cfg["cc_analyzer"], "contour_extractor": det_cfg["contour_extractor"],
            },
        })

        frame_times: dict[int, float] = {}
        frame_dets: dict[int, list] = {}
        frame_contexts: list[FrameContext] = []
        device_time_map = device_time_us_by_field[_DEVICE_TIME_FIELD[cam]]

        for (fn, _path), img in zip(frame_files, frames):
            pidx = filename_to_pair_index[fn]
            device_time_s = device_time_map[pidx] / 1e6

            with profiler.timer("detect"):
                dets = detector.detect(img, bg, frame_idx=pidx)

            frame_times[pidx] = device_time_s
            frame_dets[pidx] = dets
            frame_contexts.append(FrameContext(pair_index=pidx, device_time_s=device_time_s, detections=dets))

            if save_debug_images:
                with profiler.timer("debug_save"):
                    overlay = draw_overlay(img, dets)
                    imwrite_unicode(str(debug_dir / f"{cam}_pair{pidx:04d}.png"), overlay)

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
            step_result = manager.step(fc)
            profiler.record("predict", step_result.predict_ms)
            profiler.record("associate", step_result.associate_ms)
            profiler.record("manage", step_result.manage_ms)
        records = manager.finalize()

        per_camera_records[cam] = records
        per_camera_frame_times[cam] = frame_times
        per_camera_frame_dets[cam] = frame_dets

    # --- 双路匹配（复用阶段4；当前无标定，match_frame每次都会走
    #     calibration_unavailable分支，但耗时依然真实计入"stereo_match"）------
    matcher = build_stereo_matcher({
        "name": "gated_cost_stereo_matcher",
        "params": {
            "epipolar_gate": cfg["stereo"]["epipolar_gate"],
            "match_gate": cfg["stereo"]["match_gate"],
            "cost_matrix_builder": cfg["stereo"]["cost_matrix_builder"],
            "max_total_cost": cfg["stereo"]["max_total_cost"],
            "fixed_exposure_delay_calibrated": cfg["stereo"]["fixed_exposure_delay_calibrated"],
            "calibration": calibration,
        },
    })
    all_pair_indices = sorted(set(per_camera_frame_dets["a"]) | set(per_camera_frame_dets["b"]))
    n_stereo_matched = 0
    for pidx in all_pair_indices:
        dets_a = per_camera_frame_dets["a"].get(pidx, [])
        dets_b = per_camera_frame_dets["b"].get(pidx, [])
        with profiler.timer("stereo_match"):
            result = matcher.match_frame(pidx, dets_a, dets_b)
        n_stereo_matched += sum(1 for r in result.rows if r.status == "matched")

    # --- 测速（复用阶段5）----------------------------------------------------
    velocity_estimator = build_velocity_estimator(cfg["measurement"]["linear_velocity_estimator"])
    n_velocity_fits = 0
    for cam in ("a", "b"):
        for rec in per_camera_records[cam]:
            with profiler.timer("velocity_fit"):
                velocity_estimator.estimate(rec)
            n_velocity_fits += 1

    # --- 统计（复用阶段6）-----------------------------------------------------
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
    for cam in ("a", "b"):
        with profiler.timer("statistics_single_camera"):
            window_stats.compute(
                cam, per_camera_records[cam], per_camera_frame_times[cam],
                per_camera_frame_dets[cam], flagged_pairs,
            )

    stereo_common_stats = build_stereo_common_volume_statistics({
        "name": "gated_stereo_common_volume_statistics",
        "params": {
            "window_seconds": stats_cfg["window_seconds"],
            "concentration_estimator": stats_cfg["concentration_estimator"],
            "spatial_calibration_status": stats_cfg["spatial_calibration_status"],
        },
    })
    with profiler.timer("statistics_stereo_common"):
        stereo_common_stats.compute(
            calibration_available=calibration is not None,
            t_min=min(per_camera_frame_times["a"].values()),
            t_max=max(per_camera_frame_times["a"].values()),
            matched_pair_entry_times=[],
        )

    return {
        "n_pairs": len(all_pair_indices),
        "n_detections_a": sum(len(v) for v in per_camera_frame_dets["a"].values()),
        "n_detections_b": sum(len(v) for v in per_camera_frame_dets["b"].values()),
        "n_tracks_a": len(per_camera_records["a"]),
        "n_tracks_b": len(per_camera_records["b"]),
        "n_stereo_matched": n_stereo_matched,
        "n_velocity_fits": n_velocity_fits,
    }


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
# 输出
# ---------------------------------------------------------------------------


def _build_run_record(name: str, wall_s: float, n_pairs: int, resource: dict, summary: dict) -> dict:
    return {
        "run": name,
        "wall_s": wall_s,
        "throughput_pairs_per_s": n_pairs / wall_s if wall_s > 0 else 0.0,
        "peak_rss_mb": resource["peak_rss_mb"],
        "cpu_percent": resource["cpu_percent"],
        "disk_io": resource["disk_io"],
        **summary,
    }


def _print_run_record(run: dict) -> None:
    print(f"  耗时 {run['wall_s']:.2f}s  吞吐 {run['throughput_pairs_per_s']:.1f}对/秒  "
          f"峰值内存 {run['peak_rss_mb']:.1f}MB  CPU {run['cpu_percent']:.0f}%")
    if run["disk_io"] is not None:
        print(f"  磁盘读 {run['disk_io']['read_bytes']/1e6:.1f}MB  "
              f"磁盘写 {run['disk_io']['write_bytes']/1e6:.1f}MB")
    print(f"  A检测{run['n_detections_a']} B检测{run['n_detections_b']}  "
          f"A轨迹{run['n_tracks_a']} B轨迹{run['n_tracks_b']}  "
          f"双路匹配成功{run['n_stereo_matched']}  测速调用{run['n_velocity_fits']}次")


def _analyze_repeat_trend(repeat_runs: list[dict]) -> dict:
    walls = [r["wall_s"] for r in repeat_runs]
    peaks = [r["peak_rss_mb"] for r in repeat_runs]
    if len(walls) < 2:
        return {"note": "重复轮次少于2轮，无法判断趋势", "wall_s_first": walls[0] if walls else None,
                "wall_s_last": walls[-1] if walls else None}

    wall_drift_pct = 100 * (walls[-1] - walls[0]) / walls[0] if walls[0] > 0 else 0.0
    mem_drift_mb = peaks[-1] - peaks[0]

    if abs(wall_drift_pct) < 15 and mem_drift_mb < 20:
        note = f"耗时波动{wall_drift_pct:+.1f}%，内存变化{mem_drift_mb:+.1f}MB，{len(walls)}轮内无明显退化迹象"
    else:
        note = (f"耗时波动{wall_drift_pct:+.1f}%，内存变化{mem_drift_mb:+.1f}MB，"
                f"幅度较大——但仅{len(walls)}轮、几秒钟量级，不足以判断是否为真实性能退化"
                f"（还是系统负载波动），需要更长时间连续运行才能下结论。")
    return {
        "note": note, "wall_s_first": walls[0], "wall_s_last": walls[-1],
        "wall_s_drift_percent": wall_drift_pct, "peak_rss_mb_drift": mem_drift_mb,
    }


def _analyze_bottleneck(profiler: StageProfiler, n_pairs_total: int) -> dict:
    """从实测的各阶段耗时里找出瓶颈，数据驱动地解释"吞吐为什么是这个数"，
    而不是只甩一个数字不给原因。"""
    stats = profiler.stats()
    # 按"总耗时占比"排序，total_ms 已经是"调用次数 * 平均每次耗时"的累计值，
    # 比只看 mean_ms 更能反映"这一步吃掉了整个流水线百分之多少的时间"。
    ranked = sorted(stats.items(), key=lambda kv: kv[1].total_ms, reverse=True)
    total_ms_all = sum(s.total_ms for _, s in ranked)

    breakdown = [
        {
            "stage": stage,
            "total_ms": s.total_ms,
            "percent_of_total": 100 * s.total_ms / total_ms_all if total_ms_all > 0 else 0.0,
            "mean_ms_per_call": s.mean_ms,
            "n_calls": s.n_calls,
        }
        for stage, s in ranked
    ]

    top = breakdown[0] if breakdown else None
    note = ""
    if top is not None:
        note = (
            f"耗时占比最高的阶段是 '{top['stage']}'，占端到端总耗时的 "
            f"{top['percent_of_total']:.1f}%（平均每次调用 {top['mean_ms_per_call']:.2f}ms，"
            f"共调用 {top['n_calls']} 次）。"
        )
        if top["stage"] == "read":
            note += (
                " 这和老 Demo 已经记录过的实测结论一致（LZW压缩TIF解码约21ms/帧，"
                "占精简流水线约70%成本）——当前 config/pipeline.yaml 的 reader 用的"
                "是 lzw_tif；README 里提到的 raw_bin（未压缩裸二进制，约2ms/帧）"
                "理论上能大幅降低这一项，但需要数据本身以raw格式提供，"
                "这批数据是LZW TIF，不能凭空转换。另外当前流水线是单线程顺序执行，"
                "老 Demo 的 runners/run_batch.py 用线程池并行处理多批数据来提高吞吐，"
                "这次阶段1-7的新模块化流水线没有实现那层并发——这是达到"
                f"{312.5:.1f}对/秒目标的两个明确可优化方向，但属于后续工程优化"
                "范畴，不在阶段7的验收范围内。"
            )
    return {"ranked_by_total_time": breakdown, "note": note}


def _write_performance_csv(profilers: dict[str, StageProfiler], out_path: Path) -> None:
    columns = ["run", "stage", "n_calls", "mean_ms", "median_ms", "p95_ms", "max_ms", "total_ms"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for run_name, profiler in profilers.items():
            for stage, s in sorted(profiler.stats().items()):
                w.writerow([
                    run_name, stage, s.n_calls,
                    f"{s.mean_ms:.4f}", f"{s.median_ms:.4f}", f"{s.p95_ms:.4f}",
                    f"{s.max_ms:.4f}", f"{s.total_ms:.4f}",
                ])


def _write_report(
    report_path: Path, n_pairs_total: int, nominal_pps: float, target_pps: float,
    run_off: dict, run_on: dict, debug_overhead_s: float,
    repeat_runs: list[dict], trend: dict, calibration_available: bool, n_repeats: int,
    bottleneck_analysis: dict,
) -> None:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "python_version": platform.python_version(),
        },

        # --- 1. 功能正确性：本脚本不重复跑pytest，只引用 -----------------------
        "functional_correctness": {
            "note": (
                "阶段7本身不重新验证功能正确性——功能正确性由 tests/ 下的全量"
                "单元测试保证，运行 `python -m pytest tests/ -q` 查看最新结果。"
                "阶段7只测已经验收通过的功能在真实数据上跑起来多快、占多少资源。"
            ),
            "how_to_reproduce": "python -m pytest tests/ -q",
        },

        # --- 2. 短批性能：本脚本的主体结果 -------------------------------------
        "short_batch_performance": {
            "n_pairs": n_pairs_total,
            "debug_images_off": run_off,
            "debug_images_on": run_on,
            "debug_image_overhead": {
                "extra_wall_s": debug_overhead_s,
                "extra_percent": 100 * debug_overhead_s / run_off["wall_s"] if run_off["wall_s"] > 0 else None,
                "n_images_saved": 2 * n_pairs_total,
            },
            "throughput_vs_benchmark": {
                "nominal_input_rate_pps": nominal_pps,
                "target_rate_with_margin_pps": target_pps,
                "measured_end_to_end_pps": run_off["throughput_pairs_per_s"],
                "meets_nominal": run_off["throughput_pairs_per_s"] >= nominal_pps,
                "meets_target_with_margin": run_off["throughput_pairs_per_s"] >= target_pps,
                "note": (
                    "这里的吞吐是端到端(读取->检测->跟踪->双路匹配->测速->统计)"
                    "180对数据处理完的总吞吐，不是只测检测这一步；"
                    "双路匹配当前因无标定几乎不耗时(每个候选都在极线门控就被"
                    "拒绝)，标定完成后这一步的真实耗时会重新测。"
                ),
            },
            "bottleneck_analysis": bottleneck_analysis,
        },

        # --- 3. 长期实时性能：明确说明当前测不出来 -----------------------------
        "long_term_realtime_performance": {
            "tested": False,
            "note": (
                "当前只有180对数据（约0.58秒的原始视频时长），单次运行几秒钟"
                "就能跑完，**不能证明长时间连续运行的稳定性**。内存泄漏、"
                "句柄/文件描述符泄漏、GC暂停累积、磁盘空间耗尽、硬件降频这类"
                "问题通常需要几小时到几天的连续负载才会暴露，本次重复压力"
                "测试只是同一批180对数据重复跑几轮（用于发现明显的、短期内"
                "就能复现的退化），不能替代长时间稳定性测试。"
            ),
            "repeated_stress_test": {
                "n_repeats": n_repeats,
                "per_repeat": repeat_runs,
                "trend_analysis": trend,
            },
            "recommendation_for_future_work": (
                "如果需要验证长期稳定性，应该用真实连续采集数据（不是重复"
                "读同一批180对）跑至少数小时，同时监控内存/句柄数随时间的"
                "变化曲线，而不是只看几轮重复的总耗时。"
            ),
        },

        # --- 4. 物理标定现状：与性能无关，但如实重申 ----------------------------
        "physical_calibration_not_done": [
            "双相机内外参/基础矩阵标定——阶段4已确认，无标定文件，双路匹配"
            "0个已验证匹配，performance测试里的stereo_match耗时因此不代表"
            "标定完成后真实几何计算的耗时（当前每次调用都在极线门控最早"
            "退出）。",
            "固定曝光延迟标定——A/B设备时钟零点独立，双路对应关系仍需人工确认。",
            "像素-毫米标定——阶段5/6的速度和粒径全部停留在像素单位。",
            "共同观测体积(590.4cm^3)标定——阶段6已修正，stereo_common_volume"
            "在无标定时恒为calibration_unavailable。",
            "以上四项均与本阶段的性能测试结果无关——即使吞吐达标，功能层面"
            "仍然只能输出名义/像素结果，不构成物理量的正式测量。",
        ],

        # --- 队列积压：当前架构不适用 -------------------------------------------
        "queue_backlog": {
            "applicable": False,
            "note": (
                "当前流水线（pipeline/stage1~7*.py）是同步单线程执行，"
                "逐相机、逐帧顺序调用，没有实现生产者-消费者队列或线程池"
                "（老 Demo 的 runners/run_batch.py 有 concurrency.max_workers/"
                "queue_size 配置，但那是老数据布局的批处理管线，服务于"
                "tools/demo.py，本次阶段1-7的新流水线没有复用那套并发机制）。"
                "'是否出现队列积压'这个问题目前没有对象可测——如果未来要做"
                "实时并发处理（比如按 runners/run_batch.py 的线程池模式重构），"
                "需要重新设计并测量队列深度/积压情况。"
            ),
        },

        "known_constraints": [
            "180对数据（约0.58秒原始时长）只能验证短批功能和短批性能，"
            "不能验证长期稳定性。",
            "吞吐基准: 312.5对/秒（对应实测中位帧周期3200us）；"
            "建议目标: 406.25对/秒（留30%余量）。",
            "调试图保存(overlay写盘)有实测开销，已分别测量开启/关闭两种情况。",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
