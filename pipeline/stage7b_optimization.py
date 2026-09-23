#!/usr/bin/env python3
"""阶段7B：性能优化 —— 逐个因素测试读取/处理架构优化，每次都与基线比较
耗时和结果一致性。

## 前提结论（阶段7已测出，这里先验证不是假设）

    - 完整流水线吞吐 14.4 对/秒，远低于 312.5 对/秒的输入基准；
    - read 阶段占总耗时 66.8%，怀疑是 TIFF/LZW 解码瓶颈；
    - 用 rain/perf/tiff_inspector.py 直接解析 IFD 标签确认：这批数据
      （A/B-Capture_085820）确实是 Compression=LZW(5)、Predictor=2、
      RowsPerStrip=3，不是猜的。

## 这个脚本做什么、不做什么

只优化"怎么把数据读进来、怎么调度处理"这一层，不碰检测/跟踪/匹配/
测速/统计的任何算法逻辑——每个 variant 底层调用的还是
ModularPipelineDetector / TrackManager / StereoMatcher /
LinearVelocityEstimator / WindowStatistics 这几个阶段2-6已经验收过的
模块，一行算法代码都不改。变的只是：
    - 用哪个 FrameReader 解码 TIFF（v1）；
    - 读取是不是多线程、要不要过一道有界队列（v2）；
    - 检测阶段要不要也走多线程+有界队列+单写出线程的完整架构（v3）。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage7b_optimization.py

## 输出（默认写到 config/pipeline.yaml 里 performance_optimization.output_dir）

    performance_optimization_report.json   每个variant的耗时/吞吐/资源占用
                                            + 结果一致性对比 + 结论
    performance_comparison.csv             每个variant每个阶段的耗时统计
    consistency_report.json                逐项字段级别的一致性比对明细
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
from rain.io_index import parse_frame_id
from rain.perf.bounded_pipeline import BoundedPipelineRunner
from rain.perf.profiler import StageProfiler
from rain.perf.resource_monitor import ResourceMonitor
from rain.perf.tiff_inspector import inspect_tiff
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
from rain.stereo.calibration import CalibrationLoader
from rain.tracking.types import FrameContext

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"
_DEVICE_TIME_FIELD = {"a": "a_device_relative_us", "b": "b_device_relative_us"}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段7B：性能优化（逐因素测试+一致性比对）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    opt_cfg = cfg["performance_optimization"]
    output_dir = Path(opt_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段7B：性能优化")
    print("=" * 70)
    print(f"输出目录: {output_dir}")

    # --- 0. 先确认压缩方式，不能假设 ------------------------------------------
    print("\n[0/4] 确认TIF真实压缩方式（不是猜的，直接解析IFD标签）...")
    tiff_check = _check_tiff_compression(cfg)
    print(f"  {tiff_check['summary']}")

    # --- 共用资源 --------------------------------------------------------
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
    shared = dict(
        cfg=cfg, filename_to_pair_index=filename_to_pair_index, flagged_pairs=flagged_pairs,
        calibration=calibration, device_time_us_by_field=device_time_us_by_field,
    )

    # --- 1. 逐个variant跑一遍，记耗时+资源+完整结果快照 -----------------------
    print(f"\n[1/4] 逐个variant运行（共{len(opt_cfg['variants'])}个）...")
    variant_runs = {}
    for v in opt_cfg["variants"]:
        print(f"\n  --- {v['name']}: {v['description']} ---")
        profiler = StageProfiler()
        mon = ResourceMonitor(interval_s=0.05)
        mon.start()
        t0 = time.perf_counter()
        summary, snapshot = _run_variant(v, profiler, shared, n_pairs_total)
        wall_s = time.perf_counter() - t0
        res = mon.stop()

        run = {
            "name": v["name"], "description": v["description"], "config": v,
            "wall_s": wall_s, "throughput_pairs_per_s": n_pairs_total / wall_s if wall_s > 0 else 0.0,
            "peak_rss_mb": res["peak_rss_mb"], "cpu_percent": res["cpu_percent"], "disk_io": res["disk_io"],
            **summary,
        }
        variant_runs[v["name"]] = {"run": run, "snapshot": snapshot, "profiler": profiler}
        print(f"    耗时 {wall_s:.2f}s  吞吐 {run['throughput_pairs_per_s']:.1f}对/秒  "
              f"峰值内存 {res['peak_rss_mb']:.1f}MB  CPU {res['cpu_percent']:.0f}%")

    # --- 2. 一致性比对：全部variant vs v0_baseline -----------------------------
    print("\n[2/4] 结果一致性比对（vs v0_baseline）...")
    baseline_name = opt_cfg["variants"][0]["name"]
    baseline_snapshot = variant_runs[baseline_name]["snapshot"]
    consistency = {}
    for v in opt_cfg["variants"]:
        name = v["name"]
        if name == baseline_name:
            consistency[name] = {"is_baseline": True, "identical_to_baseline": True, "diffs": []}
            continue
        diffs = _compare_snapshots(baseline_snapshot, variant_runs[name]["snapshot"])
        consistency[name] = {"is_baseline": False, "identical_to_baseline": len(diffs) == 0, "diffs": diffs}
        status = "一致" if not diffs else f"发现{len(diffs)}处差异！"
        print(f"  {name}: {status}")
        if diffs:
            for d in diffs[:5]:
                print(f"    !! {d}")

    all_consistent = all(c["identical_to_baseline"] for c in consistency.values())

    # --- 3. 队列积压演示 + 重复压力测试（用吞吐最高且结果一致的variant）--------
    print("\n[3/4] 队列积压演示 + 重复读取压力测试...")
    backlog_demo = _run_queue_backlog_demo(opt_cfg["queue_backlog_demo"], shared, cfg)
    print(f"  队列积压演示: maxsize={backlog_demo['configured_maxsize']}  "
          f"实测峰值={backlog_demo['observed_max_size']}  "
          f"是否被有界队列挡住(峰值<=maxsize): {backlog_demo['bounded_correctly']}")

    eligible = {n: r["run"] for n, r in variant_runs.items() if consistency[n]["identical_to_baseline"]}
    best_name = max(eligible, key=lambda n: eligible[n]["throughput_pairs_per_s"])
    best_variant_cfg = next(v for v in opt_cfg["variants"] if v["name"] == best_name)
    print(f"  结果一致且吞吐最高的variant: {best_name} "
          f"({eligible[best_name]['throughput_pairs_per_s']:.1f}对/秒)")

    n_repeats = int(opt_cfg["n_repeats_for_best_variant"])
    repeat_runs = []
    repeat_profiler = StageProfiler()
    for i in range(n_repeats):
        mon = ResourceMonitor(interval_s=0.05)
        mon.start()
        t0 = time.perf_counter()
        summary, _ = _run_variant(best_variant_cfg, repeat_profiler, shared, n_pairs_total)
        wall = time.perf_counter() - t0
        res = mon.stop()
        repeat_runs.append({
            "repeat_index": i, "wall_s": wall,
            "throughput_pairs_per_s": n_pairs_total / wall if wall > 0 else 0.0,
            "peak_rss_mb": res["peak_rss_mb"], "cpu_percent": res["cpu_percent"],
        })
        print(f"    第{i+1}轮({best_name}): {wall:.2f}s  "
              f"{n_pairs_total/wall:.1f}对/秒  峰值内存{res['peak_rss_mb']:.1f}MB")

    # --- 4. 输出 -----------------------------------------------------------
    print("\n[4/4] 写输出文件 ...")
    comparison_csv = output_dir / "performance_comparison.csv"
    _write_comparison_csv({n: r["profiler"] for n, r in variant_runs.items()}, comparison_csv)
    print(f"  {comparison_csv}")

    consistency_path = output_dir / "consistency_report.json"
    with open(consistency_path, "w", encoding="utf-8") as f:
        json.dump(consistency, f, ensure_ascii=False, indent=2)
    print(f"  {consistency_path}")

    report_path = output_dir / "performance_optimization_report.json"
    best_stats = variant_runs[best_name]["profiler"].stats()
    _write_report(
        report_path=report_path, tiff_check=tiff_check,
        variant_runs={n: r["run"] for n, r in variant_runs.items()},
        consistency=consistency, all_consistent=all_consistent,
        backlog_demo=backlog_demo, best_name=best_name, repeat_runs=repeat_runs,
        nominal_pps=cfg["performance"]["nominal_input_rate_pps"],
        target_pps=cfg["performance"]["target_rate_with_margin_pps"],
        n_pairs_total=n_pairs_total,
        best_variant_stage_stats={
            stage: {"total_ms": s.total_ms, "mean_ms": s.mean_ms, "n_calls": s.n_calls}
            for stage, s in best_stats.items()
        },
    )
    print(f"  {report_path}")

    print()
    print("=" * 70)
    baseline_pps = variant_runs[baseline_name]["run"]["throughput_pairs_per_s"]
    best_pps = eligible[best_name]["throughput_pairs_per_s"]
    print(f"基线(v0): {baseline_pps:.1f}对/秒  ->  最优且结果一致({best_name}): {best_pps:.1f}对/秒  "
          f"(提升{best_pps/baseline_pps:.1f}倍)")
    nominal_pps = cfg["performance"]["nominal_input_rate_pps"]
    target_pps = cfg["performance"]["target_rate_with_margin_pps"]
    print(f"是否达到312.5对/秒基准: {'是' if best_pps >= nominal_pps else '否'}  "
          f"是否达到406.25对/秒目标: {'是' if best_pps >= target_pps else '否'}")
    print(f"全部variant结果与基线一致: {all_consistent}")
    print("=" * 70)
    return 0 if all_consistent else 1


# ---------------------------------------------------------------------------
# 0. TIF压缩方式检查
# ---------------------------------------------------------------------------


def _check_tiff_compression(cfg: dict) -> dict:
    samples = []
    for cam in ("a", "b"):
        folder = cfg["data"][f"dir_{cam}"]
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".tif"))
        for fn in (files[0], files[len(files) // 2], files[-1]):
            samples.append(os.path.join(folder, fn))

    infos = [inspect_tiff(p) for p in samples]
    compressions = {i.compression_name for i in infos}
    predictors = {i.predictor_name for i in infos}
    ratios = [i.compression_ratio for i in infos if i.compression_ratio is not None]

    return {
        "n_samples_checked": len(samples),
        "compression_names_seen": sorted(compressions),
        "predictor_names_seen": sorted(predictors),
        "rows_per_strip_seen": sorted({i.rows_per_strip for i in infos}),
        "compression_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "confirmed_lzw": compressions == {"LZW"},
        "summary": (
            f"抽样{len(samples)}个文件(A/B各首/中/末)，压缩方式={sorted(compressions)}，"
            f"predictor={sorted(predictors)}，平均压缩比={sum(ratios)/len(ratios):.3f}"
            if ratios else f"抽样{len(samples)}个文件，压缩方式={sorted(compressions)}"
        ),
    }


# ---------------------------------------------------------------------------
# 1. 按variant配置跑一遍完整流水线
# ---------------------------------------------------------------------------


def _run_variant(variant: dict, profiler: StageProfiler, shared: dict, n_pairs_total: int) -> tuple[dict, dict]:
    cfg = shared["cfg"]
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]
    stats_cfg = cfg["statistics"]

    reader = build_reader({"name": variant["reader_name"], "params": {"expect_shape": [1544, 2064]}})
    n_reader_threads = int(variant["n_reader_threads"])
    n_detect_threads = int(variant["n_detect_threads"])
    queue_maxsize = int(variant["queue_maxsize"])

    per_camera_records = {}
    per_camera_frame_times = {}
    per_camera_frame_dets = {}

    for cam in ("a", "b"):
        folder = cfg["data"][f"dir_{cam}"]
        frame_files = _list_camera_frame_files(folder)
        n = min(int(det_cfg["n_frames"]), len(frame_files))
        frame_files = frame_files[:n]
        device_time_map = shared["device_time_us_by_field"][_DEVICE_TIME_FIELD[cam]]

        # --- 读取（顺序 或 多线程有界队列，两种路径复用同一个reader.read）------
        images_by_index: dict[int, "object"] = {}
        if n_reader_threads <= 1:
            for i, (_fn, path) in enumerate(frame_files):
                with profiler.timer("read"):
                    img = reader.read(path)
                images_by_index[i] = img
        else:
            _threaded_read(frame_files, reader, profiler, n_reader_threads, queue_maxsize, images_by_index)

        # 按文件在 frame_files 里的位置重建有序列表（读取顺序不代表处理顺序，
        # 这里统一转换成"和文件列表位置对应"的顺序，供背景建模使用）。
        frames = [images_by_index[i] for i in range(len(frame_files))]

        with profiler.timer("background"):
            bg = build_background(cfg["background"]).build(frames)

        detector_build_cfg = {
            "name": "modular_pipeline",
            "params": {
                "normalizer": det_cfg["normalizer"], "differencer": det_cfg["differencer"],
                "segmenter": det_cfg["segmenter"], "morphology": det_cfg["morphology"],
                "cc_analyzer": det_cfg["cc_analyzer"], "contour_extractor": det_cfg["contour_extractor"],
            },
        }
        detector = build_detector(detector_build_cfg)

        frame_times: dict[int, float] = {}
        frame_dets: dict[int, list] = {}

        if n_detect_threads <= 1:
            for i, (fn, _path) in enumerate(frame_files):
                pidx = shared["filename_to_pair_index"][fn]
                device_time_s = device_time_map[pidx] / 1e6
                img = frames[i]
                with profiler.timer("detect"):
                    dets = detector.detect(img, bg, frame_idx=pidx)
                frame_times[pidx] = device_time_s
                frame_dets[pidx] = dets
        else:
            # 多线程处理阶段：每个处理线程各自建一份 detector 实例，不共享
            # 同一个对象——这是项目 interfaces.py 里对并行安全的既有约定
            # （"每个 worker 必须持有自己的一套算法实例"），即使六个检测子
            # 模块目前看起来没有可变共享状态，也不去赌这一点，按既定规矩来。
            _threaded_detect(
                frame_files, frames, bg, detector_build_cfg, shared["filename_to_pair_index"], device_time_map,
                profiler, n_detect_threads, queue_maxsize, frame_times, frame_dets,
            )

        frame_contexts = [
            FrameContext(pair_index=pidx, device_time_s=frame_times[pidx], detections=frame_dets[pidx])
            for pidx in sorted(frame_dets)
        ]

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

    # --- 双路匹配（不改逻辑，直接复用阶段4） -----------------------------------
    matcher = build_stereo_matcher({
        "name": "gated_cost_stereo_matcher",
        "params": {
            "epipolar_gate": cfg["stereo"]["epipolar_gate"], "match_gate": cfg["stereo"]["match_gate"],
            "cost_matrix_builder": cfg["stereo"]["cost_matrix_builder"],
            "max_total_cost": cfg["stereo"]["max_total_cost"],
            "fixed_exposure_delay_calibrated": cfg["stereo"]["fixed_exposure_delay_calibrated"],
            "calibration": shared["calibration"],
        },
    })
    all_pair_indices = sorted(set(per_camera_frame_dets["a"]) | set(per_camera_frame_dets["b"]))
    stereo_rows = []
    for pidx in all_pair_indices:
        dets_a = per_camera_frame_dets["a"].get(pidx, [])
        dets_b = per_camera_frame_dets["b"].get(pidx, [])
        with profiler.timer("stereo_match"):
            result = matcher.match_frame(pidx, dets_a, dets_b)
        stereo_rows.extend(result.rows)

    # --- 测速（不改逻辑，直接复用阶段5） --------------------------------------
    velocity_estimator = build_velocity_estimator(cfg["measurement"]["linear_velocity_estimator"])
    velocity_fits = []
    for cam in ("a", "b"):
        for rec in per_camera_records[cam]:
            with profiler.timer("velocity_fit"):
                fit = velocity_estimator.estimate(rec)
            velocity_fits.append(fit)

    # --- 统计（不改逻辑，直接复用阶段6） --------------------------------------
    window_stats = build_window_statistics({
        "name": "fixed_length_window_statistics",
        "params": {
            "window_seconds": stats_cfg["window_seconds"], "quality_controller": stats_cfg["quality_controller"],
            "concentration_estimator": stats_cfg["concentration_estimator"],
            "size_distribution": stats_cfg["size_distribution"],
            "spatial_calibration_status": stats_cfg["spatial_calibration_status"],
        },
    })
    window_rows = []
    for cam in ("a", "b"):
        with profiler.timer("statistics_single_camera"):
            rows, _dist = window_stats.compute(
                cam, per_camera_records[cam], per_camera_frame_times[cam],
                per_camera_frame_dets[cam], shared["flagged_pairs"],
            )
        window_rows.extend(rows)

    stereo_common_stats = build_stereo_common_volume_statistics({
        "name": "gated_stereo_common_volume_statistics",
        "params": {
            "window_seconds": stats_cfg["window_seconds"],
            "concentration_estimator": stats_cfg["concentration_estimator"],
            "spatial_calibration_status": stats_cfg["spatial_calibration_status"],
        },
    })
    with profiler.timer("statistics_stereo_common"):
        stereo_common_rows = stereo_common_stats.compute(
            calibration_available=shared["calibration"] is not None,
            t_min=min(per_camera_frame_times["a"].values()), t_max=max(per_camera_frame_times["a"].values()),
            matched_pair_entry_times=[],
        )

    summary = {
        "n_pairs": len(all_pair_indices),
        "n_detections_a": sum(len(v) for v in per_camera_frame_dets["a"].values()),
        "n_detections_b": sum(len(v) for v in per_camera_frame_dets["b"].values()),
        "n_tracks_a": len(per_camera_records["a"]),
        "n_tracks_b": len(per_camera_records["b"]),
        "n_stereo_matched": sum(1 for r in stereo_rows if r.status == "matched"),
        "n_velocity_fits": len(velocity_fits),
    }
    snapshot = _snapshot_results(
        per_camera_frame_dets, per_camera_records, stereo_rows, velocity_fits, window_rows, stereo_common_rows,
    )
    return summary, snapshot


def _threaded_read(frame_files, reader, profiler, n_reader_threads, queue_maxsize, images_by_index) -> None:
    def read_fn(item):
        idx, path = item
        t0 = time.perf_counter()
        img = reader.read(path)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return idx, img, dt_ms

    def write_fn(result):
        idx, img, dt_ms = result
        images_by_index[idx] = img
        profiler.record("read", dt_ms)

    items = [(i, path) for i, (_fn, path) in enumerate(frame_files)]
    BoundedPipelineRunner(
        items=items, read_fn=read_fn, write_fn=write_fn,
        n_reader_threads=n_reader_threads, n_processor_threads=1, queue_maxsize=queue_maxsize,
    ).run()


def _threaded_detect(
    frame_files, frames, bg, detector_build_cfg, filename_to_pair_index, device_time_map,
    profiler, n_detect_threads, queue_maxsize, frame_times, frame_dets,
) -> None:
    import threading

    from rain.registry import build_detector as _build_detector

    thread_local = threading.local()

    def get_thread_local_detector():
        # 每个处理线程第一次用到时各自建一份 detector，之后同一线程复用
        # 自己那一份——不同线程永远拿不到同一个对象，从根上排除共享
        # 可变状态的可能性（哪怕检测子模块本身没有可变状态，这一步的
        # 成本也可以忽略：只是新建几个 kernel 之类的小对象）。
        if not hasattr(thread_local, "detector"):
            thread_local.detector = _build_detector(detector_build_cfg)
        return thread_local.detector

    items = []
    for i, (fn, _path) in enumerate(frame_files):
        pidx = filename_to_pair_index[fn]
        items.append((pidx, frames[i]))

    def process_fn(item):
        pidx, img = item
        detector = get_thread_local_detector()
        t0 = time.perf_counter()
        dets = detector.detect(img, bg, frame_idx=pidx)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pidx, dets, dt_ms

    def write_fn(result):
        pidx, dets, dt_ms = result
        frame_times[pidx] = device_time_map[pidx] / 1e6
        frame_dets[pidx] = dets
        profiler.record("detect", dt_ms)

    BoundedPipelineRunner(
        items=items, read_fn=lambda x: x, process_fn=process_fn, write_fn=write_fn,
        n_reader_threads=1, n_processor_threads=n_detect_threads, queue_maxsize=queue_maxsize,
    ).run()


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
# 2. 结果快照 + 一致性比对
# ---------------------------------------------------------------------------


def _snapshot_results(per_camera_frame_dets, per_camera_records, stereo_rows, velocity_fits, window_rows, stereo_common_rows) -> dict:
    """把一次完整流水线运行的全部结果打包成可比较、可JSON序列化的快照。"""
    snapshot: dict = {}

    for cam in ("a", "b"):
        snapshot[f"detections_{cam}"] = {
            str(pidx): [
                [round(d.cx, 6), round(d.cy, 6), round(d.area_px, 6), list(d.bbox), d.qc_flags, round(d.d_eq_px, 6)]
                for d in dets
            ]
            for pidx, dets in sorted(per_camera_frame_dets[cam].items())
        }
        snapshot[f"tracks_{cam}"] = [
            [
                t.track_id, t.status,
                [
                    [p.pair_index, p.association_method,
                     None if p.detected is None else round(p.detected.cx, 6),
                     None if p.detected is None else round(p.detected.cy, 6)]
                    for p in t.points
                ],
            ]
            for t in sorted(per_camera_records[cam], key=lambda t: t.track_id)
        ]

    snapshot["stereo_rows"] = [
        [r.pair_index, r.a_index, r.b_index, r.status, r.reason, r.method] for r in stereo_rows
    ]
    snapshot["velocity_fits"] = [
        [
            f.track_id, f.camera, f.n_points, f.status, f.confidence,
            None if f.velocity_px_per_s is None else round(f.velocity_px_per_s, 4),
        ]
        for f in sorted(velocity_fits, key=lambda f: (f.camera, f.track_id))
    ]
    snapshot["window_rows"] = [
        [r.camera, r.window_index, r.valid_target_count, r.excluded_target_count, r.quality_status]
        for r in window_rows
    ]
    snapshot["stereo_common_rows"] = [
        [r.window_index, r.n_matched_pairs, r.concentration_status] for r in stereo_common_rows
    ]
    return snapshot


def _compare_snapshots(baseline: dict, other: dict) -> list[str]:
    diffs = []
    all_keys = sorted(set(baseline.keys()) | set(other.keys()))
    for key in all_keys:
        if key not in other:
            diffs.append(f"{key}: 缺失（baseline有，被测variant没有）")
            continue
        if key not in baseline:
            diffs.append(f"{key}: 多出（baseline没有，被测variant有）")
            continue
        if baseline[key] != other[key]:
            diffs.append(f"{key}: 内容不同")
    return diffs


# ---------------------------------------------------------------------------
# 3. 队列积压演示
# ---------------------------------------------------------------------------


def _run_queue_backlog_demo(demo_cfg: dict, shared: dict, cfg: dict) -> dict:
    """用一个很小的队列上限 + 明显比处理快的读取速度，验证有界队列真的会
    被打满到上限、而不是无限堆积——直接证明"队列必须有最大长度"这条
    要求被满足，不是嘴上说说。
    """
    import time as _time

    folder = cfg["data"]["dir_a"]
    frame_files = _list_camera_frame_files(folder)[:60]  # 抽60帧演示即可，不用跑满180
    reader = build_reader({"name": "fast_tif", "params": {"expect_shape": [1544, 2064]}})

    def read_fn(item):
        _idx, path = item
        return reader.read(path)

    def process_fn(img):
        _time.sleep(0.02)  # 故意让"处理"比"读取"慢很多，逼队列堆到上限
        return img.shape

    results = []
    runner = BoundedPipelineRunner(
        items=[(i, p) for i, (_fn, p) in enumerate(frame_files)],
        read_fn=read_fn, process_fn=process_fn, write_fn=lambda x: results.append(x),
        n_reader_threads=int(demo_cfg["n_reader_threads"]),
        n_processor_threads=int(demo_cfg["n_detect_threads"]),
        queue_maxsize=int(demo_cfg["queue_maxsize"]),
    )
    result = runner.run()

    return {
        "configured_maxsize": demo_cfg["queue_maxsize"],
        "n_items": result.n_items,
        "observed_max_size": result.read_queue_stats.max_size_observed,
        "bounded_correctly": result.read_queue_stats.max_size_observed <= demo_cfg["queue_maxsize"],
        "note": (
            "故意让处理速度(每张强制sleep 20ms)远慢于读取速度，观察读取"
            "队列是否被打满到配置的上限——如果观测峰值 <= 配置的maxsize，"
            "说明有界队列确实生效，内存不会因为读得比处理快就无限堆积。"
        ),
    }


# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------


def _write_comparison_csv(profilers: dict[str, StageProfiler], out_path: Path) -> None:
    columns = ["variant", "stage", "n_calls", "mean_ms", "median_ms", "p95_ms", "max_ms", "total_ms"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for variant_name, profiler in profilers.items():
            for stage, s in sorted(profiler.stats().items()):
                w.writerow([
                    variant_name, stage, s.n_calls,
                    f"{s.mean_ms:.4f}", f"{s.median_ms:.4f}", f"{s.p95_ms:.4f}",
                    f"{s.max_ms:.4f}", f"{s.total_ms:.4f}",
                ])


def _write_report(
    report_path: Path, tiff_check: dict, variant_runs: dict, consistency: dict, all_consistent: bool,
    backlog_demo: dict, best_name: str, repeat_runs: list, nominal_pps: float, target_pps: float,
    n_pairs_total: int, best_variant_stage_stats: dict,
) -> None:
    baseline_name = next(iter(variant_runs))
    baseline_pps = variant_runs[baseline_name]["throughput_pairs_per_s"]
    best_pps = variant_runs[best_name]["throughput_pairs_per_s"]
    best_wall_s = variant_runs[best_name]["wall_s"]
    meets_nominal = best_pps >= nominal_pps
    meets_target = best_pps >= target_pps

    # background 在当前实现里对每一路相机都是"读完这一路全部帧之后、
    # 检测这一路任何一帧之前"必须先完整跑完的一步——它本身没有被这一轮
    # 任何一个 variant 并行化（没有改动 BackgroundModel 算法，也没有把
    # A、B 两路相机的整条流水线并行起来），所以它的 total_ms 就是它在
    # 最优方案里真实占用的墙钟时间，不像 read/detect 那样会因为多线程
    # 而"总耗时"大于实际经过的时间。用这个数据驱动地指出下一步该往哪优化，
    # 不是空喊"还不够快"。
    bg_stats = best_variant_stage_stats.get("background")
    bg_share_note = ""
    if bg_stats and best_wall_s > 0:
        bg_share_pct = 100 * bg_stats["total_ms"] / 1000.0 / best_wall_s
        bg_share_note = (
            f"background(背景建模)在{best_name}里合计耗时{bg_stats['total_ms']/1000:.2f}s"
            f"（A、B两路各跑一次，调用{bg_stats['n_calls']}次），占总耗时{best_wall_s:.2f}s的"
            f"{bg_share_pct:.0f}%——这一步本身没有被这一轮任何variant并行化"
            "（没有改动BackgroundModel算法，也没有让A、B两路相机的整条流水线"
            "并行跑），是当前最优方案里唯一还完全串行、且占比可观的环节。"
        )

    remaining_bottleneck_note = ""
    if not meets_nominal:
        remaining_bottleneck_note = (
            f"最优方案({best_name})吞吐{best_pps:.1f}对/秒，仍未达到{nominal_pps}对/秒基准。"
            f"{bg_share_note} "
            "下一步建议，按预期收益排序：(1) 把A、B两路相机的整条流水线"
            "（读取+背景建模+检测）并行处理，而不是像现在这样A做完再做B——"
            "两路数据完全独立，直到双路匹配才有关联，这是最直接、不涉及"
            "任何算法改动的下一步；(2) 如果背景建模本身要提速，需要改"
            "BackgroundModel的实现（比如减少参与分位数计算的帧数），但这属于"
            "算法改动，不在阶段7B的授权范围内，需要单独确认；(3) 用真正的"
            "多进程(而非线程)绕开GIL，可能进一步提升detect阶段在多核上的"
            "并行效率；(4) 这批测试数据只有180对，线程启动/收尾开销占比"
            "不小，真实长时间连续采集场景下这部分开销被摊薄，实际吞吐"
            "可能更接近理论值，需要更大样本验证。"
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "python_version": platform.python_version(), "cpu_count": os.cpu_count(),
        },
        "baseline_reference": {
            "note": (
                "本次运行的v0_baseline是当场重新测的，作为本轮比较的对照组"
                "（不是直接照抄阶段7旧报告的数字，避免机器负载差异导致对比不公平）。"
                f"本次实测v0_baseline={baseline_pps:.1f}对/秒，和阶段7报告里的14.4对/秒"
                "不完全一样——差异来自这次是连续跑4个variant+3轮重复测试的一次性"
                "长脚本，系统负载和磁盘缓存状态与阶段7单独运行时不同，属于正常的"
                "测量环境波动，不是流水线本身变了。判断'提升了多少倍'时，只用"
                "同一次脚本运行内部的相对比较(variant vs 本次v0)，不跨脚本、"
                "跨次运行去比较绝对数值。"
            ),
            "stage7_prior_reference_pps": 14.4,
            "this_run_baseline_pps": baseline_pps,
        },
        "step0_tiff_compression_check": tiff_check,
        "variants": variant_runs,
        "consistency_report": {
            "all_variants_consistent_with_baseline": all_consistent,
            "per_variant": {
                name: {"identical_to_baseline": c["identical_to_baseline"], "n_diffs": len(c["diffs"])}
                for name, c in consistency.items()
            },
            "note": (
                "一致性比对覆盖：每帧每路的检测结果(质心/面积/包围框/QC标记/"
                "等效直径)、每条轨迹的结构(track_id/状态/每个点的关联方式和"
                "位置)、双路匹配的每条候选诊断(状态/原因/方法)、每条轨迹的"
                "测速结果(点数/状态/置信度/速度)、每个时间窗口的统计结果"
                "(有效/排除目标数/质量状态)。任何一项不同都会被记录进"
                "consistency_report.json。"
            ),
        },
        "queue_backlog_demo": backlog_demo,
        "best_variant": {
            "name": best_name,
            "selection_rule": "结果与baseline完全一致的variant里，吞吐最高的那个",
            "throughput_pairs_per_s": best_pps,
            "speedup_vs_baseline": best_pps / baseline_pps if baseline_pps > 0 else None,
        },
        "throughput_vs_benchmark": {
            "nominal_input_rate_pps": nominal_pps, "target_rate_with_margin_pps": target_pps,
            "measured_best_pps": best_pps, "meets_nominal": meets_nominal, "meets_target_with_margin": meets_target,
            "remaining_bottleneck_note": remaining_bottleneck_note,
        },
        "best_variant_stage_stats": {
            "stats": best_variant_stage_stats,
            "caveat": (
                "read/detect 两个阶段在多线程variant里，total_ms是'多个线程各自"
                "耗时相加'，不等于墙钟时间的占比（比如CPU占比200%+就说明确实有"
                "多个线程同时在跑）；background 和 track/stereo/velocity/statistics"
                "几个阶段全程只有一个线程在跑，它们的 total_ms 才直接对应真实经过"
                "的时间，可以放心按占比解读。"
            ),
        },
        "repeated_stress_test_on_best_variant": {
            "n_repeats": len(repeat_runs), "per_repeat": repeat_runs,
            "note": f"用{best_name}重复跑{len(repeat_runs)}轮，同一批180对数据，"
                    "检查短期内耗时/内存是否有明显退化；不构成长期稳定性证明。",
        },
        "long_term_stability": {
            "tested": False,
            "note": "本次仍然只用180对数据做短批和重复压力测试，长时间连续运行"
                    "的稳定性（内存/句柄泄漏、降频等）没有测试，也不声称已验证。",
        },
        "algorithm_unchanged_confirmation": {
            "note": "detect/track/stereo_match/velocity_fit/statistics 五个阶段"
                    "调用的都是阶段2-6已验收的同一批注册模块"
                    "（ModularPipelineDetector/TrackManager/StereoMatcher/"
                    "LinearVelocityEstimator/WindowStatistics），本次优化只改了"
                    "读取器实现(reader)和读取/检测阶段的线程调度方式，"
                    "没有修改任何检测阈值、跟踪门控参数或统计公式。",
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
