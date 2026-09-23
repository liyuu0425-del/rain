#!/usr/bin/env python3
"""阶段5：测速 —— 对A、B两路各自的单路轨迹独立做线性测速。

## 用途

复用阶段2的六步检测流水线 + 阶段3的卡尔曼跟踪（原地重新跑一遍，不重新
实现），对每条轨迹用 LinearVelocityEstimator 拟合 z(t)=z0+v*t，
z 取竖直方向像素坐标 cy。A、B 两路完全独立测速，不依赖阶段4的双路匹配
（阶段4已确认当前无标定文件、0个已验证匹配，双路融合测速留到有标定之后）。

## 时间基准（硬性约束）

    A 路轨迹用 a_device_relative_us；B 路轨迹用 b_device_relative_us。
    不使用文件名毫秒、host_receive时间、pair_wall_datetime_ms、
    帧号或固定帧率——这条约束由阶段3的 TrackPoint.device_time_s 保证，
    这个脚本不重新引入任何时间来源。

## 单位现状

    没有像素-毫米标定，只输出 px/s，不输出 mm/s 或 m/s；
    spatial_calibration_status 恒为 spatial_calibration_unavailable。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage5_velocity.py

## 输出（默认写到 config/pipeline.yaml 里 measurement.output_dir 指定的目录）

    velocities.csv              每条轨迹的拟合结果
    plots/{cam}_track{id}.png   位置-时间拟合图（抽样）
    stage5_report.json          点数分布、A/B速度结果、现状说明、模块耗时
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

matplotlib.use("Agg")  # 无显示器环境下渲染PNG，不弹窗、不依赖GUI后端
import matplotlib.pyplot as plt
import numpy as np
import yaml

import rain  # noqa: F401  触发所有实现的注册
from rain.data.timing_reader import TimingReader
from rain.io_index import parse_frame_id
from rain.measurement.types import VelocityFitResult
from rain.registry import (
    build_background,
    build_detector,
    build_reader,
    build_track_manager,
    build_velocity_estimator,
)
from rain.tracking.types import FrameContext, TrackRecord

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"
_DEVICE_TIME_FIELD = {"a": "a_device_relative_us", "b": "b_device_relative_us"}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段5：单路线性测速（A/B独立）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    meas_cfg = cfg["measurement"]
    output_dir = Path(meas_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段5：测速（单路线性拟合，A/B独立，暂不依赖双路匹配）")
    print("=" * 70)
    print(f"输出目录 : {output_dir}")

    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}
    device_time_us_by_field = {
        "a_device_relative_us": {r.pair_index: r.a_device_relative_us for r in timing_records},
        "b_device_relative_us": {r.pair_index: r.b_device_relative_us for r in timing_records},
    }

    reader = build_reader(cfg["reader"])
    estimator = build_velocity_estimator(meas_cfg["linear_velocity_estimator"])

    camera_results: dict[str, dict] = {}
    all_fits: list[VelocityFitResult] = []
    track_lookup: dict[tuple[str, int], TrackRecord] = {}

    for cam in ("a", "b"):
        print(f"\n{'-'*70}\n处理 {cam.upper()} 路\n{'-'*70}")
        records, track_ms = _detect_and_track_camera(
            cam=cam, folder=cfg["data"][f"dir_{cam}"], cfg=cfg, reader=reader,
            filename_to_pair_index=filename_to_pair_index,
            device_time_us_map=device_time_us_by_field[_DEVICE_TIME_FIELD[cam]],
        )
        print(f"  {len(records)} 条轨迹（检测+跟踪耗时 {track_ms:.0f} ms）")

        t0 = time.perf_counter()
        fits = [estimator.estimate(rec) for rec in records]
        fit_ms = (time.perf_counter() - t0) * 1000.0

        for rec in records:
            track_lookup[(cam, rec.track_id)] = rec

        all_fits.extend(fits)
        camera_results[cam] = _summarize_camera(cam, fits, fit_ms, len(records))
        _print_camera_summary(cam, camera_results[cam])

    # --- 输出 velocities.csv ------------------------------------------------
    velocities_csv = output_dir / "velocities.csv"
    _write_velocities_csv(all_fits, velocities_csv)
    print(f"\n{velocities_csv} 共 {len(all_fits)} 行")

    # --- 位置-时间拟合图 -------------------------------------------------------
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_plot = int(meas_cfg.get("n_plot_samples", 4))
    for cam in ("a", "b"):
        _save_sample_plots(cam, all_fits, track_lookup, plots_dir, n_plot)

    # --- 报告 -----------------------------------------------------------------
    report_path = output_dir / "stage5_report.json"
    _write_report(report_path, camera_results)
    print(f"{report_path}")

    print()
    print("=" * 70)
    print("阶段5完成。重要提醒：以上速度结果单位是 px/s（像素/秒），")
    print("没有像素-毫米标定，不能代表物理下落速度(m/s)，也不是双路融合结果。")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# 检测 + 跟踪（复用阶段2/阶段3，不重新实现）
# ---------------------------------------------------------------------------


def _detect_and_track_camera(
    cam: str, folder: str, cfg: dict, reader,
    filename_to_pair_index: dict, device_time_us_map: dict,
) -> tuple[list[TrackRecord], float]:
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
    frame_contexts: list[FrameContext] = []
    for (fn, _path), img in zip(frame_files, frames):
        if fn not in filename_to_pair_index:
            raise KeyError(f"文件 {fn} 在 capture_timing.csv 里找不到 pair_index，停止处理。")
        pidx = filename_to_pair_index[fn]
        device_time_s = device_time_us_map[pidx] / 1e6
        dets = detector.detect(img, bg, frame_idx=pidx)
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

    return records, ms


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


def _summarize_camera(cam: str, fits: list[VelocityFitResult], fit_ms: float, n_tracks: int) -> dict:
    n_regular = sum(1 for f in fits if f.status == "fitted")
    n_two_point = sum(1 for f in fits if f.status == "fitted_two_point")
    n_insufficient = sum(1 for f in fits if f.status == "insufficient_points")
    n_rejected = sum(1 for f in fits if f.status == "rejected")

    conf_dist: dict[str, int] = {}
    for f in fits:
        conf_dist[f.confidence] = conf_dist.get(f.confidence, 0) + 1

    velocities = [f.velocity_px_per_s for f in fits if f.velocity_px_per_s is not None]
    velocities_confident = [
        f.velocity_px_per_s for f in fits
        if f.velocity_px_per_s is not None and f.confidence in ("medium", "high")
    ]

    return {
        "camera": cam,
        "n_tracks_total": n_tracks,
        "n_fitted_regular_ge3_points": n_regular,
        "n_fitted_two_point": n_two_point,
        "n_insufficient_points_no_velocity": n_insufficient,
        "n_rejected_degenerate_time": n_rejected,
        "confidence_distribution": conf_dist,
        "velocity_px_per_s_all": _velocity_stats(velocities),
        "velocity_px_per_s_medium_or_higher_confidence": _velocity_stats(velocities_confident),
        "fit_module_ms": fit_ms,
    }


def _velocity_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    arr = np.array(values)
    return {"n": len(arr), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}


def _print_camera_summary(cam: str, s: dict) -> None:
    print(f"  轨迹总数: {s['n_tracks_total']}")
    print(f"  >=3点正规拟合: {s['n_fitted_regular_ge3_points']}  "
          f"仅2点低置信度: {s['n_fitted_two_point']}  "
          f"点数不足无速度: {s['n_insufficient_points_no_velocity']}  "
          f"时间退化被拒: {s['n_rejected_degenerate_time']}")
    print(f"  置信度分布: {s['confidence_distribution']}")
    v_all = s["velocity_px_per_s_all"]
    if v_all["n"]:
        print(f"  速度(px/s，全部) 中位 {v_all['median']:.1f}  范围 [{v_all['min']:.1f}, {v_all['max']:.1f}]")


def _write_velocities_csv(fits: list[VelocityFitResult], out_path: Path) -> None:
    columns = [
        "track_id", "camera_id", "n_points", "duration_s",
        "velocity_px_per_s", "vx_px_per_s", "z0_px",
        "residual_rms_px", "r_squared", "time_quality",
        "spatial_calibration_status", "unit", "confidence", "status", "reason",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in sorted(fits, key=lambda r: (r.camera, r.track_id)):
            w.writerow([
                r.track_id, r.camera, r.n_points,
                "" if r.duration_s is None else f"{r.duration_s:.6f}",
                "" if r.velocity_px_per_s is None else f"{r.velocity_px_per_s:.4f}",
                "" if r.vx_px_per_s is None else f"{r.vx_px_per_s:.4f}",
                "" if r.z0_px is None else f"{r.z0_px:.4f}",
                "" if r.residual_rms_px is None else f"{r.residual_rms_px:.4f}",
                "" if r.r_squared is None else f"{r.r_squared:.4f}",
                r.time_quality, r.spatial_calibration_status, r.unit,
                r.confidence, r.status, r.reason,
            ])


def _save_sample_plots(
    cam: str, all_fits: list[VelocityFitResult], track_lookup: dict, plots_dir: Path, n_plot: int
) -> None:
    """挑这一路里点数最多的几条轨迹，画位置(cy)-时间散点 + 拟合直线。"""
    candidates = [f for f in all_fits if f.camera == cam and f.n_points >= 2]
    candidates.sort(key=lambda f: f.n_points, reverse=True)

    for fit in candidates[:n_plot]:
        rec = track_lookup[(cam, fit.track_id)]
        pts = [p for p in rec.points if p.detected is not None]
        pts.sort(key=lambda p: p.device_time_s)
        t = np.array([p.device_time_s for p in pts])
        z = np.array([p.detected.cy for p in pts])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(t, z, color="tab:blue", label="real detection point (cy)", zorder=3)
        if fit.velocity_px_per_s is not None and fit.z0_px is not None:
            t_line = np.linspace(t.min(), t.max(), 50)
            z_line = fit.z0_px + fit.velocity_px_per_s * t_line
            ax.plot(t_line, z_line, color="tab:red", linestyle="--",
                    label=f"fit: v={fit.velocity_px_per_s:.1f}px/s (confidence={fit.confidence})")
        ax.set_xlabel("device_time_s")
        ax.set_ylabel("cy (px, positive = downward)")
        ax.invert_yaxis()  # image y-axis points down; inverting makes "falling" visually go down
        ax.set_title(f"camera={cam.upper()}  track_id={fit.track_id}  n_points={fit.n_points}  "
                     f"R2={'' if fit.r_squared is None else f'{fit.r_squared:.3f}'}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{cam}_track{fit.track_id}.png", dpi=120)
        plt.close(fig)


def _write_report(report_path: Path, camera_results: dict) -> None:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_camera": camera_results,
        "time_field_used": {"a": "a_device_relative_us", "b": "b_device_relative_us"},
        "unit_and_calibration": {
            "unit": "px/s",
            "spatial_calibration_status": "spatial_calibration_unavailable",
            "note": (
                "没有像素-毫米标定，速度只能是像素/秒，不能换算成毫米/秒或米/秒。"
                "以上数值只表示图像里目标每秒移动多少像素，不表示物理下落速度。"
            ),
        },
        "stage5_caveats": [
            {
                "item": "not_physical_velocity",
                "detail": "当前结果只能表示像素速度(px/s)，不能表示物理下落速度(m/s)。"
                          "换算成物理速度需要像素-毫米标定，这在本项目现阶段尚未完成。",
            },
            {
                "item": "single_camera_only",
                "detail": "A、B两路完全独立测速，没有用阶段4的双路匹配结果——"
                          "阶段4已确认当前无真实标定、0个已验证匹配，双路融合测速"
                          "（用两路观测互相验证/提高精度）留到标定完成后再做。",
            },
            {
                "item": "kalman_not_used_as_observation",
                "detail": "拟合只使用轨迹里真实检测关联的点（TrackRecord.points，"
                          "阶段3设计上已排除卡尔曼'仅预测'点），"
                          "LinearVelocityEstimator 自身也做了防御性过滤，双重保证。",
            },
            {
                "item": "detection_tracking_upstream_limitation",
                "detail": "测速结果的样本量直接受阶段2检测参数、阶段3跟踪门控参数"
                          "未标定的影响——本批数据检出稀疏，多数轨迹只有1~2个真实点，"
                          "能做>=3点正规拟合的轨迹很少，这是上游限制，不是测速模块本身的缺陷。",
            },
        ],
        "known_constraints": [
            "禁止使用文件名毫秒、host_receive时间、pair_wall_datetime_ms、"
            "帧号或固定帧率计算时间/速度。",
            "少于2个真实检测点不输出速度；恰好2个点只输出低置信度速度；"
            "predicted_only点不参与拟合。",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
