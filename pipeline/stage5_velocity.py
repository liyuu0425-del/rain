#!/usr/bin/env python3
"""阶段5：测速 —— 对A、B两路各自的单路轨迹独立做线性测速。

## 用途

直接读取阶段3的 tracks.csv，不重复读图、检测或跟踪。对每条轨迹用
LinearVelocityEstimator 拟合 z(t)=z0+v*t，z 取竖直方向像素坐标 cy。
A、B 两路各自测速，再关联阶段4 status=matched 的轨迹对，保留两路各自的
速度、拟合质量和原匹配置信度，不将配对本身当成测速质量合格的证明。

## 时间基准（硬性约束）

    A 路轨迹用 a_device_relative_us；B 路轨迹用 b_device_relative_us。
    不使用文件名毫秒、host_receive时间、pair_wall_datetime_ms、
    帧号或固定帧率——直接使用阶段3保存的 device_time_s。

## 单位

    原始拟合速度为 px/s；阶段5按 A/B 相机分别读取竖直尺度，
    同时输出 mm/s 和 m/s。尺度来源由 measurement.spatial_calibration_path 指定。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage5_velocity.py

## 输出（默认写到 config/pipeline.yaml 里 measurement.output_dir 指定的目录）

    velocities.csv              每条轨迹的拟合结果
    matched_velocities.csv      阶段4已匹配轨迹对的两路测速结果
    plots/{cam}_track{id}.png   位置-时间拟合图（抽样）
    stage5_report.json          点数分布、A/B速度结果、现状说明、模块耗时

程序结束时还会在终端列出每条可计算速度的轨迹及速度值，并打印各步骤耗时
和阶段5总体耗时。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")  # 无显示器环境下渲染PNG，不弹窗、不依赖GUI后端
import matplotlib.pyplot as plt
import numpy as np
import yaml

import rain  # noqa: F401  触发所有实现的注册
from rain.measurement.types import VelocityFitResult
from rain.registry import build_velocity_estimator
from rain.tracking.types import TQC_PREDICTED_ONLY, TrackPoint, TrackRecord
from rain.types import Detection

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"
_PROJECT_ROOT = Path(__file__).parent.parent
_SPATIAL_STATUS = "spatial_calibration_applied"
_CAMERA_SCALE_KEY = {"a": "camera_A", "b": "camera_B"}


def main() -> int:
    stage_t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="阶段5：复用阶段3轨迹测速，关联阶段4配对")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    step_t0 = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config_load_ms = (time.perf_counter() - step_t0) * 1000.0
    meas_cfg = cfg["measurement"]
    tracks_path = _resolve_project_path(meas_cfg["tracks_csv"])
    matches_value = meas_cfg.get("stereo_matches_csv")
    matches_path = _resolve_project_path(matches_value) if matches_value else None
    calibration_value = meas_cfg.get("spatial_calibration_path")
    calibration_path = _resolve_project_path(calibration_value) if calibration_value else None
    vertical_scales, calibration_status, calibration_note = _load_vertical_scales(calibration_path)

    print("=" * 70)
    print("阶段5：复用阶段3轨迹测速，关联阶段4配对")
    print("=" * 70)
    print(f"轨迹输入 : {tracks_path}")
    print(f"竖直尺度 : {_format_scales(vertical_scales)} ({calibration_status})")
    t0 = time.perf_counter()
    records_by_camera, input_counts = _read_tracks_csv(tracks_path)
    tracks_read_ms = (time.perf_counter() - t0) * 1000.0
    estimator = build_velocity_estimator(meas_cfg["linear_velocity_estimator"])

    camera_results: dict[str, dict] = {}
    all_fits: list[VelocityFitResult] = []
    track_lookup: dict[tuple[str, int], TrackRecord] = {}

    for cam in ("a", "b"):
        print(f"\n{'-'*70}\n处理 {cam.upper()} 路\n{'-'*70}")
        records = records_by_camera[cam]
        print(f"  从阶段3读取 {len(records)} 条轨迹")

        t0 = time.perf_counter()
        fits = [replace(estimator.estimate(rec), spatial_calibration_status=calibration_status)
                for rec in records]
        fit_ms = (time.perf_counter() - t0) * 1000.0

        for rec in records:
            track_lookup[(cam, rec.track_id)] = rec

        all_fits.extend(fits)
        camera_results[cam] = _summarize_camera(
            cam, fits, fit_ms, len(records), vertical_scales[cam]
        )
        _print_camera_summary(cam, camera_results[cam])

    t0 = time.perf_counter()
    matched_rows = (
        _join_stereo_matches(matches_path, all_fits, vertical_scales)
        if matches_path else []
    )
    matches_join_ms = (time.perf_counter() - t0) * 1000.0
    output_dir = _resolve_project_path(meas_cfg["output_dir"])
    step_t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_setup_ms = (time.perf_counter() - step_t0) * 1000.0

    # --- 输出 velocities.csv ------------------------------------------------
    t0 = time.perf_counter()
    velocities_csv = output_dir / "velocities.csv"
    _write_velocities_csv(all_fits, velocities_csv, vertical_scales)
    print(f"\n{velocities_csv} 共 {len(all_fits)} 行")
    _write_matched_velocities_csv(matched_rows, output_dir / "matched_velocities.csv")
    csv_write_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  阶段4匹配轨迹对: {len(matched_rows)}")

    # --- 位置-时间拟合图 -------------------------------------------------------
    t0 = time.perf_counter()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plots_dir.glob("[ab]_track[0-9]*.png"):
        old_plot.unlink()
    n_plot = int(meas_cfg.get("n_plot_samples", 4))
    for cam in ("a", "b"):
        _save_sample_plots(cam, all_fits, track_lookup, plots_dir, n_plot, vertical_scales)
    plots_ms = (time.perf_counter() - t0) * 1000.0

    # --- 报告 -----------------------------------------------------------------
    report_path = output_dir / "stage5_report.json"
    stage_timings_ms = {
        "config_load": config_load_ms,
        "output_directory_setup": output_setup_ms,
        "stage3_tracks_read": tracks_read_ms,
        "velocity_fit_a": camera_results["a"]["fit_module_ms"],
        "velocity_fit_b": camera_results["b"]["fit_module_ms"],
        "velocity_fit_total": sum(s["fit_module_ms"] for s in camera_results.values()),
        "velocity_fit": sum(s["fit_module_ms"] for s in camera_results.values()),
        "stage4_matches_read_and_join": matches_join_ms,
        "csv_write": csv_write_ms,
        "plots": plots_ms,
        "images_read": 0.0,
        "detection_or_tracking_repeated": 0.0,
    }
    report_t0 = time.perf_counter()
    _write_report(
        report_path, camera_results, tracks_path, matches_path, input_counts, matched_rows,
        calibration_path, vertical_scales, calibration_status, calibration_note,
        {**stage_timings_ms, "total_before_report_write": (time.perf_counter() - stage_t0) * 1000.0},
    )
    report_write_ms = (time.perf_counter() - report_t0) * 1000.0
    total_stage_ms = (time.perf_counter() - stage_t0) * 1000.0
    print(f"{report_path}")

    _print_timing_summary({**stage_timings_ms, "report_write": report_write_ms}, total_stage_ms)

    print()
    print("=" * 70)
    print("阶段5完成。速度同时保留原始 px/s，并按 A/B 各自竖直尺度输出 mm/s、m/s。")
    print("匹配置信度和测速置信度分别保留，未进行双路速度融合。")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# 读取阶段3轨迹，关联阶段4轨迹对
# ---------------------------------------------------------------------------


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _load_vertical_scales(path: Path | None) -> tuple[dict[str, float | None], str, str]:
    """读取A/B各自的竖直 mm/px；显式禁用或缺失时不伪造物理速度。"""
    if path is None:
        return {"a": None, "b": None}, "spatial_calibration_not_applied", "未配置尺度文件"
    if not path.exists():
        raise FileNotFoundError(f"尺度标定文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("telecentric_scale_mm_per_px")
    if not isinstance(source, dict):
        raise ValueError(f"尺度标定文件缺少 telecentric_scale_mm_per_px: {path}")
    scales: dict[str, float] = {}
    for cam, key in _CAMERA_SCALE_KEY.items():
        try:
            value = float(source[key]["vertical"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"尺度标定文件缺少 {key}.vertical: {path}") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key}.vertical 必须是正的有限数值，得到 {value!r}")
        scales[cam] = value
    return scales, _SPATIAL_STATUS, f"source={path}"


def _format_scales(vertical_scales: dict[str, float | None]) -> str:
    return ", ".join(
        f"{cam.upper()}={(f'{vertical_scales[cam]:.9f} mm/px' if vertical_scales[cam] is not None else 'unavailable')}"
        for cam in ("a", "b")
    )


def _scaled_velocity(
    velocity_px_per_s: float | None, mm_per_px: float | None,
) -> tuple[float | None, float | None]:
    if velocity_px_per_s is None or mm_per_px is None:
        return None, None
    mm_per_s = float(velocity_px_per_s * mm_per_px)
    return mm_per_s, mm_per_s / 1000.0


def _read_tracks_csv(path: Path) -> tuple[dict[str, list[TrackRecord]], dict]:
    """保留所有真实检测，不额外改变上游QC规则；仅预测点不参与测速。"""
    tracks: dict[tuple[str, int], TrackRecord] = {}
    seen: set[tuple[str, int, int]] = set()
    counts = {"rows_read": 0, "predicted_only_skipped": 0, "real_points_kept": 0}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "track_id", "camera_id", "pair_index", "device_time_s", "detected_cx_px",
            "detected_cy_px", "detected_area_px", "detected_d_eq_px", "association_method",
            "quality_flags",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"tracks.csv 缺少必要列: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            counts["rows_read"] += 1
            cam = row["camera_id"].strip().lower()
            if cam not in ("a", "b"):
                raise ValueError(f"tracks.csv camera_id 必须是 a/b，收到 {cam!r}")
            flags = int(row["quality_flags"] or 0)
            if (row["association_method"] == "predicted_only" or flags & TQC_PREDICTED_ONLY
                    or not (row["detected_cx_px"] or row["detected_cy_px"])):
                counts["predicted_only_skipped"] += 1
                continue
            tid, pidx = int(row["track_id"]), int(row["pair_index"])
            key = (cam, tid, pidx)
            if key in seen:
                raise ValueError(f"轨迹 {cam}:{tid} 在 pair_index={pidx} 有重复检测")
            seen.add(key)
            t, x, y = (float(row[k]) for k in ("device_time_s", "detected_cx_px", "detected_cy_px"))
            if not np.all(np.isfinite([t, x, y])):
                raise ValueError(f"轨迹 {cam}:{tid} 的时间或位置不是有限数值")
            det = Detection(
                frame_idx=pidx, cx=x, cy=y, area_px=float(row["detected_area_px"]),
                bbox=(0, 0, 0, 0),  # 阶段3 CSV不保存包围框；测速不使用此字段。
                d_eq_px=float(row["detected_d_eq_px"]), qc_flags=flags & 0xFF,
            )
            rec = tracks.setdefault((cam, tid), TrackRecord(track_id=tid, camera=cam, status="ended"))
            rec.points.append(TrackPoint(
                track_id=tid, camera=cam, pair_index=pidx, device_time_s=t, detected=det,
                predicted_cx=x, predicted_cy=y, association_method=row["association_method"],
                quality_flags=flags,
            ))
            counts["real_points_kept"] += 1
    # 恢复采集顺序而非时间排序，避免掩盖设备时间倒退。
    for rec in tracks.values():
        rec.points.sort(key=lambda p: p.pair_index)
    return {
        cam: sorted((r for (c, _), r in tracks.items() if c == cam), key=lambda r: r.track_id)
        for cam in ("a", "b")
    }, counts


def _join_stereo_matches(
    path: Path, fits: list[VelocityFitResult], vertical_scales: dict[str, float | None] | None = None,
) -> list[dict]:
    """只关联轨迹级matched行，不使用拒绝候选，也不平均两路像素速度。"""
    lookup = {(f.camera, f.track_id): f for f in fits}
    rows = []
    used_a, used_b = set(), set()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"track_a_id", "track_b_id", "n_common_frames", "status", "confidence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("阶段4输入必须是轨迹级 stereo_matches.csv，缺少轨迹配对字段")
        for row in reader:
            if row["status"] != "matched":
                continue
            a, b = int(row["track_a_id"]), int(row["track_b_id"])
            if ("a", a) not in lookup or ("b", b) not in lookup:
                raise ValueError(f"阶段4匹配 A:{a}/B:{b} 在当前阶段3轨迹中不存在，请核对输入批次")
            if a in used_a or b in used_b:
                raise ValueError(f"阶段4匹配不是一对一: A:{a}/B:{b}")
            used_a.add(a)
            used_b.add(b)
            fa, fb = lookup[("a", a)], lookup[("b", b)]
            result = {
                "track_a_id": a, "track_b_id": b,
                "n_common_frames": int(row["n_common_frames"]),
                "match_confidence": row["confidence"], "match_reason": row.get("reason", ""),
                "both_velocity_available": all(f.velocity_px_per_s is not None for f in (fa, fb)),
                "both_medium_or_higher_confidence": all(f.confidence in ("medium", "high") for f in (fa, fb)),
            }
            for cam, fit in (("a", fa), ("b", fb)):
                mm_per_s, m_per_s = _scaled_velocity(
                    fit.velocity_px_per_s,
                    None if vertical_scales is None else vertical_scales.get(cam),
                )
                result[f"{cam}_vertical_scale_mm_per_px"] = (
                    None if vertical_scales is None else vertical_scales.get(cam)
                )
                result[f"{cam}_velocity_mm_per_s"] = mm_per_s
                result[f"{cam}_velocity_m_per_s"] = m_per_s
                for key in ("n_points", "duration_s", "velocity_px_per_s", "vx_px_per_s",
                            "residual_rms_px", "r_squared", "time_quality", "confidence", "status", "reason"):
                    result[f"{cam}_{key}"] = getattr(fit, key)
            rows.append(result)
    return rows


def _write_matched_velocities_csv(rows: list[dict], path: Path) -> None:
    columns = ["track_a_id", "track_b_id", "n_common_frames", "match_confidence", "match_reason",
               "both_velocity_available", "both_medium_or_higher_confidence"]
    columns += [f"{cam}_{key}" for cam in ("a", "b") for key in (
        "vertical_scale_mm_per_px", "n_points", "duration_s", "velocity_px_per_s",
        "velocity_mm_per_s", "velocity_m_per_s", "vx_px_per_s", "residual_rms_px",
        "r_squared", "time_quality", "confidence", "status", "reason",
    )]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 汇总 / 输出
# ---------------------------------------------------------------------------


def _summarize_camera(
    cam: str, fits: list[VelocityFitResult], fit_ms: float, n_tracks: int,
    vertical_scale_mm_per_px: float | None,
) -> dict:
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
    velocities_mm = (
        [v * vertical_scale_mm_per_px for v in velocities]
        if vertical_scale_mm_per_px is not None else []
    )
    velocities_mm_confident = (
        [v * vertical_scale_mm_per_px for v in velocities_confident]
        if vertical_scale_mm_per_px is not None else []
    )

    return {
        "camera": cam,
        "vertical_scale_mm_per_px": vertical_scale_mm_per_px,
        "n_tracks_total": n_tracks,
        "real_point_count_distribution": dict(sorted(Counter(f.n_points for f in fits).items())),
        "n_velocity_available": len(velocities),
        "n_medium_or_higher_confidence": len(velocities_confident),
        "n_downward_medium_or_higher_confidence": sum(v > 0 for v in velocities_confident),
        "n_fitted_regular_ge3_points": n_regular,
        "n_fitted_two_point": n_two_point,
        "n_insufficient_points_no_velocity": n_insufficient,
        "n_rejected_degenerate_time": n_rejected,
        "confidence_distribution": conf_dist,
        "reason_distribution": dict(Counter(f.reason for f in fits)),
        "velocity_px_per_s_all": _velocity_stats(velocities),
        "velocity_px_per_s_medium_or_higher_confidence": _velocity_stats(velocities_confident),
        "velocity_mm_per_s_all": _velocity_stats(velocities_mm),
        "velocity_mm_per_s_medium_or_higher_confidence": _velocity_stats(velocities_mm_confident),
        "fit_module_ms": fit_ms,
    }


def _velocity_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    arr = np.array(values)
    return {"n": len(arr), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}


def _print_camera_summary(cam: str, s: dict) -> None:
    print(f"  轨迹总数: {s['n_tracks_total']}")
    print(f"  >=3点拟合(含低置信度): {s['n_fitted_regular_ge3_points']}  "
          f"仅2点低置信度: {s['n_fitted_two_point']}  "
          f"点数不足无速度: {s['n_insufficient_points_no_velocity']}  "
          f"时间退化被拒: {s['n_rejected_degenerate_time']}")
    print(f"  置信度分布: {s['confidence_distribution']}")
    print(f"  有速度: {s['n_velocity_available']}  中/高置信度: {s['n_medium_or_higher_confidence']}")
    v_all = s["velocity_px_per_s_all"]
    if v_all["n"]:
        print(f"  速度(px/s，全部) 中位 {v_all['median']:.1f}  范围 [{v_all['min']:.1f}, {v_all['max']:.1f}]")


def _print_timing_summary(timings_ms: dict[str, float], total_stage_ms: float) -> None:
    print("\n" + "-" * 70)
    print("阶段5步骤耗时")
    print("-" * 70)
    labels = (
        ("config_load", "读取配置"),
        ("output_directory_setup", "准备输出目录"),
        ("stage3_tracks_read", "读取阶段3轨迹"),
        ("velocity_fit_a", "A路速度拟合"),
        ("velocity_fit_b", "B路速度拟合"),
        ("velocity_fit_total", "速度拟合合计"),
        ("stage4_matches_read_and_join", "读取并关联阶段4配对"),
        ("csv_write", "写出CSV"),
        ("plots", "生成拟合图"),
        ("report_write", "写出JSON报告"),
    )
    for key, label in labels:
        print(f"  {label:<18}: {timings_ms[key]:10.3f} ms")
    print(f"  {'阶段5总体耗时':<16}: {total_stage_ms:10.3f} ms ({total_stage_ms / 1000.0:.3f} s)")


def _write_velocities_csv(
    fits: list[VelocityFitResult], out_path: Path, vertical_scales: dict[str, float | None],
) -> None:
    columns = [
        "track_id", "camera_id", "n_points", "duration_s",
        "vertical_scale_mm_per_px", "velocity_px_per_s", "velocity_mm_per_s", "velocity_m_per_s",
        "vx_px_per_s", "z0_px",
        "residual_rms_px", "r_squared", "time_quality",
        "spatial_calibration_status", "unit", "confidence", "status", "reason",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in sorted(fits, key=lambda r: (r.camera, r.track_id)):
            mm_per_s, m_per_s = _scaled_velocity(r.velocity_px_per_s, vertical_scales[r.camera])
            w.writerow([
                r.track_id, r.camera, r.n_points,
                "" if r.duration_s is None else f"{r.duration_s:.6f}",
                "" if vertical_scales[r.camera] is None else f"{vertical_scales[r.camera]:.9f}",
                "" if r.velocity_px_per_s is None else f"{r.velocity_px_per_s:.4f}",
                "" if mm_per_s is None else f"{mm_per_s:.6f}",
                "" if m_per_s is None else f"{m_per_s:.9f}",
                "" if r.vx_px_per_s is None else f"{r.vx_px_per_s:.4f}",
                "" if r.z0_px is None else f"{r.z0_px:.4f}",
                "" if r.residual_rms_px is None else f"{r.residual_rms_px:.4f}",
                "" if r.r_squared is None else f"{r.r_squared:.4f}",
                r.time_quality, r.spatial_calibration_status, r.unit,
                r.confidence, r.status, r.reason,
            ])


def _save_sample_plots(
    cam: str, all_fits: list[VelocityFitResult], track_lookup: dict, plots_dir: Path,
    n_plot: int, vertical_scales: dict[str, float | None],
) -> None:
    """优先抽样置信度高、点数多的轨迹，画位置(cy)-时间散点 + 拟合直线。"""
    candidates = [f for f in all_fits if f.camera == cam and f.n_points >= 2]
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    candidates.sort(key=lambda f: (rank[f.confidence], f.n_points), reverse=True)

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
            _, velocity_m_per_s = _scaled_velocity(
                fit.velocity_px_per_s, vertical_scales.get(cam)
            )
            if velocity_m_per_s is None:
                fit_label = f"fit: v={fit.velocity_px_per_s:.1f}px/s (confidence={fit.confidence})"
            else:
                fit_label = (
                    f"fit: v={velocity_m_per_s:.3f}m/s "
                    f"({fit.velocity_px_per_s:.1f}px/s, confidence={fit.confidence})"
                )
            ax.plot(t_line, z_line, color="tab:red", linestyle="--",
                    label=fit_label)
        ax.set_xlabel("device_time_s")
        ax.set_ylabel("cy (px, positive = downward)")
        ax.invert_yaxis()  # image y-axis points down; inverting makes "falling" visually go down
        ax.set_title(f"camera={cam.upper()}  track_id={fit.track_id}  n_points={fit.n_points}  "
                     f"R2={'' if fit.r_squared is None else f'{fit.r_squared:.3f}'}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{cam}_track{fit.track_id}.png", dpi=120)
        plt.close(fig)


def _write_report(
    report_path: Path, camera_results: dict, tracks_path: Path, matches_path: Path | None,
    input_counts: dict, matched_rows: list[dict], calibration_path: Path | None,
    vertical_scales: dict[str, float | None], calibration_status: str,
    calibration_note: str, stage_timings_ms: dict,
) -> None:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_camera": camera_results,
        "input": {
            "tracks_csv": str(tracks_path),
            "stereo_matches_csv": str(matches_path) if matches_path else None,
            **input_counts,
        },
        "stage_timings_ms": stage_timings_ms,
        "spatial_calibration": {
            "status": calibration_status,
            "path": str(calibration_path) if calibration_path else None,
            "vertical_scale_mm_per_px": vertical_scales,
            "note": calibration_note,
            "conversion": "velocity_mm_per_s = velocity_px_per_s * vertical_scale_mm_per_px; "
                          "velocity_m_per_s = velocity_mm_per_s / 1000",
        },
        "time_field_used": {
            "a": "stage3.device_time_s (a_device_relative_us / 1e6)",
            "b": "stage3.device_time_s (b_device_relative_us / 1e6)",
        },
        "stereo_association": {
            "status": "associated" if matches_path else "not_requested",
            "n_matched_track_pairs": len(matched_rows),
            "n_both_velocity_available": sum(r["both_velocity_available"] for r in matched_rows),
            "n_both_medium_or_higher_confidence": sum(
                r["both_medium_or_higher_confidence"] for r in matched_rows
            ),
            "note": "关联的是各路整条轨迹的独立拟合，不限定于共同帧；未融合速度，"
                    "未修正B路倾斜，不能直接把两路原始像素速度之差作为物理一致性判据。"
                    "原阶段4匹配置信度单独保留，配对不提升测速置信度。",
        },
        "unit_and_calibration": {
            "unit": "px/s (raw), mm/s and m/s (scaled)",
            "spatial_calibration_status": calibration_status,
            "note": (
                "A、B使用各自竖直mm/px尺度换算；原始px/s同时保留。"
                "物理结果仍受轨迹关联、定位误差和尺度标定适用性影响。"
            ),
        },
        "stage5_caveats": [
            {
                "item": "not_physical_velocity",
                "detail": "本次已输出按A/B竖直尺度换算的mm/s和m/s，同时保留px/s原始值。",
            },
            {
                "item": "independent_fits_with_stereo_association",
                "detail": "A、B各自拟合，按阶段4轨迹对关联。不会平均两路原始像素速度，"
                          "也不会把低置信度匹配升级成高置信度测量。",
            },
            {
                "item": "kalman_not_used_as_observation",
                "detail": "拟合只使用轨迹里真实检测关联的点（TrackRecord.points，"
                          "阶段3设计上已排除卡尔曼'仅预测'点），"
                          "LinearVelocityEstimator 自身也做了防御性过滤，双重保证。",
            },
            {
                "item": "detection_tracking_upstream_limitation",
                "detail": "测速样本量和可靠性取决于输入轨迹的真实点数和关联质量。"
                          "中高置信度仅按时间、点数、残差和R²分级，未额外按检测QC剔除点。",
            },
            {
                "item": "saved_input_precision",
                "detail": "位置和时间使用阶段3 CSV保存的数值精度；需要核对输入来自同一批次，"
                          "更新阶段3后应重新运行阶段4。阶段五不重读原始设备时间或图像。",
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
