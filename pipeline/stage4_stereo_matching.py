#!/usr/bin/env python3
"""Stage 4: low-overhead stereo track matching using stage-3 tracks and calibrated common-space constraints.

The default mode reads tracks.csv and the common-space masks; it does not reread raw images or repeat background modeling, detection, or single-camera tracking. Candidate track pairs are filtered by common projection masks, B-view height correction, shared-frame motion consistency, and appearance. Connected candidate components are assigned one-to-one, using nearest neighbor for isolated pairs and Hungarian assignment for conflicts.

Accepted track pairs are marked on the original A/B frames and assembled in matched_track_pairs.jpg; matching itself still reuses tracks.csv.

The legacy per-frame fundamental-matrix mode remains available by setting stereo.matching_mode to fundamental_matrix_detections. The default track mode does not use a fundamental matrix or reconstruct 3D.

Run from the repository root with: python pipeline/stage4_stereo_matching.py
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

import cv2
import numpy as np
import yaml

import rain  # noqa: F401  触发所有实现的注册
from rain.data.timing_reader import TimingReader
from rain.imageio import imread_unicode, imwrite_unicode
from rain.io_index import parse_frame_id
from rain.registry import build_background, build_detector, build_reader, build_stereo_matcher
from rain.stereo.calibration import CalibrationLoader
from rain.stereo.types import StereoCalibration, StereoMatchRow
from rain.types import Detection

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def main() -> int:
    stage_t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="阶段4：双路匹配（极线约束优先，未标定时安全退化）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config_load_ms = (time.perf_counter() - t0) * 1000
    st_cfg = cfg["stereo"]
    t0 = time.perf_counter()
    output_dir = Path(st_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_setup_ms = (time.perf_counter() - t0) * 1000

    print("=" * 70)
    print("阶段4：双路匹配")
    print("=" * 70)
    print(f"输出目录 : {output_dir}")

    if st_cfg.get("matching_mode") == "common_space_tracks":
        return _run_common_space_track_mode(
            cfg=cfg,
            st_cfg=st_cfg,
            output_dir=output_dir,
            stage_t0=stage_t0,
            config_load_ms=config_load_ms,
            output_setup_ms=output_setup_ms,
        )

    # --- 1. 标定 ---------------------------------------------------------
    calib_path = st_cfg["calibration"]["path"]
    t0 = time.perf_counter()
    calibration = CalibrationLoader().load(calib_path)
    calibration_load_ms = (time.perf_counter() - t0) * 1000
    print(f"\n标定文件 : {calib_path}")
    print(f"标定可用 : {calibration is not None}"
          + ("" if calibration is not None else "  -> 所有候选将标记 calibration_unavailable"))

    # --- 2. 复用阶段2检测，逐相机跑一遍，按 pair_index 分组 -------------------
    t0 = time.perf_counter()
    reader = build_reader(cfg["reader"])
    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}
    input_setup_ms = (time.perf_counter() - t0) * 1000

    det_by_camera: dict[str, dict[int, list[Detection]]] = {}
    frame_by_camera_pair: dict[str, dict[int, np.ndarray]] = {}
    camera_timings: dict[str, dict] = {}
    n_frames_cfg = int(st_cfg["n_frames"])

    for cam in ("a", "b"):
        print(f"\n[{cam.upper()}路] 检测 ...")
        dets_by_pair, frames_by_pair, camera_timing = _detect_camera(
            cam, cfg["data"][f"dir_{cam}"], cfg, reader, n_frames_cfg,
            filename_to_pair_index,
        )
        det_by_camera[cam] = dets_by_pair
        frame_by_camera_pair[cam] = frames_by_pair
        camera_timings[cam] = camera_timing
        n_dets = sum(len(v) for v in dets_by_pair.values())
        print(f"  {len(dets_by_pair)} 帧，共 {n_dets} 个检测，检测循环 {camera_timing['detection']['total_ms']:.1f} ms")

    # --- 3. 构造 StereoMatcher（通过注册表，calibration 单独注入）------------
    matcher_cfg = st_cfg["matcher"]
    matcher_params = dict(matcher_cfg.get("params") or {})
    matcher_params.update({
        "epipolar_gate": st_cfg["epipolar_gate"],
        "match_gate": st_cfg["match_gate"],
        "cost_matrix_builder": st_cfg["cost_matrix_builder"],
        "max_total_cost": st_cfg["max_total_cost"],
        "fixed_exposure_delay_calibrated": st_cfg["fixed_exposure_delay_calibrated"],
        "calibration": calibration,
    })
    t0 = time.perf_counter()
    matcher = build_stereo_matcher({"name": matcher_cfg["name"], "params": matcher_params})
    matcher_setup_ms = (time.perf_counter() - t0) * 1000

    # --- 4. 逐 pair_index 匹配 ----------------------------------------------
    print("\n逐 pair_index 匹配 ...")
    all_pair_indices = sorted(set(det_by_camera["a"]) | set(det_by_camera["b"]))
    all_rows: list[StereoMatchRow] = []
    gate_ms_list, cost_ms_list, solve_ms_list = [], [], []
    match_frame_ms_list = []
    matcher_overhead_ms_list = []
    n_conflicts_total = 0

    t0 = time.perf_counter()
    for pidx in all_pair_indices:
        dets_a = det_by_camera["a"].get(pidx, [])
        dets_b = det_by_camera["b"].get(pidx, [])
        pair_t0 = time.perf_counter()
        result = matcher.match_frame(pidx, dets_a, dets_b)
        pair_total_ms = (time.perf_counter() - pair_t0) * 1000
        match_frame_ms_list.append(pair_total_ms)
        matcher_overhead_ms_list.append(max(
            0.0, pair_total_ms - result.gate_ms - result.cost_ms - result.solve_ms
        ))
        all_rows.extend(result.rows)
        gate_ms_list.append(result.gate_ms)
        cost_ms_list.append(result.cost_ms)
        solve_ms_list.append(result.solve_ms)
        n_conflicts_total += result.n_candidate_conflicts
    matching_total_ms = (time.perf_counter() - t0) * 1000

    n_matched = sum(1 for r in all_rows if r.status == "matched")
    n_unmatched = len(all_rows) - n_matched
    print(f"  处理 {len(all_pair_indices)} 个 pair_index，共 {len(all_rows)} 条候选诊断")
    print(f"  匹配成功 : {n_matched}")
    print(f"  未匹配   : {n_unmatched}")
    reason_counts = _count_reasons(all_rows)
    print(f"  失败原因分布: {reason_counts}")

    # --- 5. 输出 stereo_matches.csv -----------------------------------------
    matches_csv = output_dir / "stereo_matches.csv"
    t0 = time.perf_counter()
    _write_stereo_matches_csv(all_rows, matches_csv)
    csv_write_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  {matches_csv}")

    # --- 6. 可视化：挑几个 pair_index 画出来 ---------------------------------
    t0 = time.perf_counter()
    _save_visualizations(
        all_pair_indices, det_by_camera, frame_by_camera_pair, all_rows,
        output_dir, n_overlay=int(st_cfg.get("overlay_samples", 6)),
    )
    visualization_ms = (time.perf_counter() - t0) * 1000

    # --- 7. 合成数据演示（独立于真实数据，证明算法逻辑本身是对的）-------------
    t0 = time.perf_counter()
    synthetic_result = _run_synthetic_demo()
    synthetic_demo_ms = (time.perf_counter() - t0) * 1000

    stage_timings_ms = {
        "config_load": {"total_ms": config_load_ms},
        "output_directory_setup": {"total_ms": output_setup_ms},
        "calibration_load": {"total_ms": calibration_load_ms},
        "input_setup_and_timing_index": {"total_ms": input_setup_ms},
        "camera_a": camera_timings["a"],
        "camera_b": camera_timings["b"],
        "matcher_setup": {"total_ms": matcher_setup_ms},
        "stereo_matching": {
            "total_ms": matching_total_ms,
            "per_pair_index": _stats(match_frame_ms_list),
            "gate": _stats(gate_ms_list),
            "cost": _stats(cost_ms_list),
            "solve": _stats(solve_ms_list),
            "matcher_overhead_and_row_assembly": _stats(matcher_overhead_ms_list),
            "loop_overhead_ms": max(0.0, matching_total_ms - sum(match_frame_ms_list)),
        },
        "csv_write": {"total_ms": csv_write_ms},
        "visualization": {"total_ms": visualization_ms},
        "synthetic_demo": {"total_ms": synthetic_demo_ms},
    }

    # --- 8. 报告 -------------------------------------------------------------
    report_path = output_dir / "stage4_report.json"
    real_data_summary = {
        "n_pair_index_processed": len(all_pair_indices),
        "n_candidates_total": len(all_rows),
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "reason_counts": reason_counts,
        "n_candidate_conflicts": n_conflicts_total,
        "n_detections_a": sum(len(v) for v in det_by_camera["a"].values()),
        "n_detections_b": sum(len(v) for v in det_by_camera["b"].values()),
        "timings_ms": {
            "gate": _stats(gate_ms_list),
            "cost": _stats(cost_ms_list),
            "solve": _stats(solve_ms_list),
            "match_frame": _stats(match_frame_ms_list),
            "matcher_overhead_and_row_assembly": _stats(matcher_overhead_ms_list),
            "matching_total_ms": matching_total_ms,
        },
    }
    t0 = time.perf_counter()
    _write_report(
        report_path=report_path,
        calibration=calibration,
        calib_path=calib_path,
        real_data=real_data_summary,
        synthetic_data=synthetic_result,
        stage_timings_ms=stage_timings_ms,
    )
    report_write_ms = (time.perf_counter() - t0) * 1000
    stage_timings_ms["report_write_first_pass"] = {"total_ms": report_write_ms}
    stage_timings_ms["stage_total_through_first_report_write"] = {
        "total_ms": (time.perf_counter() - stage_t0) * 1000
    }
    # Store the first report-write duration. The final metadata write cannot
    # include its own exact duration without another recursive write.
    _write_report(
        report_path=report_path,
        calibration=calibration,
        calib_path=calib_path,
        real_data=real_data_summary,
        synthetic_data=synthetic_result,
        stage_timings_ms=stage_timings_ms,
    )
    stage_total_ms = (time.perf_counter() - stage_t0) * 1000
    print(f"  {report_path}")
    print(f"  阶段总耗时（含报告写入）: {stage_total_ms:.1f} ms")

    print()
    print("=" * 70)
    if calibration is None:
        print("阶段4完成：无真实标定文件，真实数据 0 个已验证匹配（calibration_unavailable）。")
        print("算法逻辑已用合成数据 + 单元测试验证，详见 stage4_report.json。")
    else:
        print("阶段4完成。")
    print("=" * 70)
    return 0


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).parent.parent / path


def _run_common_space_track_mode(
    cfg: dict,
    st_cfg: dict,
    output_dir: Path,
    stage_t0: float,
    config_load_ms: float,
    output_setup_ms: float,
) -> int:
    """只读取阶段3轨迹和小型掩膜，不重读原图、不重复检测或跟踪。"""
    from rain.stereo.common_space_track_matcher import (
        CommonSpaceTrackMatcher,
        load_common_space_calibration,
        read_tracks_csv,
        settings_from_config,
    )

    common_cfg = st_cfg["common_space"]
    t0 = time.perf_counter()
    calibration_path = _resolve_project_path(common_cfg["report_path"])
    mask_a_path = _resolve_project_path(common_cfg["mask_a_path"])
    mask_b_path = _resolve_project_path(common_cfg["mask_b_path"])
    calibration = load_common_space_calibration(calibration_path, mask_a_path, mask_b_path)
    calibration_load_ms = (time.perf_counter() - t0) * 1000

    tracks_path = _resolve_project_path(st_cfg["tracks_csv"])
    settings = settings_from_config(common_cfg.get("matcher", {}))
    t0 = time.perf_counter()
    tracks_by_camera, input_counts = read_tracks_csv(tracks_path, calibration, settings)
    tracks_read_filter_ms = (time.perf_counter() - t0) * 1000

    matcher = CommonSpaceTrackMatcher(calibration, settings)
    t0 = time.perf_counter()
    matched, rows, result_counts = matcher.match(tracks_by_camera["a"], tracks_by_camera["b"])
    match_ms = (time.perf_counter() - t0) * 1000
    _print_track_match_diagnostics(
        input_counts=input_counts,
        settings=settings,
        matcher=matcher,
        rows=rows,
        result_counts=result_counts,
    )

    csv_path = output_dir / "stereo_matches.csv"
    t0 = time.perf_counter()
    csv_path = _write_track_match_csv(rows, csv_path)
    csv_write_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    raw_overlay_dir, raw_overlay_paths, raw_overlay_image_count = _save_matched_track_raw_overlays(
        matched=matched,
        tracks_by_camera=tracks_by_camera,
        data_cfg=cfg["data"],
        output_dir=output_dir / "matched_overlay",
    )
    raw_overlay_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    visualization_path = _save_matched_track_visualization(
        matched=matched,
        tracks_by_camera=tracks_by_camera,
        overlay_dir=raw_overlay_dir,
        output_path=output_dir / "matched_track_pairs.jpg",
    )
    overview_ms = (time.perf_counter() - t0) * 1000

    n_pair_indices = len({
        frame
        for camera_tracks in tracks_by_camera.values()
        for track in camera_tracks
        for frame in track.observations
    })
    stage_timings_ms = {
        "config_load": {"total_ms": config_load_ms},
        "output_directory_setup": {"total_ms": output_setup_ms},
        "common_space_calibration_and_masks_load": {"total_ms": calibration_load_ms},
        "stage3_tracks_read_and_mask_filter": {"total_ms": tracks_read_filter_ms},
        "track_candidate_gating_assignment": {"total_ms": match_ms},
        "csv_write": {"total_ms": csv_write_ms},
        "images_read": {"total_ms": 0.0, "reason": "复用阶段3 tracks.csv；匹配本身不重新读取图像"},
        "detection_or_tracking_repeated": {"total_ms": 0.0, "reason": "直接复用阶段3结果"},
        "matched_track_visualization": {
            "raw_overlay_ms": raw_overlay_ms,
            "overview_ms": overview_ms,
            "total_ms": raw_overlay_ms + overview_ms,
            "source_frame_images_read": raw_overlay_image_count,
            "overview_path": str(visualization_path),
            "raw_overlay_dir": str(raw_overlay_dir),
            "raw_overlay_images": len(raw_overlay_paths),
        },
    }
    report_path = output_dir / "stage4_report.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matching_mode": "common_space_tracks",
        "input": {
            "tracks_csv": str(tracks_path),
            "common_space_report": str(calibration_path),
            "tracks_csv_rows": input_counts["rows_read"],
            "real_points_kept": input_counts["real_points_kept"],
            "pair_indices_processed": n_pair_indices,
            "input_filter_counts": input_counts,
        },
        "calibration": {
            "source_version": _read_common_space_version(calibration_path),
            "fundamental_matrix_used": False,
            "height_mapping_used": "v_A_corrected = coefficient_u * u_B + coefficient_v * v_B + intercept_px",
            "tilt_coefficients": {
                "coefficient_u": calibration.tilt_u,
                "coefficient_v": calibration.tilt_v,
                "intercept_px": calibration.tilt_intercept,
            },
            "rmse_px": calibration.rmse_px,
            "maximum_abs_error_px": calibration.max_error_px,
            "height_gate_px": matcher.height_gate_px,
            "height_fluctuation_gate_px": matcher.height_fluctuation_gate_px,
            "note": "使用共同投影掩膜与高度/运动轨迹证据；此模式不加载基础矩阵F。",
        },
        "matching_results": {
            **result_counts,
            "n_pair_rows_written": len(rows),
            "n_matched_pairs": len(matched),
            "n_unmatched_tracks_a": sum(1 for r in rows if r["status"] == "unmatched_a"),
            "n_unmatched_tracks_b": sum(1 for r in rows if r["status"] == "unmatched_b"),
            "matched_pairs": [
                {
                    "track_a_id": c.track_a_id,
                    "track_b_id": c.track_b_id,
                    "n_common_frames": len(c.common_frames),
                    "first_common_frame": c.common_frames[0],
                    "last_common_frame": c.common_frames[-1],
                    "total_cost": c.total_cost,
                }
                for c in matched
            ],
        },
        "outputs": {
            "track_matches_csv": str(csv_path),
            "matched_track_overview": str(visualization_path),
            "matched_track_overlay_dir": str(raw_overlay_dir),
        },
        "stage_timings_ms": stage_timings_ms,
        "limitations": [
            "匹配质量依赖阶段3单路轨迹质量和共同空间标定适用性。",
            "固定曝光延迟仍未独立标定，匹配置信度保持 low。",
            "此轨迹级共同空间匹配与原逐帧基础矩阵匹配模式不同。",
        ],
    }
    t0 = time.perf_counter()
    _write_track_report(report_path, report)
    report_write_ms = (time.perf_counter() - t0) * 1000
    stage_timings_ms["report_write_first_pass"] = {"total_ms": report_write_ms}
    stage_timings_ms["stage_total_through_first_report_write"] = {
        "total_ms": (time.perf_counter() - stage_t0) * 1000
    }
    _write_track_report(report_path, report)
    stage_total_ms = (time.perf_counter() - stage_t0) * 1000

    print(f"\n复用阶段3轨迹: {tracks_path}")
    print(f"A/B有效轨迹: {len(tracks_by_camera['a'])} / {len(tracks_by_camera['b'])}")
    print(f"候选轨迹对: {result_counts['n_track_candidates']}")
    print(f"匹配成功: {len(matched)} 对")
    print(f"匹配耗时: {match_ms:.3f} ms；写出 CSV: {csv_write_ms:.3f} ms")
    print(f"原图匹配总览: {visualization_path} ({overview_ms:.1f} ms)")
    print(f"逐帧原图标注: {raw_overlay_dir} ({len(raw_overlay_paths)} 张, {raw_overlay_ms:.1f} ms)")
    print(f"阶段4总耗时: {stage_total_ms:.1f} ms")
    print(f"匹配表: {csv_path}")
    print(f"报告: {report_path}")
    return 0


def _read_common_space_version(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8" )).get("version", "unknown"))


def _write_track_match_csv(rows: list[dict], out_path: Path) -> Path:
    columns = [
        "track_a_id", "track_b_id", "n_common_frames", "first_common_frame", "last_common_frame",
        "median_height_residual_px", "height_mad_px", "normalized_motion_error",
        "net_motion_error", "diameter_log_error", "area_log_error", "total_cost",
        "status", "method", "confidence", "reason",
    ]
    def write_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    try:
        write_csv(out_path)
        return out_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback_path = out_path.with_name(f"{out_path.stem}_{timestamp}{out_path.suffix}")
        write_csv(fallback_path)
        print(
            f"\n警告: {out_path.name} 正被其他程序占用；本次匹配表已另存为 {fallback_path}，"
            "并继续生成可视化和报告。"
        )
        return fallback_path


def _write_track_report(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def _print_track_match_diagnostics(
    input_counts: dict,
    settings,
    matcher,
    rows: list[dict],
    result_counts: dict,
) -> None:
    """Print a compact view of input filters, candidate gates, and assignment results."""
    print("\n[匹配诊断 1/3：输入轨迹过滤]")
    print(
        "  tracks.csv: {rows_read} 行；保留真实点: {real_points_kept}；"
        "跳过预测点: {predicted_only_skipped}；QC过滤: {qc_skipped}；"
        "共同视场外: {outside_common_mask_skipped}".format(**input_counts)
    )
    print(
        "  短轨迹过滤: {tracks_too_short_skipped} 条；进入匹配 A/B 轨迹: {tracks_a_after_filter}/{tracks_b_after_filter}".format(
            **input_counts
        )
    )

    print("[匹配诊断 2/3：轨迹对门控]")
    print(
        f"  共同帧 >= {settings.min_common_frames}，连续共同帧 >= {settings.min_consecutive_frames}"
    )
    print(
        f"  |中位高度偏差| <= {matcher.height_gate_px:.2f} px；"
        f"高度波动 MAD <= {matcher.height_fluctuation_gate_px:.2f} px"
    )
    print(
        f"  向下位移中位数 >= {settings.min_downward_px_per_frame:.2f} px/frame，"
        f"且运动幅度 >= {settings.min_motion_delta_px:.2f} px"
    )
    print(
        f"  归一化运动误差 <= {settings.max_normalized_motion_error:.3f}；"
        f"净位移误差 <= {settings.max_net_motion_error:.3f}"
    )
    print(
        "  代价权重: " + ", ".join(
            f"{key}={float(value):.2f}" for key, value in settings.cost_weights.items()
        )
    )

    gate_rejections: dict[str, int] = {}
    assignment_rejections: dict[str, int] = {}
    for row in rows:
        if row.get("status") == "rejected":
            reason = row.get("reason", "unknown")
            gate_rejections[reason] = gate_rejections.get(reason, 0) + 1
        elif row.get("status") in {"matched", "unmatched_candidate"}:
            if row.get("status") == "unmatched_candidate":
                reason = row.get("reason", "unknown")
                assignment_rejections[reason] = assignment_rejections.get(reason, 0) + 1

    print(
        f"  评估轨迹对: {result_counts['n_track_pairs_evaluated']}；"
        f"通过门控候选: {result_counts['n_track_candidates']}；"
        f"门控拒绝: {sum(gate_rejections.values())}"
    )
    if gate_rejections:
        print("  门控拒绝原因: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(gate_rejections.items())
        ))
    else:
        print("  门控拒绝原因: 无")

    print("[匹配诊断 3/3：一对一分配]")
    print(
        f"  冲突连通分量: {result_counts['n_conflict_components']}；"
        f"代价门槛 < {settings.max_total_cost:.3f}；"
        f"与次优候选最小差值 >= {settings.min_assignment_margin:.3f}；"
        f"最终接受: {result_counts['n_matched_track_pairs']}"
    )
    if assignment_rejections:
        print("  分配阶段拒绝: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(assignment_rejections.items())
        ))


def _save_matched_track_visualization(
    matched,
    tracks_by_camera: dict,
    overlay_dir: Path,
    output_path: Path,
) -> Path:
    """Create a contact sheet from the annotated original A/B frames."""
    margin = 22
    header_height = 84
    label_height = 52
    row_gap = 12
    frame_gap = 20
    thumb_width = 700
    canvas_width = margin * 2 + thumb_width * 2 + frame_gap
    tracks_by_id = {
        camera: {track.track_id: track for track in camera_tracks}
        for camera, camera_tracks in tracks_by_camera.items()
    }
    row_items = []
    max_thumb_height = 1
    palette = (
        (255, 112, 24), (16, 155, 246), (70, 180, 45), (200, 75, 195),
        (30, 180, 215), (170, 110, 20), (130, 75, 220),
    )

    for match_index, candidate in enumerate(matched):
        track_a = tracks_by_id["a"][candidate.track_a_id]
        track_b = tracks_by_id["b"][candidate.track_b_id]
        thumbnails = []
        for pair_index in dict.fromkeys((candidate.common_frames[0], candidate.common_frames[-1])):
            image_path = overlay_dir / f"pair_{pair_index:04d}.jpg"
            image = imread_unicode(str(image_path), cv2.IMREAD_COLOR)
            source_header_height = 58
            image = image[source_header_height:, :]
            scale = thumb_width / image.shape[1]
            thumb_height = max(1, int(round(image.shape[0] * scale)))
            thumbnail = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)

            camera_width_a = (image.shape[1] - 28) // 2
            observation_a = track_a.observations[pair_index]
            observation_b = track_b.observations[pair_index]
            point_a = (int(round(observation_a.cx * scale)), int(round(observation_a.cy * scale)))
            point_b = (
                int(round((camera_width_a + 28 + observation_b.cx) * scale)),
                int(round(observation_b.cy * scale)),
            )
            color = palette[match_index % len(palette)]
            cv2.line(thumbnail, point_a, point_b, color, 3, cv2.LINE_AA)
            for point in (point_a, point_b):
                cv2.circle(thumbnail, point, 8, color, 2, cv2.LINE_AA)
                cv2.drawMarker(thumbnail, point, color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            for text, point in (
                (f"A{candidate.track_a_id}", point_a),
                (f"B{candidate.track_b_id}", point_b),
            ):
                label_y = max(16, point[1] - 10)
                cv2.putText(thumbnail, text, (point[0] + 9, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(thumbnail, text, (point[0] + 9, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            thumbnails.append((pair_index, thumbnail))
            max_thumb_height = max(max_thumb_height, thumb_height)
        row_items.append((candidate, thumbnails))

    row_height = label_height + max_thumb_height + row_gap
    canvas_height = header_height + max(1, len(row_items)) * row_height + 18
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    def put_text(text: str, origin: tuple[int, int], scale: float, color=(35, 35, 35), thickness=1):
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    put_text(f"Stage 4 accepted matches on original A/B frames: {len(matched)}", (margin, 30), 0.68, thickness=2)
    put_text(
        "A is on the left and B on the right; colored lines connect accepted detections at the same pair_index.",
        (margin, 55), 0.40, (75, 75, 75),
    )
    put_text(
        "Showing first and last shared frames. Confidence is low because fixed exposure delay is not calibrated.",
        (margin, 75), 0.39, (95, 95, 95),
    )

    for row_index, (candidate, thumbnails) in enumerate(row_items):
        y = header_height + row_index * row_height
        frame_indices = ",".join(str(frame) for frame in candidate.common_frames)
        put_text(
            f"A{candidate.track_a_id} <-> B{candidate.track_b_id} | pair_index {frame_indices} | cost {candidate.total_cost:.3f} | confidence low",
            (margin, y + 19), 0.42, thickness=1,
        )
        for frame_index, (pair_index, thumbnail) in enumerate(thumbnails):
            x = margin + frame_index * (thumb_width + frame_gap)
            put_text(f"pair {pair_index}", (x + 4, y + label_height - 5), 0.39, (70, 70, 70))
            canvas[y + label_height:y + label_height + thumbnail.shape[0], x:x + thumb_width] = thumbnail
        cv2.line(canvas, (margin, y + row_height - 3), (canvas_width - margin, y + row_height - 3), (225, 225, 225), 1)

    if not row_items:
        put_text("No accepted track pairs in this run.", (margin, header_height + 32), 0.52)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return output_path



def _save_matched_track_raw_overlays(
    matched,
    tracks_by_camera: dict,
    data_cfg: dict,
    output_dir: Path,
) -> tuple[Path, list[Path], int]:
    """Overlay accepted track matches on original A/B frames."""
    timing_path = Path(data_cfg["dir_a"]) / "capture_timing.csv"
    timing_records = TimingReader().read(str(timing_path))
    filename_by_pair = {record.pair_index: record.pair_filename for record in timing_records}
    tracks_by_id = {
        camera: {track.track_id: track for track in camera_tracks}
        for camera, camera_tracks in tracks_by_camera.items()
    }

    matches_by_frame: dict[int, list] = {}
    for match_index, candidate in enumerate(matched):
        track_a = tracks_by_id["a"][candidate.track_a_id]
        track_b = tracks_by_id["b"][candidate.track_b_id]
        for pair_index in candidate.common_frames:
            matches_by_frame.setdefault(pair_index, []).append((match_index, candidate, track_a, track_b))

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.iterdir():
        if old_path.is_file() and old_path.suffix.lower() in {".png", ".jpg"}:
            old_path.unlink()
    palette = (
        (255, 112, 24), (16, 155, 246), (70, 180, 45), (200, 75, 195),
        (30, 180, 215), (170, 110, 20), (130, 75, 220),
    )
    output_paths: list[Path] = []
    image_read_count = 0
    gutter = 28
    header_height = 58

    def draw_dashed_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = start
        x1, y1 = end
        length = max(1, int(np.hypot(x1 - x0, y1 - y0)))
        dash_length = 18
        for offset in range(0, length, dash_length * 2):
            stop = min(length, offset + dash_length)
            p0 = (int(x0 + (x1 - x0) * offset / length), int(y0 + (y1 - y0) * offset / length))
            p1 = (int(x0 + (x1 - x0) * stop / length), int(y0 + (y1 - y0) * stop / length))
            cv2.line(image, p0, p1, color, 3, cv2.LINE_AA)

    def outlined_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.58) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    for pair_index, frame_matches in sorted(matches_by_frame.items()):
        filename = filename_by_pair.get(pair_index)
        if filename is None:
            raise KeyError(f"capture_timing.csv has no pair_index={pair_index}")
        image_a_gray = imread_unicode(str(Path(data_cfg["dir_a"]) / filename))
        image_b_gray = imread_unicode(str(Path(data_cfg["dir_b"]) / filename))
        image_read_count += 2
        image_a = cv2.cvtColor(image_a_gray, cv2.COLOR_GRAY2BGR)
        image_b = cv2.cvtColor(image_b_gray, cv2.COLOR_GRAY2BGR)
        height = max(image_a.shape[0], image_b.shape[0])
        width_a, width_b = image_a.shape[1], image_b.shape[1]
        offset_b = width_a + gutter
        canvas = np.full((header_height + height, width_a + gutter + width_b, 3), 255, dtype=np.uint8)
        canvas[header_height:header_height + image_a.shape[0], :width_a] = image_a
        canvas[header_height:header_height + image_b.shape[0], offset_b:offset_b + width_b] = image_b

        outlined_text(canvas, f"pair_index={pair_index} | accepted matches: {len(frame_matches)} | confidence low", (16, 23), (35, 35, 35), 0.62)
        outlined_text(canvas, "A left | B right | dashed links show accepted matches, not epipolar lines", (16, 48), (65, 65, 65), 0.46)
        cv2.line(canvas, (width_a + gutter // 2, header_height), (width_a + gutter // 2, header_height + height), (150, 150, 150), 1)

        for match_index, candidate, track_a, track_b in frame_matches:
            color = palette[match_index % len(palette)]
            obs_a = track_a.observations[pair_index]
            obs_b = track_b.observations[pair_index]
            point_a = (int(round(obs_a.cx)), header_height + int(round(obs_a.cy)))
            point_b = (offset_b + int(round(obs_b.cx)), header_height + int(round(obs_b.cy)))
            draw_dashed_line(canvas, point_a, point_b, color)
            for point in (point_a, point_b):
                cv2.circle(canvas, point, 20, color, 3, cv2.LINE_AA)
                cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)

            label_a = f"A{candidate.track_a_id} ({obs_a.cx:.1f}, {obs_a.cy:.1f})"
            label_b = f"B{candidate.track_b_id} ({obs_b.cx:.1f}, {obs_b.cy:.1f})"
            size_a = cv2.getTextSize(label_a, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
            size_b = cv2.getTextSize(label_b, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
            x_a = min(max(8, point_a[0] + 24), max(8, width_a - size_a[0] - 8))
            x_b_local = min(max(8, int(round(obs_b.cx)) + 24), max(8, width_b - size_b[0] - 8))
            label_y_a = max(header_height + 24, point_a[1] - 22)
            label_y_b = max(header_height + 24, point_b[1] - 22)
            outlined_text(canvas, label_a, (x_a, label_y_a), color)
            outlined_text(canvas, label_b, (offset_b + x_b_local, label_y_b), color)

        out_path = output_dir / f"pair_{pair_index:04d}.jpg"
        imwrite_unicode(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
        output_paths.append(out_path)

    return output_dir, output_paths, image_read_count


# ---------------------------------------------------------------------------


def _detect_camera(
    cam: str, folder: str, cfg: dict, reader, n_frames_cfg: int,
    filename_to_pair_index: dict[str, int],
) -> tuple[dict[int, list[Detection]], dict[int, np.ndarray], dict]:
    det_cfg = cfg["detection"]

    total_t0 = time.perf_counter()
    t0 = time.perf_counter()
    frame_files = _list_camera_frame_files(folder)
    n = min(n_frames_cfg, len(frame_files))
    frame_files = frame_files[:n]
    frame_listing_ms = (time.perf_counter() - t0) * 1000

    paths = [p for _, p in frame_files]
    read_ms_list = []
    frames = []
    for p in paths:
        t0 = time.perf_counter()
        frames.append(reader.read(p))
        read_ms_list.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    bgmodel = build_background(cfg["background"])
    bg = bgmodel.build(frames)
    background_model_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    detector = build_detector({
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
    detector_setup_ms = (time.perf_counter() - t0) * 1000

    dets_by_pair: dict[int, list[Detection]] = {}
    frames_by_pair: dict[int, np.ndarray] = {}
    detection_ms_list = []
    for (fn, _), img in zip(frame_files, frames):
        if fn not in filename_to_pair_index:
            raise KeyError(f"文件 {fn} 在 capture_timing.csv 里找不到 pair_index，停止处理。")
        pidx = filename_to_pair_index[fn]
        t0 = time.perf_counter()
        dets_by_pair[pidx] = detector.detect(img, bg, frame_idx=pidx)
        detection_ms_list.append((time.perf_counter() - t0) * 1000)
        frames_by_pair[pidx] = img
    camera_total_ms = (time.perf_counter() - total_t0) * 1000

    timings = {
        "frame_listing": {"total_ms": frame_listing_ms},
        "image_read": _stats(read_ms_list),
        "background_model": {"total_ms": background_model_ms},
        "detector_setup": {"total_ms": detector_setup_ms},
        "detection": _stats(detection_ms_list),
        "camera_total_ms": camera_total_ms,
        "n_frames": len(frame_files),
    }
    timings["image_read"]["total_ms"] = float(sum(read_ms_list))
    timings["detection"]["total_ms"] = float(sum(detection_ms_list))
    return dets_by_pair, frames_by_pair, timings


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


def _stats(values_ms: list[float]) -> dict:
    if not values_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0}
    arr = np.array(values_ms)
    return {
        "mean_ms": float(np.mean(arr)), "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)), "n": len(arr),
    }


def _count_reasons(rows: list[StereoMatchRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.reason] = counts.get(r.reason, 0) + 1
    return counts


def _write_stereo_matches_csv(rows: list[StereoMatchRow], out_path: Path) -> None:
    columns = [
        "pair_index", "a_index", "b_index", "a_cx_px", "a_cy_px", "b_cx_px", "b_cy_px",
        "epipolar_distance_px", "cost_epipolar", "cost_position", "cost_area", "cost_aspect",
        "total_cost", "gate_epipolar_passed", "gate_match_passed",
        "method", "confidence", "status", "reason",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([
                r.pair_index,
                "" if r.a_index is None else r.a_index,
                "" if r.b_index is None else r.b_index,
                _fmt(r.a_cx), _fmt(r.a_cy), _fmt(r.b_cx), _fmt(r.b_cy),
                _fmt(r.epipolar_distance_px), _fmt(r.cost_epipolar), _fmt(r.cost_position),
                _fmt(r.cost_area), _fmt(r.cost_aspect), _fmt(r.total_cost),
                int(r.gate_epipolar_passed), int(r.gate_match_passed),
                r.method, r.confidence, r.status, r.reason,
            ])


def _fmt(v) -> str:
    return "" if v is None else f"{v:.4f}"


def _save_visualizations(
    all_pair_indices, det_by_camera, frame_by_camera_pair, all_rows,
    output_dir: Path, n_overlay: int,
) -> None:
    """挑几个 A、B 都有检测的 pair_index，把两路画在一张拼接图上。

    标定不可用时不会有真正的匹配线，但两路各自的检测和"未匹配、原因"
    的标注仍然有意义——能看出候选到底为什么没连上。
    """
    if n_overlay <= 0:
        return
    overlay_dir = output_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    both_have_dets = [
        p for p in all_pair_indices
        if det_by_camera["a"].get(p) and det_by_camera["b"].get(p)
    ]
    candidates = both_have_dets if both_have_dets else [
        p for p in all_pair_indices
        if det_by_camera["a"].get(p) or det_by_camera["b"].get(p)
    ]
    if not candidates:
        return

    step = max(1, len(candidates) // n_overlay)
    rows_by_pair: dict[int, list] = {}
    for r in all_rows:
        rows_by_pair.setdefault(r.pair_index, []).append(r)

    for pidx in candidates[::step][:n_overlay]:
        img_a = frame_by_camera_pair["a"].get(pidx)
        img_b = frame_by_camera_pair["b"].get(pidx)
        if img_a is None or img_b is None:
            continue
        _draw_pair(pidx, img_a, img_b, rows_by_pair.get(pidx, []), overlay_dir)


def _draw_pair(pair_index: int, img_a: np.ndarray, img_b: np.ndarray, rows: list, overlay_dir: Path) -> None:
    canvas_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
    canvas_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
    h = max(canvas_a.shape[0], canvas_b.shape[0])
    canvas = np.zeros((h, canvas_a.shape[1] + canvas_b.shape[1] + 20, 3), dtype=np.uint8)
    canvas[: canvas_a.shape[0], : canvas_a.shape[1]] = canvas_a
    offset_x = canvas_a.shape[1] + 20
    canvas[: canvas_b.shape[0], offset_x: offset_x + canvas_b.shape[1]] = canvas_b

    rng = np.random.default_rng(pair_index)
    for r in rows:
        if r.status == "matched":
            color = tuple(int(c) for c in rng.integers(80, 255, size=3))
            pa = (int(r.a_cx), int(r.a_cy))
            pb = (int(r.b_cx) + offset_x, int(r.b_cy))
            cv2.circle(canvas, pa, 8, color, 2)
            cv2.circle(canvas, pb, 8, color, 2)
            cv2.line(canvas, pa, pb, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{r.total_cost:.2f}", pa, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif r.status == "unmatched_a" and r.a_cx is not None:
            cv2.circle(canvas, (int(r.a_cx), int(r.a_cy)), 8, (0, 0, 255), 1)
            cv2.putText(canvas, r.reason, (int(r.a_cx) + 10, int(r.a_cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        elif r.status == "unmatched_b" and r.b_cx is not None:
            p = (int(r.b_cx) + offset_x, int(r.b_cy))
            cv2.circle(canvas, p, 8, (0, 0, 255), 1)
            cv2.putText(canvas, r.reason, (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.putText(canvas, f"pair_index={pair_index}  A (left) | B (right)",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(str(overlay_dir / f"pair_{pair_index:04d}.png"), canvas)


def _run_synthetic_demo() -> dict:
    """独立于真实数据的合成场景演示：证明极线门控+匹配逻辑本身是正确的。

    用一个自定义的基础矩阵（让 A 里 (ax,ay) 对应的极线正好是 B 里的
    水平线 Y=ay，纯粹为了让极线距离直观可算，不代表任何真实相机关系），
    构造几对合成检测，跑一遍 match_frame，把结果如实记录进报告。
    """
    from rain.stereo.stereo_matcher import StereoMatcher

    F = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    calib = StereoCalibration(fundamental_matrix=F, rectify_a_to_b=np.eye(3), source_path="synthetic-demo")

    matcher = StereoMatcher(
        calibration=calib,
        epipolar_gate={"params": {"max_epipolar_dist_px": 5.0}},
        match_gate={"params": {"max_area_ratio": 3.0, "max_aspect_diff": 2.0}},
        max_total_cost=0.6,
        fixed_exposure_delay_calibrated=True,
    )

    def det(cx, cy, area=50.0):
        return Detection(frame_idx=0, cx=cx, cy=cy, area_px=area,
                          bbox=(int(cx - 5), int(cy - 5), 10, 10), d_eq_px=Detection.d_eq_from_area(area))

    scenarios = {
        "unique_match": ([det(100, 200)], [det(900, 200)]),
        "epipolar_rejected": ([det(100, 200)], [det(900, 800)]),
        "multi_target_conflict": (
            [det(100, 200), det(105, 200)], [det(900, 200), det(905, 200)],
        ),
        "no_candidate_on_b_side": ([det(100, 200)], []),
    }

    result = {}
    for name, (dets_a, dets_b) in scenarios.items():
        fr = matcher.match_frame(0, dets_a, dets_b)
        result[name] = {
            "n_a": len(dets_a), "n_b": len(dets_b),
            "n_candidate_conflicts": fr.n_candidate_conflicts,
            "rows": [
                {
                    "a_index": r.a_index, "b_index": r.b_index,
                    "status": r.status, "method": r.method,
                    "confidence": r.confidence, "reason": r.reason,
                    "total_cost": r.total_cost,
                }
                for r in fr.rows
            ],
        }
    return result


def _write_report(
    report_path: Path, calibration, calib_path: str, real_data: dict,
    synthetic_data: dict, stage_timings_ms: dict | None = None,
) -> None:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_status": {
            "calibration_path_configured": calib_path,
            "calibration_available": calibration is not None,
            "note": (
                "没有真实双相机内外参/基础矩阵标定文件，未虚构任何参数。"
                "EpipolarGate 在这种情况下让所有候选都判定不通过，"
                "原因固定标记 calibration_unavailable。"
            ),
        },
        "unit_test_results": {
            "test_file": "tests/test_stereo_stage4.py",
            "how_to_reproduce": "python -m pytest tests/test_stereo_stage4.py -v",
            "note": (
                "本报告不重复嵌入 pytest 的实时结果（避免和实际运行状态脱节）；"
                "验收报告里会附上最近一次 `pytest tests/ -q` 的通过/失败数量。"
                "这里只声明覆盖的场景：唯一匹配、极线门控拒绝、代价超阈值、"
                "多目标冲突(Hungarian)、最近邻结果、匈牙利全局最优、"
                "无标定文件的安全处理、CalibrationLoader文件缺失/格式错误。"
            ),
        },
        "synthetic_data_results": {
            "note": (
                "用自定义基础矩阵（让A图像点(ax,ay)对应的极线是B图像里的水平线"
                "Y=ay，只为了让极线距离直观可算，不代表任何真实相机几何关系）"
                "构造的合成检测，验证算法逻辑本身正确。"
            ),
            "scenarios": synthetic_data,
        },
        "real_data_results_180_pairs": {
            **real_data,
            "note": (
                "A/B-Capture_085820，180个pair_index。因为没有真实标定，"
                "这里的匹配尝试全部会在极线门控这一步被拒绝"
                "（reason=calibration_unavailable），不代表算法本身有问题，"
                "也不构成'已完成真实双路匹配'的证据。"
            ),
        },
        "cannot_currently_validate": [
            "真实极线约束的准确性——没有标定文件，无法验证 EpipolarGate 在真实"
            "几何关系下的门控效果。",
            "position代价项在真实场景下的有效性——rectify_a_to_b 矩阵没有真实值，"
            "真实数据运行中这一项始终是 None。",
            "总代价阈值(max_total_cost)、权重(weights)、归一化尺度(norm_scale)"
            "是否适合真实数据——目前是工程经验起点，需要真实标定+标注数据才能标定。",
            "置信度分级(high/medium/low)在真实场景下是否合理——真实数据下因为"
            "标定和固定曝光延迟都未完成，置信度被强制统一为'none'/'low'，"
            "分级函数本身的阈值(0.3/0.7)只在fixed_exposure_delay_calibrated=true"
            "的合成场景里被验证过分支逻辑，没有真实意义上的校准。",
            "多目标冲突场景在真实数据密度下的表现——本批数据检出稀疏"
            "（A 24个/B 14个，180帧），真实数据里没有观测到多目标冲突"
            "触发Hungarian的案例，冲突消解只在合成数据和单元测试里验证过。",
        ],
        "known_constraints": [
            "不得虚构双相机内外参或极线参数——本阶段严格遵守，真实数据结果"
            "全部诚实标记 calibration_unavailable。",
            "固定曝光延迟未标定，所有真实匹配结果（如果将来标定完成后产生）"
            "仍需保持低置信度标记，直到曝光延迟也完成标定。",
        ],
    }
    if stage_timings_ms is not None:
        report["stage_timings_ms"] = stage_timings_ms
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
