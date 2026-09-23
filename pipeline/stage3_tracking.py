#!/usr/bin/env python3
"""阶段3：单路跨帧跟踪 —— A、B两路各自独立跟踪，暂不做双路匹配。

## 用途

对 A、B 两路各180帧真实图像，复用阶段2的六步检测流水线逐帧检测，
再用卡尔曼预测 + 门控关联 + 轨迹生命周期管理，把逐帧检测串成轨迹。
两路完全独立处理，互不影响（双路匹配是阶段4的事）。

## 时间基准（硬性约束，任务书明确要求）

    A 路轨迹记录 a_device_relative_us；B 路轨迹记录 b_device_relative_us。
    时间戳一律来自阶段1已验收的 capture_timing.csv（dir_a 是确认过的
    主记录来源），换算成秒；不使用文件名毫秒、host接收时间、帧号或固定帧率。
    排序键是 pair_index，不是文件名或检测顺序。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage3_tracking.py

## 输出（默认写到 config/pipeline.yaml 里 tracking.output_dir 指定的目录）

    tracks.csv                       A、B两路全部轨迹点（含仅预测点）
    a_tracks_overlay.png             A路轨迹叠加图
    b_tracks_overlay.png             B路轨迹叠加图
    stage3_report.json               性能统计 + 轨迹统计 + 现状说明
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
from rain.imageio import imwrite_unicode
from rain.io_index import parse_frame_id
from rain.registry import build_background, build_detector, build_reader, build_track_manager
from rain.tracking.types import FrameContext, TrackPoint, TrackRecord, track_qc_to_text
from rain.types import qc_to_text

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"

# A/B 两路各自使用的设备时间字段——任务书的硬性约束，写死在这里，
# 不放进配置文件：这不是一个"可调参数"，而是一条不允许更改的规则。
_DEVICE_TIME_FIELD = {"a": "a_device_relative_us", "b": "b_device_relative_us"}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段3：单路跨帧跟踪（A/B独立）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    trk_cfg = cfg["tracking"]
    output_dir = Path(trk_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段3：单路跨帧跟踪（A/B独立，暂不做双路匹配）")
    print("=" * 70)
    print(f"输出目录 : {output_dir}")

    # 阶段1已确认 A/B 两路 capture_timing.csv 字节级相同，dir_a 是主记录来源。
    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}
    device_time_us_by_field = {
        "a_device_relative_us": {r.pair_index: r.a_device_relative_us for r in timing_records},
        "b_device_relative_us": {r.pair_index: r.b_device_relative_us for r in timing_records},
    }

    reader = build_reader(cfg["reader"])

    camera_results: dict[str, dict] = {}
    all_points: list[TrackPoint] = []

    for cam in ("a", "b"):
        print(f"\n{'-'*70}\n处理 {cam.upper()} 路\n{'-'*70}")
        result = _process_camera(
            cam=cam,
            folder=cfg["data"][f"dir_{cam}"],
            cfg=cfg,
            trk_cfg=trk_cfg,
            reader=reader,
            filename_to_pair_index=filename_to_pair_index,
            device_time_us_map=device_time_us_by_field[_DEVICE_TIME_FIELD[cam]],
            output_dir=output_dir,
        )
        camera_results[cam] = result
        all_points.extend(result["points"])

    tracks_csv = output_dir / "tracks.csv"
    _write_tracks_csv(all_points, tracks_csv)
    print(f"\n{tracks_csv} 共 {len(all_points)} 行"
          f"（A={sum(1 for p in all_points if p.camera=='a')}，"
          f"B={sum(1 for p in all_points if p.camera=='b')}）")

    report_path = output_dir / "stage3_report.json"
    _write_report(report_path, camera_results)
    print(f"{report_path}")

    print()
    print("=" * 70)
    print("阶段3完成：A、B两路各自独立跟踪，未做双路匹配（阶段4的工作）。")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# 单路处理
# ---------------------------------------------------------------------------


def _process_camera(
    cam: str, folder: str, cfg: dict, trk_cfg: dict, reader,
    filename_to_pair_index: dict, device_time_us_map: dict, output_dir: Path,
) -> dict:
    det_cfg = cfg["detection"]

    # --- 1. 列出全部帧文件，用 timing 里的 pair_index 联表（而不是直接信任
    #        文件名解析出的帧号）；两者在本数据集数值相同，但联表是显式的、
    #        可审计的，不是"恰好一样"的隐性假设。 -----------------------------
    frame_files = _list_camera_frame_files(folder)
    n = min(int(det_cfg["n_frames"]), len(frame_files))
    frame_files = frame_files[:n]

    paths = [p for _, p in frame_files]
    t0 = time.perf_counter()
    frames = [reader.read(p) for p in paths]
    ms_read = (time.perf_counter() - t0) * 1000
    print(f"  读取 {n}/{len(frame_files) if len(frame_files)>=n else n} 帧: {ms_read:.0f} ms")

    bgmodel = build_background(cfg["background"])
    bg = bgmodel.build(frames)

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

    # --- 2. 逐帧检测，同时按 pair_index 组装 FrameContext -----------------
    detect_ms_list: list[float] = []
    frame_contexts: list[FrameContext] = []
    for fn, path, img in zip((f for f, _ in frame_files), paths, frames):
        if fn not in filename_to_pair_index:
            raise KeyError(
                f"文件 {fn} 在 capture_timing.csv 里找不到对应的 pair_index，"
                "阶段1的配对结果和这次扫描的文件列表不一致，停止处理（不能强行猜配对）。"
            )
        pair_index = filename_to_pair_index[fn]
        device_time_s = device_time_us_map[pair_index] / 1e6  # us -> s

        t0 = time.perf_counter()
        dets = detector.detect(img, bg, frame_idx=pair_index)
        detect_ms_list.append((time.perf_counter() - t0) * 1000)

        frame_contexts.append(FrameContext(pair_index=pair_index, device_time_s=device_time_s, detections=dets))

    # 要求1：显式按 pair_index 排序后再喂给跟踪器。
    frame_contexts.sort(key=lambda f: f.pair_index)

    # --- 3. 逐帧跑跟踪器 ----------------------------------------------------
    manager = build_track_manager({
        "name": trk_cfg["track_manager"]["name"],
        "params": {
            **trk_cfg["track_manager"]["params"],
            "camera": cam,
            "kalman_predictor": trk_cfg["kalman_predictor"],
            "track_associator": trk_cfg["track_associator"],
        },
    })

    predict_ms_list: list[float] = []
    associate_ms_list: list[float] = []
    manage_ms_list: list[float] = []
    all_points: list[TrackPoint] = []

    for fc in frame_contexts:
        step_result = manager.step(fc)
        predict_ms_list.append(step_result.predict_ms)
        associate_ms_list.append(step_result.associate_ms)
        manage_ms_list.append(step_result.manage_ms)
        all_points.extend(step_result.points)

    records = manager.finalize()
    counters = manager.get_summary_counters()

    n_dets_total = sum(len(fc.detections) for fc in frame_contexts)
    print(f"  {n} 帧，共 {n_dets_total} 个检测 -> {len(records)} 条轨迹")
    lens = [r.n_points for r in records]
    if lens:
        print(f"  轨迹长度 平均 {np.mean(lens):.1f}  最大 {max(lens)}")
    n_short = sum(1 for r in records if r.status == "ended_short")
    print(f"  短轨迹(<{trk_cfg['track_manager']['params']['min_track_len']}点): {n_short}  "
          f"丢失次数: {counters['n_lost_events']}  "
          f"候选冲突次数: {counters['n_candidate_conflicts']}  "
          f"未关联检测数: {counters['n_unassociated_detections']}")

    # --- 4. 轨迹叠加图 -------------------------------------------------------
    show_frame = frames[0] if frames else None
    if show_frame is not None:
        _save_track_overlay(cam, show_frame, records, output_dir)

    return {
        "n_frames": n,
        "n_detections_total": n_dets_total,
        "n_tracks": len(records),
        "track_len_mean": float(np.mean(lens)) if lens else 0.0,
        "track_len_max": int(max(lens)) if lens else 0,
        "n_short_tracks": n_short,
        "n_lost_events": counters["n_lost_events"],
        "n_candidate_conflicts": counters["n_candidate_conflicts"],
        "n_unassociated_detections": counters["n_unassociated_detections"],
        "timings_ms": {
            "detect": _stats(detect_ms_list),
            "predict": _stats(predict_ms_list),
            "associate": _stats(associate_ms_list),
            "manage": _stats(manage_ms_list),
        },
        "points": all_points,  # 供 main() 合并写 CSV
        "n_records_status": {
            status: sum(1 for r in records if r.status == status)
            for status in {"ended", "ended_short"}
        },
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _list_camera_frame_files(folder: str) -> list[tuple[str, str]]:
    """列出目录下全部 TIF，返回 (文件名, 完整路径)，按文件名里的帧号排序。

    排序只是为了让读取顺序稳定、读盘尽量顺序化；真正参与跟踪的顺序键是
    FrameContext.pair_index（来自 capture_timing.csv），在 main 流程里
    显式重新排序过，不依赖这里的顺序。
    """
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


def _stats(values_ms: list[float]) -> dict:
    if not values_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0}
    arr = np.array(values_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "n": len(arr),
    }


def _combined_qc_text(flags: int) -> str:
    """quality_flags 里同时压了 Detection 的QC位(0~6)和轨迹级QC位(8+)，
    分别翻译再拼起来，方便人工复核。"""
    det_text = qc_to_text(flags & 0xFF)
    trk_text = track_qc_to_text(flags & ~0xFF)
    parts = [t for t in (det_text, trk_text) if t and t != "ok"]
    return "|".join(parts) if parts else "ok"


def _write_tracks_csv(points: list[TrackPoint], out_path: Path) -> None:
    columns = [
        "track_id", "camera_id", "pair_index", "device_time_s",
        "detected_cx_px", "detected_cy_px", "detected_area_px", "detected_d_eq_px",
        "predicted_cx_px", "predicted_cy_px",
        "association_method", "quality_flags", "quality_text",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for p in sorted(points, key=lambda p: (p.camera, p.track_id, p.pair_index)):
            det = p.detected
            w.writerow([
                p.track_id, p.camera, p.pair_index, f"{p.device_time_s:.6f}",
                "" if det is None else f"{det.cx:.2f}",
                "" if det is None else f"{det.cy:.2f}",
                "" if det is None else f"{det.area_px:.1f}",
                "" if det is None else f"{det.d_eq_px:.2f}",
                f"{p.predicted_cx:.2f}", f"{p.predicted_cy:.2f}",
                p.association_method, p.quality_flags, _combined_qc_text(p.quality_flags),
            ])


def _save_track_overlay(cam: str, base_img: np.ndarray, records: list[TrackRecord], output_dir: Path) -> None:
    """把这一路全部轨迹（只画真实检测点，不画仅预测点）画在同一张图上。

    正确的结果应该是一组基本平行、指向下方的线段（雨滴下落）。
    大量横向连线或折返，说明门控/代价参数不合适，需要用真实数据复核。
    """
    canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    canvas = (canvas * 0.35).astype(np.uint8)  # 压暗底图，让轨迹更醒目

    rng = np.random.default_rng(0)  # 固定种子，保证每次配色一样
    n_drawn = 0
    for rec in records:
        pts = [(int(round(p.detected.cx)), int(round(p.detected.cy))) for p in rec.points]
        if len(pts) < 2:
            continue
        color = tuple(int(c) for c in rng.integers(80, 255, size=3))
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pts[0], 3, color, -1)
        cv2.circle(canvas, pts[-1], 3, color, 1)
        n_drawn += 1

    cv2.putText(
        canvas, f"{cam.upper()}: {n_drawn} tracks (>=2 points)",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    imwrite_unicode(str(output_dir / f"{cam}_tracks_overlay.png"), canvas)


def _write_report(report_path: Path, camera_results: dict) -> None:
    per_camera = {}
    for cam, r in camera_results.items():
        per_camera[cam] = {k: v for k, v in r.items() if k != "points"}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_camera": per_camera,
        "time_field_used": {
            "a": "a_device_relative_us",
            "b": "b_device_relative_us",
        },
        "sort_key": "pair_index",
        "stage3_caveats": [
            {
                "item": "no_velocity_computed",
                "detail": (
                    "本阶段不计算正式速度。tracks.csv 里只有位置和设备时间，"
                    "位置-时间线性拟合和速度计算留到阶段5（LinearVelocityEstimator）。"
                ),
            },
            {
                "item": "kalman_role",
                "detail": (
                    "卡尔曼滤波只用于预测下一帧位置、缩小关联搜索范围，"
                    "不产出、不替代真实检测——TrackRecord.points（计入轨迹长度的点）"
                    "只包含真实检测关联上的点；'仅预测'的点只写进 tracks.csv 做审计追溯，"
                    "association_method='predicted_only'，不计入轨迹长度。"
                ),
            },
            {
                "item": "gate_and_cost_parameters_not_calibrated",
                "detail": (
                    "门控距离(gate_dx_max/gate_dy_max)、代价权重、最大匹配代价、"
                    "卡尔曼噪声参数都参考了老Demo在旧数据集(cama/camb)上的经验范围，"
                    "本批新数据(A/B-Capture_085820)候选目标本就偏少（检测阈值未标定），"
                    "尚未有足够样本重新标定这些跟踪参数。"
                ),
            },
            {
                "item": "detection_upstream_limitation",
                "detail": (
                    "跟踪结果的质量直接受阶段2检测参数未标定的影响：如果检测阈值"
                    "漏检了某些雨滴帧，对应轨迹会出现断裂或过短；这不是跟踪算法本身的缺陷。"
                ),
            },
        ],
        "known_constraints": [
            "测速只能用 a_device_relative_us / b_device_relative_us，"
            "禁止用文件名毫秒、host 接收时间或帧号/固定帧率。",
            "A、B两路独立跟踪，尚未做双路匹配（阶段4）。",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
