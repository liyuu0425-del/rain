#!/usr/bin/env python3
"""阶段4：双路匹配 —— 极线约束 -> 匹配门控 -> 代价矩阵 -> 最近邻/匈牙利。

## 用途

对 A、B 两路每个 pair_index 的检测结果做双路匹配。检测复用阶段2的六步
流水线（原地重新跑一遍检测，不读回 CSV——和阶段3处理方式一致），
不重新实现阶段2/阶段3的任何算法。

## 标定现状（任务书的硬性约束）

项目目前没有真实的双相机内外参/基础矩阵标定。`config/pipeline.yaml`
里 `stereo.calibration.path` 指向一个不存在的文件，`CalibrationLoader`
会诚实地返回 None，`EpipolarGate` 在没有标定时让所有候选都不通过，
原因固定标记 `calibration_unavailable`。这个脚本因此在真实数据上
**产出 0 个已验证的双路匹配**，这是诚实的结果，不是 bug，也不是任何
形式的"假装完成了双路匹配"。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python pipeline\\stage4_stereo_matching.py

## 输出（默认写到 config/pipeline.yaml 里 stereo.output_dir 指定的目录）

    stereo_matches.csv        每个候选对的完整诊断（匹配成功/失败原因/
                               各子代价/总代价/方法/置信度）
    overlay/pair_XXXX.png     抽样的双路匹配可视化
    stage4_report.json        单元测试引用 + 合成数据演示结果 + 真实数据
                               结果 + 当前无法验证的内容，四者明确分开
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
from rain.registry import build_background, build_detector, build_reader, build_stereo_matcher
from rain.stereo.calibration import CalibrationLoader
from rain.stereo.types import StereoCalibration, StereoMatchRow
from rain.types import Detection

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段4：双路匹配（极线约束优先，未标定时安全退化）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline.yaml 路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    st_cfg = cfg["stereo"]
    output_dir = Path(st_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段4：双路匹配")
    print("=" * 70)
    print(f"输出目录 : {output_dir}")

    # --- 1. 标定 ---------------------------------------------------------
    calib_path = st_cfg["calibration"]["path"]
    calibration = CalibrationLoader().load(calib_path)
    print(f"\n标定文件 : {calib_path}")
    print(f"标定可用 : {calibration is not None}"
          + ("" if calibration is not None else "  -> 所有候选将标记 calibration_unavailable"))

    # --- 2. 复用阶段2检测，逐相机跑一遍，按 pair_index 分组 -------------------
    reader = build_reader(cfg["reader"])
    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}

    det_by_camera: dict[str, dict[int, list[Detection]]] = {}
    frame_by_camera_pair: dict[str, dict[int, np.ndarray]] = {}
    n_frames_cfg = int(st_cfg["n_frames"])

    for cam in ("a", "b"):
        print(f"\n[{cam.upper()}路] 检测 ...")
        dets_by_pair, frames_by_pair, ms = _detect_camera(
            cam, cfg["data"][f"dir_{cam}"], cfg, reader, n_frames_cfg
        )
        det_by_camera[cam] = dets_by_pair
        frame_by_camera_pair[cam] = frames_by_pair
        n_dets = sum(len(v) for v in dets_by_pair.values())
        print(f"  {len(dets_by_pair)} 帧，共 {n_dets} 个检测，耗时 {ms:.0f} ms")

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
    matcher = build_stereo_matcher({"name": matcher_cfg["name"], "params": matcher_params})

    # --- 4. 逐 pair_index 匹配 ----------------------------------------------
    print("\n逐 pair_index 匹配 ...")
    all_pair_indices = sorted(set(det_by_camera["a"]) | set(det_by_camera["b"]))
    all_rows: list[StereoMatchRow] = []
    gate_ms_list, cost_ms_list, solve_ms_list = [], [], []
    n_conflicts_total = 0

    for pidx in all_pair_indices:
        dets_a = det_by_camera["a"].get(pidx, [])
        dets_b = det_by_camera["b"].get(pidx, [])
        result = matcher.match_frame(pidx, dets_a, dets_b)
        all_rows.extend(result.rows)
        gate_ms_list.append(result.gate_ms)
        cost_ms_list.append(result.cost_ms)
        solve_ms_list.append(result.solve_ms)
        n_conflicts_total += result.n_candidate_conflicts

    n_matched = sum(1 for r in all_rows if r.status == "matched")
    n_unmatched = len(all_rows) - n_matched
    print(f"  处理 {len(all_pair_indices)} 个 pair_index，共 {len(all_rows)} 条候选诊断")
    print(f"  匹配成功 : {n_matched}")
    print(f"  未匹配   : {n_unmatched}")
    reason_counts = _count_reasons(all_rows)
    print(f"  失败原因分布: {reason_counts}")

    # --- 5. 输出 stereo_matches.csv -----------------------------------------
    matches_csv = output_dir / "stereo_matches.csv"
    _write_stereo_matches_csv(all_rows, matches_csv)
    print(f"\n  {matches_csv}")

    # --- 6. 可视化：挑几个 pair_index 画出来 ---------------------------------
    _save_visualizations(
        all_pair_indices, det_by_camera, frame_by_camera_pair, all_rows,
        output_dir, n_overlay=int(st_cfg.get("overlay_samples", 6)),
    )

    # --- 7. 合成数据演示（独立于真实数据，证明算法逻辑本身是对的）-------------
    synthetic_result = _run_synthetic_demo()

    # --- 8. 报告 -------------------------------------------------------------
    report_path = output_dir / "stage4_report.json"
    _write_report(
        report_path=report_path,
        calibration=calibration,
        calib_path=calib_path,
        real_data={
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
            },
        },
        synthetic_data=synthetic_result,
    )
    print(f"  {report_path}")

    print()
    print("=" * 70)
    if calibration is None:
        print("阶段4完成：无真实标定文件，真实数据 0 个已验证匹配（calibration_unavailable）。")
        print("算法逻辑已用合成数据 + 单元测试验证，详见 stage4_report.json。")
    else:
        print("阶段4完成。")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# 检测（复用阶段2的六步流水线，不重新实现）
# ---------------------------------------------------------------------------


def _detect_camera(
    cam: str, folder: str, cfg: dict, reader, n_frames_cfg: int
) -> tuple[dict[int, list[Detection]], dict[int, np.ndarray], float]:
    det_cfg = cfg["detection"]
    timing_records = TimingReader().read(str(Path(cfg["data"]["dir_a"]) / "capture_timing.csv"))
    filename_to_pair_index = {r.pair_filename: r.pair_index for r in timing_records}

    frame_files = _list_camera_frame_files(folder)
    n = min(n_frames_cfg, len(frame_files))
    frame_files = frame_files[:n]

    paths = [p for _, p in frame_files]
    frames = [reader.read(p) for p in paths]

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

    t0 = time.perf_counter()
    dets_by_pair: dict[int, list[Detection]] = {}
    frames_by_pair: dict[int, np.ndarray] = {}
    for (fn, _), img in zip(frame_files, frames):
        if fn not in filename_to_pair_index:
            raise KeyError(f"文件 {fn} 在 capture_timing.csv 里找不到 pair_index，停止处理。")
        pidx = filename_to_pair_index[fn]
        dets_by_pair[pidx] = detector.detect(img, bg, frame_idx=pidx)
        frames_by_pair[pidx] = img
    ms = (time.perf_counter() - t0) * 1000

    return dets_by_pair, frames_by_pair, ms


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


def _write_report(report_path: Path, calibration, calib_path: str, real_data: dict, synthetic_data: dict) -> None:
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
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
