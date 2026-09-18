"""结果写出：CSV、轨迹表、检测叠加图。

所有写文件的路径都可能含中文（D:\\Codex\\毕设相关），
所以图像一律走 imageio.imwrite_unicode，CSV 一律显式 encoding="utf-8-sig"。

为什么是 utf-8-sig：
    带 BOM 的 UTF-8。Excel 直接双击打开含中文的 utf-8 CSV 会乱码，
    加了 BOM 才能正确识别。对 pandas / Python 读取没有影响。
"""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np

from .imageio import imwrite_unicode
from .types import BatchResult, Detection, Track, qc_to_text

# ---------------------------------------------------------------------------
# CSV 表头。字段顺序固定 —— 下游脚本和人工复核都依赖它。
# 单位一律是像素，因为像素-毫米标定尚未完成。
# ---------------------------------------------------------------------------
DETECTION_COLUMNS = [
    "batch",
    "cam",
    "frame_idx",
    "cx_px",
    "cy_px",
    "area_px",
    "d_eq_px",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "perimeter_px",
    "major_axis_px",
    "minor_axis_px",
    "qc_flags",
    "qc_text",
]

TRACK_COLUMNS = [
    "batch",
    "cam",
    "track_id",
    "n_frames",
    "frame_first",
    "frame_last",
    # 注意命名：位移/帧，不是速度。真实时间基准未确认前不得输出速度。
    "dx_px_per_frame",
    "dy_px_per_frame",
    "mean_d_eq_px",
    "qc_any",
]


def _detection_row(batch: str, cam: str, d: Detection) -> list:
    x, y, w, h = d.bbox
    return [
        batch,
        cam,
        d.frame_idx,
        f"{d.cx:.2f}",
        f"{d.cy:.2f}",
        f"{d.area_px:.1f}",
        f"{d.d_eq_px:.2f}",
        x,
        y,
        w,
        h,
        "" if d.perimeter_px is None else f"{d.perimeter_px:.2f}",
        "" if d.major_axis_px is None else f"{d.major_axis_px:.2f}",
        "" if d.minor_axis_px is None else f"{d.minor_axis_px:.2f}",
        d.qc_flags,
        qc_to_text(d.qc_flags),
    ]


def _track_row(batch: str, tr: Track) -> list:
    first, last = tr.frame_span
    disp = tr.displacement_px_per_frame()
    dx = "" if disp is None else f"{disp[0]:.3f}"
    dy = "" if disp is None else f"{disp[1]:.3f}"
    mean_d = float(np.mean([d.d_eq_px for d in tr.detections]))
    qc_any = 0
    for d in tr.detections:
        qc_any |= d.qc_flags
    return [
        batch,
        tr.cam,
        tr.track_id,
        tr.n_frames,
        first,
        last,
        dx,
        dy,
        f"{mean_d:.2f}",
        qc_any,
    ]


def write_batch_csv(result: BatchResult, out_dir: str) -> tuple[str, str]:
    """把一个批次的检测和轨迹写成两个 CSV。

    每批独立成文件，这样：
      - 并行 worker 之间不用抢同一个文件，无需加锁；
      - 文件存在即视为已完成，天然支持断点续跑；
      - 单批出问题不影响其他批。

    返回 (检测CSV路径, 轨迹CSV路径)。
    """
    os.makedirs(out_dir, exist_ok=True)

    det_path = os.path.join(out_dir, f"{result.batch_name}_detections.csv")
    trk_path = os.path.join(out_dir, f"{result.batch_name}_tracks.csv")

    with open(det_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETECTION_COLUMNS)
        for d in result.detections_a:
            w.writerow(_detection_row(result.batch_name, "a", d))
        for d in result.detections_b:
            w.writerow(_detection_row(result.batch_name, "b", d))

    with open(trk_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(TRACK_COLUMNS)
        for tr in result.tracks_a:
            w.writerow(_track_row(result.batch_name, tr))
        for tr in result.tracks_b:
            w.writerow(_track_row(result.batch_name, tr))

    return det_path, trk_path


def draw_overlay(
    img: np.ndarray,
    detections: list[Detection],
    show_size: bool = True,
) -> np.ndarray:
    """在灰度帧上画出检测结果，供人工复核。

    颜色约定：
        绿色 = QC 通过
        橙色 = 有 QC 标记（触边、粘连、形状异常等）
    """
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for d in detections:
        color = (0, 255, 0) if d.qc_flags == 0 else (0, 165, 255)

        if d.contour is not None:
            cv2.drawContours(overlay, [d.contour], -1, color, 1)
        else:
            # 没保留轮廓时退回画包围框
            x, y, w, h = d.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)

        cv2.circle(overlay, (int(round(d.cx)), int(round(d.cy))), 2, color, -1)

        if show_size:
            cv2.putText(
                overlay,
                f"{d.d_eq_px:.0f}",
                (int(d.cx) + 5, int(d.cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    return overlay


def save_overlay(
    img: np.ndarray,
    detections: list[Detection],
    out_path: str,
    show_size: bool = True,
) -> None:
    """画叠加图并保存。路径可含中文。"""
    imwrite_unicode(out_path, draw_overlay(img, detections, show_size))


def write_profile_csv(rows: list[dict], out_path: str) -> None:
    """写性能剖析表：每批每阶段的耗时。

    这是交接文档验收标准里"记录 CPU、内存、磁盘占用"的数据来源，
    也是判断有没有队列积压的依据。
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 收集所有出现过的列名，保证各批阶段不同时也能写出完整表
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
