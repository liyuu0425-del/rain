#!/usr/bin/env python3
"""算法演示程序：一个脚本跑通整条检测链，每一步都输出可看的中间结果。

## 用途

验证算法能跑通，看清每一步在做什么。**不是生产程序**——
不做并行、不做双路匹配、不算物理量。有效数据采回来之后，
参数要重新标定，这个脚本也可以直接拿来对比调参效果。

## 运行

    cd D:\\Codex\\毕设相关\\rain
    python tools\\demo.py                      # 用 config.yaml 的默认参数
    python tools\\demo.py --frames 60           # 只跑 60 帧（默认 180）
    python tools\\demo.py --thresh 15           # 临时改检测阈值
    python tools\\demo.py --tracker nn_gate     # 临时换跟踪算法
    python tools\\demo.py --batch Capture_20260727_232430

## 输出（都在 demo_output/ 下）

    01_原图.png            输入的一帧原始图像
    02_背景图.png          从整批合成出来的背景
    03_背景差分.png        bg - img，雨滴变亮点（已拉伸对比度便于观看）
    04_二值化.png          阈值分割结果
    05_形态学.png          开运算去噪后
    06_检测叠加.png        画上轮廓、质心和等效直径
    07_轨迹图.png          整批的轨迹连线
    detections.csv        逐个候选雨滴的位置/面积/等效直径/质量标记
    tracks.csv            逐条轨迹的长度和像素位移/帧

## 重要提醒

输出的单位全部是**像素**。
    - 像素-毫米标定未完成 -> 不能输出毫米粒径
    - 逐帧真实时间戳未确认 -> 只能输出"像素位移/帧"，不是速度
    - 有效观测体积未标定 -> 不能输出浓度和雨强
这些限制来自交接文档的物理约束，不是代码没写完。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml

from rain.imageio import imwrite_unicode
from rain.io_index import pair_batches
from rain.registry import build_background, build_detector, build_reader, build_tracker
from rain.sink import draw_overlay, write_batch_csv
from rain.types import BatchResult, qc_to_text

OUT_DIR = Path(__file__).parent.parent / "demo_output"


def stretch(img: np.ndarray) -> np.ndarray:
    """把低对比度的图拉伸到 0-255，只为了肉眼看得清。

    注意：这只用于保存演示图，不参与任何计算。
    背景差分图里雨滴的差值只有 20-47 灰阶，不拉伸的话
    存出来几乎是全黑，看不出东西。
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return img
    return ((img.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="降雨检测算法演示")
    ap.add_argument("--batch", default=None, help="批次名，默认用第一个")
    ap.add_argument("--frames", type=int, default=180, help="处理多少帧，默认 180")
    ap.add_argument("--thresh", type=int, default=None, help="临时覆盖检测阈值")
    ap.add_argument("--tracker", default=None, help="临时换跟踪器：gravity_gate|nn_gate")
    ap.add_argument("--show-frame", type=int, default=0, help="中间图取第几帧，默认 0")
    args = ap.parse_args()

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 命令行覆盖配置，方便快速试参数
    if args.thresh is not None:
        cfg["detector"]["params"]["thresh"] = args.thresh
    if args.tracker is not None:
        cfg["tracker"]["name"] = args.tracker
        if args.tracker == "nn_gate":
            cfg["tracker"]["params"] = {"max_dist_px": 150.0, "min_track_len": 2}

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 62)
    print("降雨双路测量 - 算法演示")
    print("=" * 62)
    print(f"检测器  : {cfg['detector']['name']}  {cfg['detector']['params']}")
    print(f"跟踪器  : {cfg['tracker']['name']}")
    print(f"背景    : {cfg['background']['name']}")
    print()

    # --- 找批次 -----------------------------------------------------------
    pairs, issues = pair_batches(cfg["data"]["dir_a"], cfg["data"]["dir_b"])
    if not pairs:
        print("没有配对成功的批次。检查 config.yaml 里的 data.dir_a / dir_b")
        for it in issues:
            print("  ", it)
        return 1

    batch = pairs[0]
    if args.batch:
        found = [p for p in pairs if p.name == args.batch]
        if not found:
            print(f"找不到批次 {args.batch}。可用的有：")
            for p in pairs[:20]:
                print("  ", p.name)
            return 1
        batch = found[0]

    n = min(args.frames, batch.n_frames)
    paths = batch.paths_a[:n]
    frame_ids = batch.frame_ids[:n]

    print(f"批次    : {batch.name}")
    print(f"帧数    : {n} / {batch.n_frames}（只演示 A 路）")
    if batch.note:
        print(f"备注    : {batch.note}")
    print()

    reader = build_reader(cfg["reader"])
    bgmodel = build_background(cfg["background"])
    detector = build_detector(cfg["detector"])
    tracker = build_tracker(cfg["tracker"])

    # --- 步骤 1：读取 ------------------------------------------------------
    print("[1/6] 读取图像 ...", end=" ", flush=True)
    t = time.perf_counter()
    frames = [reader.read(p) for p in paths]
    ms_read = (time.perf_counter() - t) * 1000
    print(f"{ms_read:.0f} ms  ({ms_read/n:.1f} ms/帧)")
    print(f"      尺寸 {frames[0].shape}  dtype {frames[0].dtype}  "
          f"灰度范围 {frames[0].min()}-{frames[0].max()}")

    k = min(args.show_frame, n - 1)
    imwrite_unicode(str(OUT_DIR / "01_原图.png"), frames[k])

    # --- 步骤 2：背景建模 --------------------------------------------------
    print("[2/6] 合成背景 ...", end=" ", flush=True)
    t = time.perf_counter()
    bg = bgmodel.build(frames)
    ms_bg = (time.perf_counter() - t) * 1000
    print(f"{ms_bg:.0f} ms")
    print(f"      背景平均灰度 {bg.mean():.1f}")
    imwrite_unicode(str(OUT_DIR / "02_背景图.png"), bg)

    # --- 步骤 3-5：检测（带中间图）-----------------------------------------
    print("[3/6] 检测中间步骤（第 %d 帧）..." % frame_ids[k])
    dbg: dict = {}
    dets_k = detector.detect(frames[k], bg, frame_idx=frame_ids[k], debug=dbg)

    if "diff" in dbg:
        d = dbg["diff"]
        print(f"      背景差分：最大差值 {d.max()}  非零像素 {int((d > 0).sum())}")
        imwrite_unicode(str(OUT_DIR / "03_背景差分.png"), stretch(d))
    if "binary" in dbg:
        b = dbg["binary"]
        print(f"      二值化后：前景像素 {int(b.sum() // 255)}")
        imwrite_unicode(str(OUT_DIR / "05_形态学.png"), b)
    print(f"      这一帧检出 {len(dets_k)} 个候选目标")

    # --- 步骤 4：全批检测 --------------------------------------------------
    print("[4/6] 全批检测 ...", end=" ", flush=True)
    t = time.perf_counter()
    per_frame = [
        detector.detect(img, bg, frame_idx=fid) for img, fid in zip(frames, frame_ids)
    ]
    ms_det = (time.perf_counter() - t) * 1000
    all_dets = [d for lst in per_frame for d in lst]
    counts = [len(lst) for lst in per_frame]
    print(f"{ms_det:.0f} ms  ({ms_det/n:.1f} ms/帧)")
    print(f"      共 {len(all_dets)} 个检测，每帧 {np.mean(counts):.1f} 个"
          f"（最少 {min(counts)}，最多 {max(counts)}）")

    if all_dets:
        areas = np.array([d.area_px for d in all_dets])
        deqs = np.array([d.d_eq_px for d in all_dets])
        print(f"      面积(px)  中位 {np.median(areas):.0f}  范围 {areas.min():.0f}-{areas.max():.0f}")
        print(f"      等效直径(px) 中位 {np.median(deqs):.1f}  范围 {deqs.min():.1f}-{deqs.max():.1f}")
        n_flag = sum(1 for d in all_dets if d.qc_flags)
        print(f"      带 QC 标记的 {n_flag} 个（{100*n_flag/len(all_dets):.1f}%）")

    imwrite_unicode(str(OUT_DIR / "06_检测叠加.png"), draw_overlay(frames[k], per_frame[k]))

    # --- 步骤 5：跟踪 ------------------------------------------------------
    print("[5/6] 跟踪 ...", end=" ", flush=True)
    t = time.perf_counter()
    tracks = tracker.track(per_frame, cam="a")
    ms_trk = (time.perf_counter() - t) * 1000
    print(f"{ms_trk:.0f} ms")

    if tracks:
        lens = np.array([tr.n_frames for tr in tracks])
        used = int(lens.sum())
        print(f"      {len(tracks)} 条轨迹，平均长 {lens.mean():.1f} 帧"
              f"（最长 {lens.max()}）")
        print(f"      用掉 {used}/{len(all_dets)} 个检测（{100*used/len(all_dets):.0f}%）")

        disp = [tr.displacement_px_per_frame() for tr in tracks]
        dys = [d[1] for d in disp if d]
        dxs = [d[0] for d in disp if d]
        if dys:
            print(f"      垂直位移 中位 {np.median(dys):+.0f} px/帧"
                  f"（p10 {np.percentile(dys,10):+.0f}  p90 {np.percentile(dys,90):+.0f}）")
            print(f"      水平位移 中位 {np.median(dxs):+.1f} px/帧")
            print()
            print("      注意：以上是【像素位移/帧】，不是速度。")
            print("            逐帧真实时间戳未确认，禁止用 帧号差/180 换算成秒。")
    else:
        print("      没有连出任何轨迹 —— 检查跟踪器门控参数")

    _save_track_plot(frames[k], tracks, OUT_DIR / "07_轨迹图.png")

    # --- 步骤 6：写 CSV ----------------------------------------------------
    print("[6/6] 写结果 ...", end=" ", flush=True)
    result = BatchResult(
        batch_name=batch.name,
        detections_a=all_dets,
        tracks_a=tracks,
        note=batch.note,
    )
    det_csv, trk_csv = write_batch_csv(result, str(OUT_DIR))
    print("完成")
    print()

    # --- 小结 --------------------------------------------------------------
    total = ms_read + ms_bg + ms_det + ms_trk
    print("-" * 62)
    print("耗时分解（单线程，单路）")
    print(f"  读取+解码  {ms_read:7.0f} ms  {100*ms_read/total:4.1f}%   {ms_read/n:5.1f} ms/帧")
    print(f"  背景建模    {ms_bg:7.0f} ms  {100*ms_bg/total:4.1f}%")
    print(f"  检测        {ms_det:7.0f} ms  {100*ms_det/total:4.1f}%   {ms_det/n:5.1f} ms/帧")
    print(f"  跟踪        {ms_trk:7.0f} ms  {100*ms_trk/total:4.1f}%")
    print(f"  合计        {total:7.0f} ms          {total/n:5.1f} ms/帧")
    print("-" * 62)
    print(f"输出目录: {OUT_DIR}")
    print()
    print("参数提醒：当前阈值和门控都是从这批样本数据估出来的经验值，")
    print("          不是标定过的最终参数。有效数据采回来后必须重新标定。")
    return 0


def _save_track_plot(base: np.ndarray, tracks, out_path: Path):
    """把所有轨迹画在一张图上，直观看跟踪结果是否合理。

    正确的结果应该是一组基本平行的、指向下方的线段。
    如果出现大量横向连线或折返，说明门控参数不对。
    """
    canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    canvas = (canvas * 0.35).astype(np.uint8)  # 压暗底图，让轨迹更醒目

    rng = np.random.default_rng(0)  # 固定种子，保证每次配色一样
    for tr in tracks:
        color = tuple(int(c) for c in rng.integers(80, 255, size=3))
        pts = [(int(round(d.cx)), int(round(d.cy))) for d in tr.detections]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pts[0], 3, color, -1)       # 起点实心
        cv2.circle(canvas, pts[-1], 3, color, 1)       # 终点空心

    cv2.putText(
        canvas,
        f"{len(tracks)} tracks",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    imwrite_unicode(str(out_path), canvas)


if __name__ == "__main__":
    sys.exit(main())
