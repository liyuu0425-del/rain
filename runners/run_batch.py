"""处理一个批次：并行的原子单位。

## 这个文件的地位

它是整套架构的核心约定：
    - **纯函数**：输入一个 BatchPair 和配置，输出一个 BatchResult。
    - **单线程**：这里面一行并发代码都没有。
    - **无全局状态**：不读写任何模块级变量，不碰其他批次。

正因如此，run_offline.py 才能放心地把它扔进线程池并行跑，
而调试算法时又可以单独跑一个批次、单步跟进、看中间图。

## 为什么在函数内部 build_* 而不是外面建好传进来

每个 worker 必须持有自己的一套算法实例。
如果实现里缓存了复用 buffer（比如预分配 morphology 输出数组，
避免每帧分配 3 MB），跨线程共享就会数据竞争，
症状是偶发的结果错乱，极难排查。
构造开销可以忽略（只是建几个 kernel），但省掉了整类并发 bug。

## 时间约束

帧号只用于排序和配对。这里输出的是像素位移/帧，不是速度。
真实速度必须等逐帧时间戳或经验证的固定触发周期确认后才能计算。
"""

from __future__ import annotations

import os
import time
import traceback

import cv2

from rain.registry import build_background, build_detector, build_reader, build_tracker
from rain.sink import save_overlay
from rain.types import BatchPair, BatchResult


def process_batch(
    batch: BatchPair,
    cfg: dict,
    overlay_dir: str | None = None,
) -> BatchResult:
    """处理一个已配对的双路批次。

    参数：
        batch       : 配对好的批次（paths_a / paths_b 已按帧号对齐）
        cfg         : 完整配置字典（config.yaml 的内容）
        overlay_dir : 不为 None 时保存检测叠加图到这个目录

    返回：
        BatchResult。处理失败时 error 字段填原因，**不抛异常**——
        否则线程池里一个坏批次会打断整轮处理。
    """
    result = BatchResult(batch_name=batch.name, note=batch.note)
    t_total = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # 每个 worker 自己的算法实例。见文件头"为什么在函数内部 build_*"。
        # ------------------------------------------------------------------
        reader = build_reader(cfg["reader"])
        bgmodel = build_background(cfg["background"])
        detector = build_detector(cfg["detector"])
        tracker = build_tracker(cfg["tracker"])

        debug_cfg = cfg.get("debug", {}) or {}
        n_overlay = int(debug_cfg.get("overlay_samples", 0)) if overlay_dir else 0

        # 两路分别处理。结构完全对称，所以抽成内部函数。
        det_a, trk_a, ms_a = _process_one_camera(
            paths=batch.paths_a,
            frame_ids=batch.frame_ids,
            cam="a",
            reader=reader,
            bgmodel=bgmodel,
            detector=detector,
            tracker=tracker,
            overlay_dir=overlay_dir,
            batch_name=batch.name,
            n_overlay=n_overlay,
        )
        det_b, trk_b, ms_b = _process_one_camera(
            paths=batch.paths_b,
            frame_ids=batch.frame_ids,
            cam="b",
            reader=reader,
            bgmodel=bgmodel,
            detector=detector,
            tracker=tracker,
            overlay_dir=overlay_dir,
            batch_name=batch.name,
            n_overlay=n_overlay,
        )

        result.detections_a = det_a
        result.tracks_a = trk_a
        result.detections_b = det_b
        result.tracks_b = trk_b

        # 阶段耗时合并。这是性能剖析和工控机压力测试的数据来源。
        for k, v in ms_a.items():
            result.stage_ms[f"a_{k}"] = v
        for k, v in ms_b.items():
            result.stage_ms[f"b_{k}"] = v

        # 双路匹配、物理量计算、统计 —— 当前留空。
        # 双相机内外参标定和有效观测体积标定完成前，这些不能实现。

    except Exception as exc:  # noqa: BLE001 - 故意兜住所有异常
        result.error = f"{type(exc).__name__}: {exc}"
        result.stage_ms["traceback"] = 0  # 占位，真正的栈存在下面
        result.note = (result.note + "; " if result.note else "") + traceback.format_exc(limit=3)

    result.stage_ms["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return result


def _process_one_camera(
    paths: list[str],
    frame_ids: list[int],
    cam: str,
    reader,
    bgmodel,
    detector,
    tracker,
    overlay_dir: str | None,
    batch_name: str,
    n_overlay: int,
):
    """处理单路的一整个批次：读取 -> 背景 -> 逐帧检测 -> 跟踪。

    返回 (检测列表, 轨迹列表, 各阶段耗时ms)。
    """
    stage_ms: dict[str, float] = {}

    # --- 1. 读取全批 --------------------------------------------------------
    # 一批 180 帧 × 3.19 MB ≈ 573 MB。这是内存占用的峰值来源。
    # 之所以能全批读进来，是因为背景建模需要看到整批。
    # 如果内存吃紧，可以改成两遍扫描（第一遍抽样建背景，第二遍流式检测），
    # 代价是读盘翻倍——在 LZW 解码占 70% 成本的前提下不划算。
    t = time.perf_counter()
    frames = [reader.read(p) for p in paths]
    stage_ms["read_ms"] = (time.perf_counter() - t) * 1000.0

    # --- 2. 背景建模 --------------------------------------------------------
    t = time.perf_counter()
    bg = bgmodel.build(frames)
    stage_ms["background_ms"] = (time.perf_counter() - t) * 1000.0

    # --- 3. 逐帧检测 --------------------------------------------------------
    t = time.perf_counter()
    per_frame = []
    all_dets = []
    for i, (img, fid) in enumerate(zip(frames, frame_ids)):
        # debug=None：主流程不产出中间图，零开销
        dets = detector.detect(img, bg, frame_idx=fid, debug=None)
        per_frame.append(dets)
        all_dets.extend(dets)
    stage_ms["detect_ms"] = (time.perf_counter() - t) * 1000.0

    # --- 4. 批内跟踪 --------------------------------------------------------
    # 跟踪是有状态、顺序依赖的，所以留在批内串行。
    # 它只处理稀疏的检测列表，不碰整幅图像，成本可以忽略。
    t = time.perf_counter()
    tracks = tracker.track(per_frame, cam=cam)
    stage_ms["track_ms"] = (time.perf_counter() - t) * 1000.0

    # --- 5. 叠加图（可选，仅供人工复核）-------------------------------------
    if overlay_dir and n_overlay > 0:
        t = time.perf_counter()
        _save_overlays(frames, per_frame, frame_ids, cam, batch_name, overlay_dir, n_overlay)
        stage_ms["overlay_ms"] = (time.perf_counter() - t) * 1000.0

    stage_ms["n_frames"] = len(frames)
    stage_ms["n_detections"] = len(all_dets)
    stage_ms["n_tracks"] = len(tracks)

    return all_dets, tracks, stage_ms


def _save_overlays(frames, per_frame, frame_ids, cam, batch_name, overlay_dir, n_overlay):
    """均匀抽 n_overlay 帧保存叠加图，不是全存 —— 180 张 PNG 太占盘也太慢。"""
    if not frames:
        return
    os.makedirs(overlay_dir, exist_ok=True)

    step = max(1, len(frames) // n_overlay)
    for i in range(0, len(frames), step):
        if i >= len(frames):
            break
        out = os.path.join(overlay_dir, f"{batch_name}_{cam}_frame{frame_ids[i]:04d}.png")
        save_overlay(frames[i], per_frame[i], out)
