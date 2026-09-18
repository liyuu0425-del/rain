#!/usr/bin/env python3
"""快速测试：验证读取、背景、检测模块能工作。

不需要完整批次,只读一个目录的前 30 帧,跑完检测,输出一张叠加图。
这是写 runners/run_batch.py 之前的冒烟测试。
"""

import os
import sys
from pathlib import Path

# 让 rain 包可以被 import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml

from rain import build_background, build_detector, build_reader, describe, qc_to_text
from rain.imageio import imwrite_unicode


def main():
    # 读配置
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=== 已注册实现 ===")
    print(describe())
    print()

    # 读一个批次的前 30 帧
    batch_dir = Path(cfg["data"]["dir_a"]) / "Capture_20260727_232420"
    if not batch_dir.exists():
        print(f"数据目录不存在: {batch_dir}")
        return

    all_files = sorted(batch_dir.glob("*.tif"))[:30]
    if not all_files:
        print(f"没有找到 TIF 文件: {batch_dir}")
        return

    print(f"=== 测试批次: {batch_dir.name} (前 {len(all_files)} 帧) ===\n")

    # 构造实例
    reader = build_reader(cfg["reader"])
    bgmodel = build_background(cfg["background"])
    detector = build_detector(cfg["detector"])

    print("1. 读取帧...")
    frames = [reader.read(str(f)) for f in all_files]
    print(f"   读取 {len(frames)} 帧, 尺寸 {frames[0].shape}, dtype {frames[0].dtype}")

    print("2. 建立背景...")
    bg = bgmodel.build(frames)
    print(f"   背景图 shape {bg.shape}, mean {bg.mean():.1f}")

    print("3. 检测第 1 帧 (frame_000)...")
    debug_dict = {}
    dets = detector.detect(frames[0], bg, frame_idx=0, debug=debug_dict)
    print(f"   检出 {len(dets)} 个候选目标")

    if dets:
        for i, d in enumerate(dets[:5]):  # 只打印前 5 个
            qc = qc_to_text(d.qc_flags)
            print(
                f"   [{i}] cx={d.cx:.1f} cy={d.cy:.1f} area={d.area_px:.0f}px "
                f"d_eq={d.d_eq_px:.1f}px qc={qc}"
            )
        if len(dets) > 5:
            print(f"   ... 还有 {len(dets)-5} 个")

    print("\n4. 保存叠加图和中间图...")
    out_dir = Path(__file__).parent.parent / "test_output"
    out_dir.mkdir(exist_ok=True)

    # 中间图: diff 和 binary
    if "diff" in debug_dict:
        imwrite_unicode(str(out_dir / "debug_diff.png"), debug_dict["diff"])
    if "binary" in debug_dict:
        imwrite_unicode(str(out_dir / "debug_binary.png"), debug_dict["binary"])

    # 叠加图: 画轮廓和质心
    overlay = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
    for d in dets:
        color = (0, 255, 0) if d.qc_flags == 0 else (0, 165, 255)  # 绿=ok, 橙=有问题
        if d.contour is not None:
            cv2.drawContours(overlay, [d.contour], -1, color, 1)
        cv2.circle(overlay, (int(d.cx), int(d.cy)), 2, color, -1)
        # 标注 d_eq
        text = f"{d.d_eq_px:.0f}"
        cv2.putText(
            overlay,
            text,
            (int(d.cx) + 5, int(d.cy) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
        )

    imwrite_unicode(str(out_dir / "overlay_frame000.png"), overlay)
    print(f"   保存到 {out_dir}")
    print(f"   - debug_diff.png     : 背景差分图")
    print(f"   - debug_binary.png   : 二值化图")
    print(f"   - overlay_frame000.png : 检测叠加图 (绿=ok, 橙=qc标记)")

    print("\n[OK] 冒烟测试通过")


if __name__ == "__main__":
    main()
