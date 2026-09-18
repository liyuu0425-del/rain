"""批次索引与双路配对。

职责：
    扫描两路图像目录 -> 找出批次 -> 按批次名配对 -> 按帧号对齐 -> 产出 BatchPair 列表。

本项目数据的真实情况（实测 D:\\毕设\\数据）：
    cama 有 12 个批次（从 Capture_20260727_232420 起）
    camb 有 18 个批次（从 Capture_20260727_232320 起）
    -> camb 多出 6 个更早的批次，**必须按批次名取交集**，不能假设两路一一对应。

    同名批次的帧数也不总是相同：178 / 179 / 180 都出现过。
    -> 按帧号取交集对齐。缺的帧记进 note，**绝不复制上一帧伪造缺帧**。

    每批 180 帧 = 恰好 1 秒连续采集；批次之间相隔 10 秒。
    -> 批内帧连续，批间雨滴早已飞出视场，所以批次是天然的并行单元。

时间约束（来自交接文档，必须遵守）：
    帧号只用于排序和双路配对。
    禁止用 帧号差/180 当作真实时间差。
    文件夹名里的秒级时间不能作为单帧时间戳。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .types import BatchPair

# 文件名形如 frame_164.tif / frame_000.tif
_FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)

_IMAGE_EXTS = (".tif", ".tiff", ".raw")


@dataclass
class BatchScan:
    """单路的一个批次扫描结果（配对之前的中间产物）。"""

    name: str
    folder: str
    frames: dict[int, str]  # 帧号 -> 完整路径

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def parse_frame_id(filename: str) -> int | None:
    """从文件名里取帧号。取不到返回 None。

    帧号只用于排序和配对，不参与任何时间计算。
    """
    m = _FRAME_RE.search(os.path.basename(filename))
    return int(m.group(1)) if m else None


def scan_camera_dir(root: str) -> dict[str, BatchScan]:
    """扫描一路相机目录，返回 {批次名: BatchScan}。

    参数 root 形如 D:\\毕设\\数据\\cama，其下每个子目录是一个批次。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"相机目录不存在: {root}")

    batches: dict[str, BatchScan] = {}

    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue

        frames: dict[int, str] = {}
        for fn in os.listdir(folder):
            if not fn.lower().endswith(_IMAGE_EXTS):
                continue
            fid = parse_frame_id(fn)
            if fid is None:
                continue
            frames[fid] = os.path.join(folder, fn)

        if frames:
            batches[entry] = BatchScan(name=entry, folder=folder, frames=frames)

    return batches


def pair_batches(
    dir_a: str,
    dir_b: str,
    require_same_length: bool = False,
) -> tuple[list[BatchPair], list[str]]:
    """配对双路批次。

    返回 (配对成功的 BatchPair 列表, 问题说明列表)。

    参数：
        require_same_length : True 时，两路帧数不同的批次直接丢弃；
                              False（默认）时按帧号交集对齐并在 note 里记录。

    配对规则：
        1. 批次名取交集 —— 只有一路有的批次记进问题列表，不参与双路处理。
        2. 批次内按帧号取交集 —— 缺的帧跳过并记录。
        3. 帧号严格升序排列。
    """
    scan_a = scan_camera_dir(dir_a)
    scan_b = scan_camera_dir(dir_b)

    names_a = set(scan_a)
    names_b = set(scan_b)
    common = sorted(names_a & names_b)

    issues: list[str] = []

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    for n in only_a:
        issues.append(f"仅 A 路有批次 {n}（{scan_a[n].n_frames} 帧），不参与双路处理")
    for n in only_b:
        issues.append(f"仅 B 路有批次 {n}（{scan_b[n].n_frames} 帧），不参与双路处理")

    pairs: list[BatchPair] = []

    for name in common:
        ba, bb = scan_a[name], scan_b[name]
        ids_a, ids_b = set(ba.frames), set(bb.frames)
        common_ids = sorted(ids_a & ids_b)

        if not common_ids:
            issues.append(f"批次 {name} 两路帧号无交集，跳过")
            continue

        note_parts = []
        if len(ids_a) != len(ids_b):
            note_parts.append(f"帧数不同 A={len(ids_a)} B={len(ids_b)}")
            if require_same_length:
                issues.append(f"批次 {name} 帧数不同且要求严格一致，跳过")
                continue

        missing_a = sorted(ids_b - ids_a)
        missing_b = sorted(ids_a - ids_b)
        if missing_a:
            note_parts.append(f"A 路缺帧 {_brief(missing_a)}")
        if missing_b:
            note_parts.append(f"B 路缺帧 {_brief(missing_b)}")

        # 检查帧号是否连续。中间断号比末尾少帧更值得警惕。
        gaps = _find_gaps(common_ids)
        if gaps:
            note_parts.append(f"帧号中间断开 {_brief(gaps)}")

        pairs.append(
            BatchPair(
                name=name,
                paths_a=[ba.frames[i] for i in common_ids],
                paths_b=[bb.frames[i] for i in common_ids],
                frame_ids=common_ids,
                note="; ".join(note_parts),
            )
        )

    return pairs, issues


def _find_gaps(sorted_ids: list[int]) -> list[int]:
    """找出帧号序列里断开的位置，返回缺失的帧号。

    注意：这只是报告，不做任何插值或补帧。
    """
    if len(sorted_ids) < 2:
        return []
    missing = []
    for prev, cur in zip(sorted_ids, sorted_ids[1:]):
        if cur - prev > 1:
            missing.extend(range(prev + 1, cur))
    return missing


def _brief(values: list[int], limit: int = 5) -> str:
    """把一长串帧号缩短成可读文本，避免 note 里塞几百个数字。"""
    if len(values) <= limit:
        return str(values)
    head = ", ".join(str(v) for v in values[:limit])
    return f"[{head}, ... 共 {len(values)} 个]"


def summarize(pairs: list[BatchPair], issues: list[str]) -> str:
    """生成人可读的配对摘要，给命令行和审计报告用。"""
    lines = [f"配对成功 {len(pairs)} 个批次"]
    total_frames = sum(p.n_frames for p in pairs)
    lines.append(f"总帧数（每路）{total_frames}")

    flagged = [p for p in pairs if p.note]
    if flagged:
        lines.append(f"其中 {len(flagged)} 个批次有异常：")
        for p in flagged:
            lines.append(f"  {p.name}: {p.note}")

    if issues:
        lines.append(f"另有 {len(issues)} 条问题：")
        for it in issues:
            lines.append(f"  {it}")

    return "\n".join(lines)
