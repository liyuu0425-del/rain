"""output_integrity：检查阶段1自己产出的 CSV/报告有没有重复记录、行数对不对。

这些不是"数据质量"检查（那是 TimeQualityChecker 的职责），而是"我们自己
写代码有没有 bug"的检查——比如 FramePairer 会不会不小心把同一个
pair_index 配对了两次。属于阶段1对自身输出的自检，所以单独放一个模块，
不掺进 FramePairer / TimeQualityChecker 里。
"""

from __future__ import annotations

from .types import FramePairRecord, TimingQualityReport


def check_frame_pairing_uniqueness(
    frame_pairs: list[FramePairRecord], expected_n: int
) -> list[str]:
    """检查 frame_pairing.csv 里"正常 pair_index"（>=0）是否唯一、数量是否对得上。

    返回问题描述列表；空列表表示检查全部通过。

    注意：extra_a / extra_b 记录用 pair_index=-1 占位（见 FramePairer），
    这些不是"真实的图像对"，要先排除掉，只检查有真实 pair_index 的那部分。
    """
    problems: list[str] = []

    real_pairs = [p for p in frame_pairs if p.pair_index >= 0]
    ids = [p.pair_index for p in real_pairs]

    if len(ids) != expected_n:
        problems.append(
            f"pair_index 数量为 {len(ids)}，与预期帧对数 {expected_n} 不符"
        )

    # set(ids) 去重后如果比原列表短，说明有重复的 pair_index。
    if len(set(ids)) != len(ids):
        seen: set[int] = set()
        duplicates: list[int] = []
        for pid in ids:
            if pid in seen:
                duplicates.append(pid)
            seen.add(pid)
        problems.append(f"发现重复的 pair_index: {sorted(set(duplicates))}")

    return problems


def check_timing_quality_coverage(report: TimingQualityReport, expected_n: int) -> list[str]:
    """检查 timing_quality.csv（即 report.per_pair）是不是每个图像对恰好一行。

    这里采用"每个图像对一行、A/B 两路的诊断都在同一行的不同列里"的格式
    （而不是逐相机拆成两行），因为时间质量诊断（间隔、倒退、漂移）
    本质上是"针对一个图像对"的判断，拆成两行反而会把 drift（双路之间的量）
    重复写两遍，容易造成"两行数字对不上"的困惑。
    """
    problems: list[str] = []

    if len(report.per_pair) != expected_n:
        problems.append(
            f"timing_quality 记录数为 {len(report.per_pair)}，与预期帧对数 {expected_n} 不符"
        )

    ids = [p.pair_index for p in report.per_pair]
    if len(set(ids)) != len(ids):
        problems.append("timing_quality 里存在重复的 pair_index")

    return problems
