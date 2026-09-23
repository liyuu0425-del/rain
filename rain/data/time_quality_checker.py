"""TimeQualityChecker：检查逐帧设备时间戳有没有缺帧、重复、倒退、异常间隔。

这是阶段1验收标准明确要求的四类检查。阈值来源优先级：

    1. capture_metadata.json 里"实测"给出的 frame_gap_threshold_a/b_us、
       max_abs_relative_drift_us —— 这些是采集程序在真实数据上算出来的，
       不是本模块猜的。
    2. 如果某次采集的元数据缺这些字段（比如换了一版采集程序），退回
       配置文件里的"倍数"兜底（interval_outlier_multiplier /
       drift_outlier_multiplier），乘在"本批实测中位数"上——
       禁止在代码里写死具体的微秒数。

Python 语法说明：
- `try / except`：
  `try:` 包住"可能失败"的代码，`except 具体异常类型 as e:` 接住失败后
  执行补救逻辑。和"先判断条件再执行"（if）的区别是：try/except 允许
  "先假设能成功，出错了再处理"，对于"字段可能不存在"这种情况更直接。
  这个文件里用它处理 metadata 没有提供某个阈值字段的情况。
"""

from __future__ import annotations

import numpy as np

from .types import CaptureMetadata, PairTimingQC, TimingQualityReport, TimingRecord


class TimeQualityChecker:
    """输入逐帧时间记录，输出"每对图像的诊断表" + "整批汇总报告"。"""

    def __init__(
        self,
        interval_outlier_multiplier: float = 1.5,
        drift_outlier_multiplier: float = 1.5,
        duplicate_time_tolerance_us: float = 0.0,
    ):
        """
        Python 语法说明：
        - `def __init__(self, ...):` 是"构造函数"。`ClassName(参数...)`
          造对象时，Python 会自动调用这个方法，把传入的参数填进
          `self.xxx`，之后对象的其它方法都能通过 `self.xxx` 取到。
        - 参数写了默认值（`= 1.5`）的，调用时可以不传，用默认值；
          三个阈值全部来自 config/pipeline.yaml 的 time_quality 段，
          这里的默认值只是"配置文件缺这一项时"的兜底，不代表推荐值。
        """
        self.interval_outlier_multiplier = interval_outlier_multiplier
        self.drift_outlier_multiplier = drift_outlier_multiplier
        self.duplicate_time_tolerance_us = duplicate_time_tolerance_us

    def check(
        self,
        records: list[TimingRecord],
        metadata: CaptureMetadata | None = None,
    ) -> TimingQualityReport:
        """对一批已按 pair_index 排序的时间记录做四类检查。

        参数：
            records  : TimingReader 读出的时间表
            metadata : 同一次采集的 CaptureMetadata，用来取"实测阈值"；
                       传 None 时全部退回配置文件的倍数兜底。
        """
        if not records:
            raise ValueError("时间记录为空，无法做时间质量检查")

        missing_pair_indices = self._find_missing_pair_indices(records)

        # NumPy 语法说明：
        # `np.array([...])` 把一个 Python list 转成 NumPy 数组。数组和
        # list 最大的区别是可以整体做数学运算，比如 `arr2 - arr1` 会
        # 逐元素相减，不用自己写 for 循环。这个项目其它地方（背景建模、
        # 检测）已经大量用到这个特性，这里是"阶段1"里第一次出现。
        a_us = np.array([r.a_device_relative_us for r in records])
        b_us = np.array([r.b_device_relative_us for r in records])

        # np.diff：数组里相邻元素两两相减，得到"逐帧间隔"数组，
        # 长度比原数组少 1（第一帧没有"上一帧"可比）。
        diffs_a = np.diff(a_us)
        diffs_b = np.diff(b_us)
        all_diffs = np.concatenate([diffs_a, diffs_b])
        median_period = float(np.median(all_diffs))

        gap_threshold_a = self._resolve_gap_threshold(
            metadata.frame_gap_threshold_a_us if metadata else None, median_period
        )
        gap_threshold_b = self._resolve_gap_threshold(
            metadata.frame_gap_threshold_b_us if metadata else None, median_period
        )
        drift_threshold = self._resolve_drift_threshold(metadata, records)

        per_pair, n_backwards, n_duplicate, n_interval_outlier, n_drift_outlier = (
            self._diagnose_each_pair(records, gap_threshold_a, gap_threshold_b, drift_threshold)
        )

        return TimingQualityReport(
            n_records=len(records),
            measured_frame_period_median_us=float(np.median(all_diffs)),
            measured_frame_period_p95_us=float(np.percentile(all_diffs, 95)),
            measured_frame_period_max_us=float(np.max(all_diffs)),
            n_time_backwards=n_backwards,
            n_duplicate_time=n_duplicate,
            n_interval_outliers=n_interval_outlier,
            n_drift_outliers=n_drift_outlier,
            missing_pair_indices=missing_pair_indices,
            per_pair=per_pair,
        )

    # -------------------------------------------------------------------
    # 内部辅助方法（不对外暴露，仅供 check() 调用）
    # -------------------------------------------------------------------

    @staticmethod
    def _find_missing_pair_indices(records: list[TimingRecord]) -> list[int]:
        """检查 pair_index 序列里有没有断号（缺帧）。

        capture_timing.csv 本身应该是 0..N-1 连续的；这里不假设它连续，
        而是显式核对，断号即视为缺帧的证据之一（另一部分缺帧证据来自
        FramePairer 对实际图像文件的核对）。
        """
        present = {r.pair_index for r in records}  # 大括号 {} 造一个"集合"，自动去重
        lo, hi = min(present), max(present)
        # 集合减法 A - B：所有在 A 里但不在 B 里的元素。
        # 用它一步求出"期望出现但实际没出现"的 pair_index。
        expected = set(range(lo, hi + 1))
        missing = sorted(expected - present)
        return missing

    def _resolve_gap_threshold(self, metadata_value: float | None, median_period: float) -> float:
        """决定"帧间隔异常"的判定阈值：优先用元数据实测值，否则用倍数兜底。"""
        if metadata_value:
            return float(metadata_value)
        return median_period * self.interval_outlier_multiplier

    def _resolve_drift_threshold(
        self, metadata: CaptureMetadata | None, records: list[TimingRecord]
    ) -> float:
        """决定"双路时间漂移异常"的判定阈值。

        metadata 里的 max_abs_relative_drift_us 是"这批数据里实际出现过的
        最大漂移"，直接拿来当阈值会导致这批数据里的最大值恰好卡线、
        一个异常都报不出来，所以还要再乘一次安全倍数。
        """
        try:
            base = float(metadata.max_abs_relative_drift_us) if metadata else 0.0
        except (AttributeError, TypeError):
            base = 0.0

        if base <= 0.0:
            # metadata 没给、或者给的是 0：退回用本批数据自己估一个基准。
            drifts = np.abs([r.b_minus_a_relative_drift_us for r in records])
            base = max(float(np.median(drifts)), 1.0)

        return base * self.drift_outlier_multiplier

    def _diagnose_each_pair(
        self,
        records: list[TimingRecord],
        gap_threshold_a: float,
        gap_threshold_b: float,
        drift_threshold: float,
    ) -> tuple[list[PairTimingQC], int, int, int, int]:
        """逐条记录算诊断结果，同时累计四类异常各自的计数。"""
        per_pair: list[PairTimingQC] = []
        n_backwards = n_duplicate = n_interval_outlier = n_drift_outlier = 0

        prev: TimingRecord | None = None  # 上一条记录，用来算间隔；第一条没有上一条
        for rec in records:
            interval_a_us = interval_b_us = None
            is_backwards_a = is_backwards_b = False
            is_dup_a = is_dup_b = False

            if prev is not None:
                interval_a_us = rec.a_device_relative_us - prev.a_device_relative_us
                interval_b_us = rec.b_device_relative_us - prev.b_device_relative_us
                is_backwards_a = interval_a_us < 0
                is_backwards_b = interval_b_us < 0
                is_dup_a = abs(interval_a_us) <= self.duplicate_time_tolerance_us
                is_dup_b = abs(interval_b_us) <= self.duplicate_time_tolerance_us

            is_outlier_a = interval_a_us is not None and abs(interval_a_us) > gap_threshold_a
            is_outlier_b = interval_b_us is not None and abs(interval_b_us) > gap_threshold_b
            is_drift_outlier = abs(rec.b_minus_a_relative_drift_us) > drift_threshold

            per_pair.append(
                PairTimingQC(
                    pair_index=rec.pair_index,
                    a_device_relative_us=rec.a_device_relative_us,
                    b_device_relative_us=rec.b_device_relative_us,
                    interval_a_us=interval_a_us,
                    interval_b_us=interval_b_us,
                    drift_us=rec.b_minus_a_relative_drift_us,
                    is_time_backwards_a=is_backwards_a,
                    is_time_backwards_b=is_backwards_b,
                    is_duplicate_time_a=is_dup_a,
                    is_duplicate_time_b=is_dup_b,
                    is_interval_outlier_a=is_outlier_a,
                    is_interval_outlier_b=is_outlier_b,
                    is_drift_outlier=is_drift_outlier,
                )
            )

            if is_backwards_a or is_backwards_b:
                n_backwards += 1
            if is_dup_a or is_dup_b:
                n_duplicate += 1
            if is_outlier_a or is_outlier_b:
                n_interval_outlier += 1
            if is_drift_outlier:
                n_drift_outlier += 1

            prev = rec

        return per_pair, n_backwards, n_duplicate, n_interval_outlier, n_drift_outlier
