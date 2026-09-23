"""StageProfiler：记录各模块耗时样本，算均值/中位数/P95/最大值/调用次数。

Python 语法说明：
- `with profiler.timer("detect"):` 这种写法叫"上下文管理器"（context
  manager）。`_Timer` 类实现了 `__enter__`/`__exit__` 两个方法，
  Python 在进入/离开 `with` 代码块时自动调用它们——进入时记录起始时间，
  离开时（哪怕代码块中途抛异常）都会计算耗时并记下来。这样计时代码不用
  在每个调用点手写"记开始时间、算差值、存起来"三行样板代码。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class StageStats:
    """一个阶段的耗时统计（对应 performance_report.csv 的一行）。"""

    stage: str
    n_calls: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    total_ms: float


class StageProfiler:
    """按阶段名字收集耗时样本，随时能导出统计表。"""

    def __init__(self):
        # dict[str, list[float]]：键是阶段名字，值是这个阶段每次调用的耗时(ms)列表。
        self._samples: dict[str, list[float]] = {}

    def record(self, stage: str, duration_ms: float) -> None:
        """手动记一条耗时样本（不想用 with 语法时可以直接调用这个）。"""
        self._samples.setdefault(stage, []).append(duration_ms)

    def timer(self, stage: str) -> "_Timer":
        """返回一个上下文管理器，用法：`with profiler.timer("detect"): ...`。"""
        return _Timer(self, stage)

    def stats(self) -> dict[str, StageStats]:
        """把收集到的样本汇总成统计表，键是阶段名字。"""
        result: dict[str, StageStats] = {}
        for stage, samples in self._samples.items():
            if not samples:
                continue
            arr = np.array(samples)
            result[stage] = StageStats(
                stage=stage,
                n_calls=len(arr),
                mean_ms=float(np.mean(arr)),
                median_ms=float(np.median(arr)),
                p95_ms=float(np.percentile(arr, 95)),
                max_ms=float(np.max(arr)),
                total_ms=float(np.sum(arr)),
            )
        return result

    def reset(self) -> None:
        self._samples.clear()

    def raw_samples(self, stage: str) -> list[float]:
        return list(self._samples.get(stage, []))


class _Timer:
    """`StageProfiler.timer()` 返回的上下文管理器对象，不单独使用。"""

    def __init__(self, profiler: StageProfiler, stage: str):
        self.profiler = profiler
        self.stage = stage
        self._t0: float | None = None

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self.profiler.record(self.stage, dt_ms)
        return False  # 不吞异常：有异常照样往外传，只是顺手把耗时记一下
