"""ResourceMonitor：后台采样进程内存峰值，配合 psutil 读 CPU% 和磁盘IO计数。

内存峰值不能只在"运行前"和"运行后"各测一次就取较大值——真正的峰值
可能出现在运行"中间"的某一瞬间（比如背景建模阶段一次性把180帧堆进内存）。
所以用一个后台线程每隔固定时间采样一次，取所有采样里的最大值。

Python 语法说明：
- `threading.Thread(target=..., daemon=True)`：另起一条线程跑
  `target` 指向的函数，`daemon=True` 表示这条线程不会阻止主程序退出
  （测量线程只是辅助，主程序结束时不需要等它）。
- `threading.Event()`：一个可以在多线程之间传递"信号"的对象。
  `.wait(超时)` 会阻塞到超时或者被 `.set()` 唤醒；用它实现"每隔
  interval_s 秒采样一次，且能被随时叫停"，比自己写 `time.sleep` +
  检查标志位更不容易漏掉停止信号。
"""

from __future__ import annotations

import os
import threading

import psutil


class ResourceMonitor:
    """后台采样当前进程的内存占用峰值；配合 psutil 读取 CPU% 和磁盘 IO 计数。"""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._process = psutil.Process(os.getpid())
        self._peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._io_start = None
        self._cpu_times_start = None

    def start(self) -> None:
        self._stop_event.clear()
        self._peak_rss_bytes = self._process.memory_info().rss
        # cpu_percent 第一次调用只是建立基线，返回值没意义，故意丢弃。
        self._process.cpu_percent(interval=None)
        try:
            self._io_start = self._process.io_counters()
        except (AttributeError, psutil.AccessDenied, NotImplementedError):
            # 部分平台/权限下拿不到磁盘IO计数，明确记None，不假装有数据。
            self._io_start = None
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self._peak_rss_bytes:
                    self._peak_rss_bytes = rss
            except psutil.Error:
                pass  # 采样线程不能把主流程打断，任何异常都吞掉只是不更新峰值
            self._stop_event.wait(self.interval_s)

    def stop(self) -> dict:
        """停止采样，返回这段时间的资源占用摘要。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        cpu_percent = self._process.cpu_percent(interval=None)

        io_delta = None
        if self._io_start is not None:
            try:
                io_end = self._process.io_counters()
                io_delta = {
                    "read_bytes": io_end.read_bytes - self._io_start.read_bytes,
                    "write_bytes": io_end.write_bytes - self._io_start.write_bytes,
                }
            except (AttributeError, psutil.AccessDenied, NotImplementedError):
                io_delta = None

        return {
            "peak_rss_mb": self._peak_rss_bytes / (1024 * 1024),
            "cpu_percent": cpu_percent,
            "disk_io": io_delta,  # None 表示这个平台/权限下测不到，不是"0字节"
        }
