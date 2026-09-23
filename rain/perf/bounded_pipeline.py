"""BoundedPipelineRunner：读取线程 -> 有界队列 -> 处理线程 -> 有界队列 -> 写出线程。

阶段7B性能优化新增，纯粹是"执行架构"基础设施，不是可替换算法，不走
registry 注册表。给它什么样的 read_fn/process_fn/write_fn，它就跑什么——
本身不包含任何检测/跟踪/匹配/测速/统计逻辑，那些逻辑由调用方（
pipeline/stage7b_optimization.py）传进来的函数提供，且那些函数直接
复用 rain.detection / rain.tracking 等已有模块，不是重新实现。

Python 语法说明：
- `queue.Queue(maxsize=N)`：标准库的线程安全队列，`maxsize` 就是"有界"
  的关键——`put()` 在队列满时会阻塞等待，不会无限堆积，这是内存不会
  无限增长的直接保证（阶段7B要求4"队列必须有最大长度"）。
- 用 `None` 作为"结束标记"塞进队列，是多线程生产者-消费者模式的常见
  写法：消费者线程收到 None 就知道"不会再有新任务了，可以退出"，
  比用一个共享的"停止标志位"更不容易漏判断（标志位需要额外加锁，
  还可能在检查和真正退出之间的空档漏掉最后几个任务）。
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class QueueStats:
    """一个队列在整次运行过程中的峰值长度和总放入次数（判断有没有"积压"）。"""

    max_size_observed: int = 0
    put_count: int = 0


@dataclass
class PipelineRunResult:
    """BoundedPipelineRunner.run() 结束后的运行统计。"""

    read_queue_stats: QueueStats = field(default_factory=QueueStats)
    write_queue_stats: QueueStats = field(default_factory=QueueStats)
    n_items: int = 0
    n_errors: int = 0


class BoundedPipelineRunner:
    """三段式有界队列流水线：读取线程池 -> 处理线程池 -> 单写出线程。

    参数：
        items                : 输入项列表（比如 [(pair_index, path), ...]）
        read_fn              : item -> 中间结果，在"读取线程"里跑（I/O+解码）
        process_fn           : 中间结果 -> 输出结果，在"处理线程"里跑；
                                传 None 表示这一段不做任何处理，读取结果
                                原样传给写出线程（用于"只测读取"这种场景）。
        write_fn             : 输出结果 -> None，在单独一条"写出线程"里跑
                                （单线程执行，避免多个线程同时写同一个
                                目标——比如同一个 dict 或同一个文件——
                                产生竞争）。
        n_reader_threads     : 读取线程数
        n_processor_threads  : 处理线程数（process_fn=None 时仍然会启动
                                这些线程，只是它们只做"原样转发"）
        queue_maxsize        : 两级队列的最大长度（有界，防止内存无限增长）

    结果顺序：写出线程按"任务完成的先后顺序"消费，不是输入顺序——
    如果调用方需要保序（比如按 pair_index 排序后再喂给跟踪器），
    请在 write_fn 里自己用某个字段重新排序，这个类不做这个假设。
    """

    def __init__(
        self,
        items: list,
        read_fn: Callable,
        write_fn: Callable,
        process_fn: Callable | None = None,
        n_reader_threads: int = 4,
        n_processor_threads: int = 1,
        queue_maxsize: int = 8,
    ):
        if queue_maxsize < 1:
            raise ValueError(f"queue_maxsize 必须>=1（有界队列不能没有上限），收到 {queue_maxsize}")
        self.items = items
        self.read_fn = read_fn
        self.process_fn = process_fn
        self.write_fn = write_fn
        self.n_reader_threads = n_reader_threads
        self.n_processor_threads = n_processor_threads
        self.queue_maxsize = queue_maxsize

        self._read_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._write_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._lock = threading.Lock()
        self._read_stats = QueueStats()
        self._write_stats = QueueStats()
        self._errors: list[BaseException] = []

    def run(self) -> PipelineRunResult:
        # 输入项本身也用一个队列分发给多个读取线程"抢活干"，
        # 不用提前按线程数切片——切片容易导致"有的线程分到的活多、
        # 有的分到的活少"，用共享队列天然做到负载均衡。
        item_queue: queue.Queue = queue.Queue()
        for it in self.items:
            item_queue.put(it)

        readers = [
            threading.Thread(target=self._reader_loop, args=(item_queue,), daemon=True)
            for _ in range(self.n_reader_threads)
        ]
        processors = [
            threading.Thread(target=self._processor_loop, daemon=True)
            for _ in range(self.n_processor_threads)
        ]
        writer = threading.Thread(target=self._writer_loop, daemon=True)

        for t in readers:
            t.start()
        for t in processors:
            t.start()
        writer.start()

        for t in readers:
            t.join()
        for _ in processors:  # 每个处理线程各发一个结束标记
            self._read_queue.put(None)
        for t in processors:
            t.join()
        self._write_queue.put(None)  # 写出线程的结束标记
        writer.join()

        if self._errors:
            raise self._errors[0]

        return PipelineRunResult(
            read_queue_stats=self._read_stats,
            write_queue_stats=self._write_stats,
            n_items=len(self.items),
            n_errors=len(self._errors),
        )

    def _reader_loop(self, item_queue: queue.Queue) -> None:
        while True:
            try:
                item = item_queue.get_nowait()
            except queue.Empty:
                return
            if self._errors:  # 别的线程已经出错了，尽快停下不再产生新任务
                return
            try:
                mid = self.read_fn(item)
            except BaseException as exc:  # noqa: BLE001 - 记下错误，不让一条线程的异常悄悄消失
                with self._lock:
                    self._errors.append(exc)
                return
            self._read_queue.put(mid)
            with self._lock:
                self._read_stats.put_count += 1
                self._read_stats.max_size_observed = max(
                    self._read_stats.max_size_observed, self._read_queue.qsize()
                )

    def _processor_loop(self) -> None:
        while True:
            mid = self._read_queue.get()
            if mid is None:
                return
            try:
                out = mid if self.process_fn is None else self.process_fn(mid)
            except BaseException as exc:  # noqa: BLE001
                with self._lock:
                    self._errors.append(exc)
                return
            self._write_queue.put(out)
            with self._lock:
                self._write_stats.put_count += 1
                self._write_stats.max_size_observed = max(
                    self._write_stats.max_size_observed, self._write_queue.qsize()
                )

    def _writer_loop(self) -> None:
        while True:
            out = self._write_queue.get()
            if out is None:
                return
            try:
                self.write_fn(out)
            except BaseException as exc:  # noqa: BLE001
                with self._lock:
                    self._errors.append(exc)
                return
