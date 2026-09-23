"""阶段7性能测试基础设施的单元测试：StageProfiler / ResourceMonitor。

性能数值本身是不确定的（受机器负载影响），测试只验证"统计逻辑本身算得对"
和"结构完整"，不对具体耗时数字做断言。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rain.perf.profiler import StageProfiler
from rain.perf.resource_monitor import ResourceMonitor


def test_profiler_records_correct_call_count():
    p = StageProfiler()
    for _ in range(7):
        p.record("demo", 1.0)
    stats = p.stats()
    assert stats["demo"].n_calls == 7


def test_profiler_computes_mean_median_correctly_on_known_values():
    p = StageProfiler()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        p.record("demo", v)
    s = p.stats()["demo"]
    assert s.mean_ms == pytest.approx(3.0)
    assert s.median_ms == pytest.approx(3.0)
    assert s.max_ms == pytest.approx(5.0)
    assert s.total_ms == pytest.approx(15.0)


def test_profiler_p95_between_median_and_max():
    p = StageProfiler()
    for v in range(1, 101):  # 1..100
        p.record("demo", float(v))
    s = p.stats()["demo"]
    assert s.median_ms <= s.p95_ms <= s.max_ms


def test_profiler_timer_context_manager_records_duration():
    p = StageProfiler()
    with p.timer("sleep_demo"):
        time.sleep(0.01)
    stats = p.stats()
    assert stats["sleep_demo"].n_calls == 1
    assert stats["sleep_demo"].mean_ms > 0  # 真的记到了正的耗时


def test_profiler_timer_records_even_when_exception_raised():
    """异常场景下也必须记到耗时，不能因为报错就丢样本——这样性能剖析
    才能覆盖到"检测/匹配失败"这类异常路径的真实耗时。"""
    p = StageProfiler()
    with pytest.raises(ValueError):
        with p.timer("failing_stage"):
            raise ValueError("模拟失败")
    stats = p.stats()
    assert stats["failing_stage"].n_calls == 1


def test_profiler_multiple_stages_independent():
    p = StageProfiler()
    p.record("a", 1.0)
    p.record("a", 2.0)
    p.record("b", 100.0)
    stats = p.stats()
    assert stats["a"].n_calls == 2
    assert stats["b"].n_calls == 1
    assert stats["a"].mean_ms != stats["b"].mean_ms


def test_profiler_reset_clears_samples():
    p = StageProfiler()
    p.record("a", 1.0)
    p.reset()
    assert p.stats() == {}


def test_profiler_empty_stats_is_empty_dict():
    p = StageProfiler()
    assert p.stats() == {}


def test_profiler_raw_samples_returns_copy_not_reference():
    p = StageProfiler()
    p.record("a", 1.0)
    samples = p.raw_samples("a")
    samples.append(999.0)  # 改动返回值不应该影响内部状态
    assert p.raw_samples("a") == [1.0]


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


def test_resource_monitor_returns_structured_result():
    mon = ResourceMonitor(interval_s=0.01)
    mon.start()
    time.sleep(0.05)
    result = mon.stop()

    assert "peak_rss_mb" in result
    assert result["peak_rss_mb"] > 0  # 进程肯定占了一些内存，不可能是0
    assert "cpu_percent" in result
    assert isinstance(result["cpu_percent"], float)
    assert "disk_io" in result
    # disk_io 要么是 None（平台/权限不支持），要么是含 read/write 字节数的字典
    assert result["disk_io"] is None or {"read_bytes", "write_bytes"} <= set(result["disk_io"])


def test_resource_monitor_peak_rss_captures_allocation_during_run():
    """故意在采样期间分配一大块内存，峰值应该能捕捉到明显增长。"""
    mon = ResourceMonitor(interval_s=0.005)
    mon.start()
    _big = bytearray(50 * 1024 * 1024)  # 分配约50MB
    time.sleep(0.05)
    result = mon.stop()
    del _big
    assert result["peak_rss_mb"] > 10  # 远超"什么都不分配"时的正常水平


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
