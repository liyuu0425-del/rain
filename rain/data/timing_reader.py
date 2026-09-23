"""TimingReader：读取 capture_timing.csv 里的逐帧设备时间戳。

capture_timing.csv 每一行是一个已经配对好的图像对（A、B 各一帧），字段
很多（host 时间、原始 64 位时间戳、增益、字节数……）。这个模块只挑出
"配对 + 时间质量检查 + 后续测速"真正需要用到的字段，塞进 TimingRecord。
"""

from __future__ import annotations

import csv  # Python 标准库，逐行读写 CSV 文本文件

from .types import TimingRecord


class TimingReader:
    """把 capture_timing.csv 解析成 list[TimingRecord]（按 pair_index 升序）。"""

    def read(self, path: str) -> list[TimingRecord]:
        """读取一份 capture_timing.csv。

        约定（来自任务书的明确要求）：
        - 测速只用 a_device_relative_us / b_device_relative_us；
        - 不使用文件名里的毫秒、host 接收时间、"帧号 / 固定帧率"这几种
          替代算法——capture_metadata.json 的 alignment_note 已经说明
          host 接收时间差不等于真实曝光时差。
          这个类只负责"读出来"，不在这里做这类禁止性检查，检查交给
          TimeQualityChecker，保持单一职责。
        """
        records: list[TimingRecord] = []

        # `with open(...) as f:` 打开文件并绑定到变量 f；不管代码块正常
        # 结束还是中途抛异常，离开这个缩进块时 Python 都会自动关闭文件，
        # 不用自己写 f.close()。这种写法叫"上下文管理器"（context manager）。
        # encoding="utf-8-sig"：capture_timing.csv 可能带 BOM 头（常见于
        # Windows 程序写出的 UTF-8 文件），用 -sig 能自动吃掉 BOM，
        # 不然第一列的列名会变成 "﻿pair_index" 导致按名取值失败。
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            # csv.DictReader 用文件第一行当表头，把之后每一行都读成
            # {列名: 字符串值} 的字典，比自己按逗号切分再对位置省心得多。
            reader = csv.DictReader(f)
            for row in reader:  # for 循环：依次取出每一行字典
                records.append(self._row_to_record(row))

        # 正常情况下 pair_index 本来就是升序，这里显式排一次是防御性写法：
        # 就算源文件因为某种原因被打乱过行序，下游的"相邻帧间隔"计算
        # 也不会因此算错。
        records.sort(key=lambda r: r.pair_index)
        return records

    @staticmethod
    def _row_to_record(row: dict) -> TimingRecord:
        """把 DictReader 读出的一行（值全是字符串）转换成带正确类型的 TimingRecord。

        `@staticmethod`：表示这个方法不需要用到 self（不依赖对象自身的
        任何状态），只是借着类名做个命名空间。调用时不用先造对象：
            TimingReader._row_to_record(row)
        也可以正常调用，写成静态方法主要是让读者一看就知道"这段逻辑跟
        对象状态无关"。
        """
        return TimingRecord(
            pair_index=int(row["pair_index"]),
            pair_filename=row["pair_filename"],
            a_device_relative_us=float(row["a_device_relative_us"]),
            b_device_relative_us=float(row["b_device_relative_us"]),
            b_minus_a_relative_drift_us=float(row["b_minus_a_relative_drift_us"]),
            a_exposure_us=float(row["a_exposure_us"]),
            b_exposure_us=float(row["b_exposure_us"]),
            a_analog_gain=float(row["a_analog_gain"]),
            b_analog_gain=float(row["b_analog_gain"]),
            a_host_receive_wall_ns=int(row["a_host_receive_wall_ns"]),
            b_host_receive_wall_ns=int(row["b_host_receive_wall_ns"]),
            a_is_trigger_frame=int(row["a_is_trigger_frame"]),
            b_is_trigger_frame=int(row["b_is_trigger_frame"]),
        )
