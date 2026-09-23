"""FramePairer：按 pair_index 和同名文件配对 A/B 两路图像。

任务书明确规定的配对依据：
    "相同pair_index和同名文件属于同一图像对"
    "文件名中的毫秒时间取A帧主机接收墙钟时间，不等同于曝光时刻"
所以这里只看"pair_index 对应的文件名是否在两路目录下都真实存在"，
不解析、不使用文件名里的时间字符串做任何计算。

这个模块和 rain/io_index.py 的 pair_batches() 不是一回事：
    - pair_batches() 面向老 Demo 的数据布局（一个根目录下有很多个
      Capture_xxx 子目录，每个子目录是一"批"）；
    - FramePairer 面向任务书里描述的新数据布局（dir_a / dir_b 各自
      就是一次采集的目录，图像直接放在里面，旁边带 capture_timing.csv）。
两种布局都是真实存在的数据，所以两套配对逻辑并存，互不覆盖。
"""

from __future__ import annotations

import os

from .types import FramePairRecord, TimingRecord

_IMAGE_EXT = ".tif"


class FramePairer:
    """把 A/B 两路目录下的图像文件，按 timing 记录里的 pair_index 配对。"""

    def pair(
        self,
        dir_a: str,
        dir_b: str,
        timing_records: list[TimingRecord],
    ) -> list[FramePairRecord]:
        """返回配对结果列表：先按 pair_index 覆盖的部分，再附上多余文件。

        参数：
            dir_a / dir_b   : 两路图像所在目录
            timing_records  : TimingReader 读到的时间表，提供
                               "pair_index -> 文件名" 的权威映射——
                               权威来源是 capture_timing.csv，不是
                               目录里恰好存在哪些文件。
        """
        files_a = self._list_tif(dir_a)
        files_b = self._list_tif(dir_b)

        pairs: list[FramePairRecord] = []
        for rec in timing_records:
            pairs.append(self._pair_one(rec, dir_a, dir_b, files_a, files_b))

        # timing.csv 里没记录、但目录里存在的文件——同样要报出来，
        # 不能悄悄忽略。可能是上一次采集残留的文件，也可能是
        # timing.csv 本身漏记了某些帧，两种情况都需要人工确认。
        referenced = {rec.pair_filename for rec in timing_records}
        pairs.extend(self._extra_file_records(files_a, referenced, dir_a, is_a=True))
        pairs.extend(self._extra_file_records(files_b, referenced, dir_b, is_a=False))

        return pairs

    @staticmethod
    def _pair_one(
        rec: TimingRecord,
        dir_a: str,
        dir_b: str,
        files_a: set[str],
        files_b: set[str],
    ) -> FramePairRecord:
        fn = rec.pair_filename
        has_a = fn in files_a
        has_b = fn in files_b
        path_a = os.path.join(dir_a, fn) if has_a else None
        path_b = os.path.join(dir_b, fn) if has_b else None

        # if / elif / else：按顺序判断，第一个满足的分支执行完就跳出，
        # 不会继续往下判断。四种组合（A/B 各有/没有）正好对应四个分支。
        if has_a and has_b:
            status, note = "ok", ""
        elif has_a and not has_b:
            status, note = "missing_b", f"B 路缺少文件 {fn}"
        elif has_b and not has_a:
            status, note = "missing_a", f"A 路缺少文件 {fn}"
        else:
            status, note = "missing_both", f"A/B 两路都缺少文件 {fn}"

        return FramePairRecord(
            pair_index=rec.pair_index,
            filename=fn,
            path_a=path_a,
            path_b=path_b,
            status=status,
            note=note,
        )

    @staticmethod
    def _extra_file_records(
        files: set[str], referenced: set[str], folder: str, is_a: bool
    ) -> list[FramePairRecord]:
        extra = sorted(files - referenced)  # 集合减法：目录里有、但 timing.csv 没提到的文件名
        records = []
        for fn in extra:
            records.append(
                FramePairRecord(
                    pair_index=-1,  # -1 表示"不属于任何已知 pair_index"，便于在 CSV 里一眼识别
                    filename=fn,
                    path_a=os.path.join(folder, fn) if is_a else None,
                    path_b=None if is_a else os.path.join(folder, fn),
                    status="extra_a" if is_a else "extra_b",
                    note=f"{'A' if is_a else 'B'} 路存在但 timing.csv 未记录该文件",
                )
            )
        return records

    @staticmethod
    def _list_tif(folder: str) -> set[str]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"图像目录不存在: {folder}")
        return {fn for fn in os.listdir(folder) if fn.lower().endswith(_IMAGE_EXT)}
