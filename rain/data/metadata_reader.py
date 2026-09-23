"""MetadataReader：读取 capture_metadata.json。

职责单一：只负责"文件 -> CaptureMetadata 对象"这一步转换，不做任何质量
判断（那是 TimeQualityChecker 的事）。这样以后想换一种元数据格式
（比如换一版采集程序、字段改名），只用改这一个文件。
"""

from __future__ import annotations

import json  # Python 标准库，负责把 JSON 文本解析成 dict / list / 数字 / 字符串
from pathlib import Path  # 面向对象的路径操作，比字符串拼路径更不容易出 \\ / 的问题

from .types import CaptureMetadata


class MetadataReader:
    """把 capture_metadata.json 解析成 CaptureMetadata。

    Python 语法说明：
    - `class MetadataReader:` 定义一个"类"，可以理解成"一类对象的模板"。
    - `def read(self, path):` 类里定义的函数叫"方法"。方法的第一个参数
      固定写作 `self`，代表"调用这个方法的对象自己"，由 Python 自动传入，
      调用时不用手写。例如：

          reader = MetadataReader()      # 造一个 MetadataReader 对象
          meta = reader.read("x.json")   # self 自动就是 reader，不用手动传

      这个类目前没有自己的状态（没有 __init__），之所以还封装成类而不是
      写成一个裸函数，是为了和项目里其它模块（Detector、Tracker……）保持
      同样的调用方式，方便以后统一接入 registry 注册表。
    """

    def read(self, path: str) -> CaptureMetadata:
        """读取一份 capture_metadata.json，返回 CaptureMetadata。

        参数：
            path : capture_metadata.json 的完整路径。A、B 两路目录下
                   各有一份，这个方法不关心调用方传的是哪一路的路径。

        失败时抛异常（文件不存在、JSON 格式错误、缺关键字段），不返回
        None —— 参考 rain/imageio.py 里的同一条约定：静默失败会让问题
        一路流到下游才暴露，非常难查。
        """
        # `Path(path).read_text(encoding="utf-8")` 一次性把整个文件读成字符串。
        # capture_metadata.json 只有几 KB，不需要流式读取。
        text = Path(path).read_text(encoding="utf-8")

        # json.loads 把 JSON 字符串解析成 Python 的 dict（大括号 {} 对应字典，
        # 方括号 [] 对应列表）。之后可以像 raw["capture_name"] 这样按键取值。
        raw: dict = json.loads(text)

        return CaptureMetadata(
            capture_name=raw["capture_name"],
            camera_a_sn=raw["camera_a_sn"],
            camera_b_sn=raw["camera_b_sn"],
            requested_frame_pairs=int(raw["requested_frame_pairs"]),
            requested_width=int(raw["requested_width"]),
            requested_height=int(raw["requested_height"]),
            requested_exposure_us=float(raw["requested_exposure_us"]),
            requested_analog_gain=int(raw["requested_analog_gain"]),
            a_trigger_mode=raw["a_trigger_mode"],
            b_trigger_mode=raw["b_trigger_mode"],
            captured_complete_frame_pairs=int(raw["captured_complete_frame_pairs"]),
            capture_complete=bool(raw["capture_complete"]),
            missing_frame_pairs=int(raw["missing_frame_pairs"]),
            valid_frame_pairs=int(raw["valid_frame_pairs"]),
            median_frame_period_a_us=float(raw["median_frame_period_a_us"]),
            median_frame_period_b_us=float(raw["median_frame_period_b_us"]),
            max_frame_period_a_us=float(raw["max_frame_period_a_us"]),
            max_frame_period_b_us=float(raw["max_frame_period_b_us"]),
            frame_gap_threshold_a_us=float(raw["frame_gap_threshold_a_us"]),
            frame_gap_threshold_b_us=float(raw["frame_gap_threshold_b_us"]),
            median_relative_drift_us=float(raw["median_relative_drift_us"]),
            p95_abs_relative_drift_us=float(raw["p95_abs_relative_drift_us"]),
            max_abs_relative_drift_us=float(raw["max_abs_relative_drift_us"]),
            timing_model=raw.get("timing_model", ""),
            alignment_note=raw.get("alignment_note", ""),
            raw=raw,
        )

    def compare(self, a: CaptureMetadata, b: CaptureMetadata) -> list[str]:
        """核对 A、B 两路各自读到的元数据是否一致，返回不一致项的说明列表。

        两路目录下的 capture_metadata.json 理论上内容完全相同（同一份采集
        记录，双路各存一份）。如果不一致，很可能是采集程序出过问题或者
        文件被后期误改过，必须报出来而不是悄悄用其中一份。

        返回空列表表示两份完全一致。
        """
        mismatches: list[str] = []

        # 这几个字段是双路共享、理应完全相等的核心字段。
        # 用 for 循环逐个比较，比每个字段单独写一个 if 更短、也更不容易漏。
        fields_to_compare = [
            "capture_name",
            "requested_frame_pairs",
            "requested_width",
            "requested_height",
            "requested_exposure_us",
            "captured_complete_frame_pairs",
            "capture_complete",
            "missing_frame_pairs",
            "valid_frame_pairs",
        ]
        for name in fields_to_compare:
            va, vb = getattr(a, name), getattr(b, name)
            if va != vb:
                mismatches.append(f"{name}: A={va!r} B={vb!r}")

        return mismatches
