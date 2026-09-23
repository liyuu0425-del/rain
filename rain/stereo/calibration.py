"""标定文件加载：CalibrationLoader。

任务书的硬性约束：没有真实双相机标定文件时，不得虚构内外参或极线参数。
所以这个模块的核心行为不是"读取标定"，而是"诚实地报告标定在不在"：
文件不存在时返回 None，调用方（EpipolarGate）必须能正确处理 None，
而不是拿一个瞎猜的矩阵去算"看起来正常"但毫无意义的极线距离。
"""

from __future__ import annotations

import json
import os

import numpy as np

from .types import StereoCalibration


class CalibrationLoader:
    """从 JSON 文件加载双相机标定。文件格式：

        {
          "fundamental_matrix": [[f11,f12,f13],[f21,f22,f23],[f31,f32,f33]],
          "rectify_a_to_b": [[...],[...],[...]]   // 可选
        }

    真实标定完成前，这个文件不应该存在——config/pipeline.yaml 里配置的
    路径目前指向一个不存在的文件，load() 会返回 None，这是预期行为，
    不是 bug。
    """

    def load(self, path: str) -> StereoCalibration | None:
        if not path or not os.path.isfile(path):
            return None

        raw = json.loads(open(path, "r", encoding="utf-8").read())
        F = np.array(raw["fundamental_matrix"], dtype=np.float64)
        if F.shape != (3, 3):
            raise ValueError(f"标定文件 {path} 的 fundamental_matrix 不是 3x3: {F.shape}")

        rectify = raw.get("rectify_a_to_b")
        rectify_arr = np.array(rectify, dtype=np.float64) if rectify is not None else None
        if rectify_arr is not None and rectify_arr.shape != (3, 3):
            raise ValueError(f"标定文件 {path} 的 rectify_a_to_b 不是 3x3: {rectify_arr.shape}")

        return StereoCalibration(fundamental_matrix=F, rectify_a_to_b=rectify_arr, source_path=path)
