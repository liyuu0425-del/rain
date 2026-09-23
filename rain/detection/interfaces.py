"""阶段2六个检测子模块的接口定义。

和 `rain/interfaces.py` 是同一类东西（用 Protocol 定义"这个模块必须提供
哪个方法、输入输出是什么"），单独放一份文件是因为它们服务于阶段2新拆出来
的子步骤，不属于原有 Detector/Tracker 等顶层协议，混在一起会让原文件的
"五大模块"结构变得不清楚。

六步串起来的完整数据流（对照 `ModularPipelineDetector.detect()`）：

    原始帧 img (uint8, H×W)
        │  BrightnessNormalizer.normalize
        ▼
    归一化帧 (uint8, H×W)
        │  BackgroundDifferencer.diff(归一化帧, bg)
        ▼
    差分图 diff (uint8, H×W；雨滴处更亮)
        │  AdaptiveThresholdSegmenter.segment
        ▼
    二值图 binary (uint8, H×W；0/255)
        │  MorphologyProcessor.process
        ▼
    形态学处理后的二值图 (uint8, H×W)
        │  ConnectedComponentAnalyzer.analyze
        ▼
    轮廓列表 contours (list[np.ndarray]，每个 shape=(n,1,2) int32)
        │  ContourExtractor.extract
        ▼
    list[Detection]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import Detection


@runtime_checkable
class BrightnessNormalizer(Protocol):
    """第1步：亮度归一化。

    原有 Demo 完全没有这一步（直接用原始灰度做差分）。加入它是因为
    实测同一批数据里逐帧平均灰度会有小幅波动（frame_quality.csv 里
    A 路第0帧均值 99.4，其余帧多在 97.6~97.7），差分+固定阈值的组合
    对这种波动没有任何补偿。
    """

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """返回亮度校正后的 uint8 图，形状与 img 相同。

        约定：不得修改传入的 img（其他线程可能在共享它）。
        """
        ...


@runtime_checkable
class BackgroundDifferencer(Protocol):
    """第2步：背景差分。

    背光成像下雨滴是暗目标，所以用 bg - img（饱和减法），差值越大代表
    该像素越暗、越可能是雨滴。这一步只做减法，不判断阈值——判断阈值是
    下一步 AdaptiveThresholdSegmenter 的职责，两者故意分开，方便单独
    替换减法策略（比如换成双向 absdiff）而不影响阈值算法。
    """

    def diff(self, img: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """返回 uint8 差分图，形状与 img/bg 相同。"""
        ...


@runtime_checkable
class AdaptiveThresholdSegmenter(Protocol):
    """第3步：（自适应）阈值分割。

    原 Demo 用的是全局固定阈值（`cv2.threshold`），docstring 里明确记录了
    "比 adaptiveThreshold 快 28 倍"这个取舍。这个协议同时支持固定阈值和
    真正的自适应阈值（`cv2.adaptiveThreshold`），通过构造参数 method 切换，
    默认仍是固定阈值——这样才能和原 Demo 做逐帧比较，同时又满足"自适应
    阈值分割器"这个模块名字要求的能力。
    """

    def segment(self, diff: np.ndarray) -> np.ndarray:
        """返回 uint8 二值图（像素值只有 0 或 255），形状与 diff 相同。"""
        ...


@runtime_checkable
class MorphologyProcessor(Protocol):
    """第4步：形态学处理（原 Demo 只做开运算，去掉孤立噪点）。"""

    def process(self, binary: np.ndarray) -> np.ndarray:
        """返回处理后的 uint8 二值图，形状与输入相同。"""
        ...


@runtime_checkable
class ConnectedComponentAnalyzer(Protocol):
    """第5步：连通域分析。

    原 Demo 用 `cv2.findContours` 一次性把"找连通域"和"取轮廓"做完
    （docstring 明确记录了弃用 `connectedComponentsWithStats` 的理由：
    快 7.5 倍）。这里把"找连通域"单独拆出来，输出仍然是轮廓点集列表
    （每个连通域的外轮廓），因为下一步 ContourExtractor 需要真实的轮廓
    点才能算周长、拟合椭圆——如果换成 `connectedComponentsWithStats`
    的 label 图，要重新找边界才能拿到同样的信息，等于绕一圈。
    """

    def analyze(self, binary: np.ndarray) -> list[np.ndarray]:
        """返回轮廓列表。每个元素是 shape=(n,1,2)、dtype=int32 的点集，
        与 `cv2.findContours` 返回的 contours 格式完全一致。
        """
        ...


@runtime_checkable
class ContourExtractor(Protocol):
    """第6步：从轮廓算特征、做尺寸/形状/触边 QC，产出最终的 Detection。"""

    def extract(
        self,
        contours: list[np.ndarray],
        img_shape: tuple[int, int],
        frame_idx: int,
    ) -> list[Detection]:
        """返回该帧的候选雨滴列表。

        参数：
            contours  : ConnectedComponentAnalyzer 的输出
            img_shape : (H, W)，用于判断轮廓是否触边
            frame_idx : 帧号，原样写进每个 Detection
        """
        ...
