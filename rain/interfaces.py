"""模块接口定义。

这个文件规定"每个模块必须提供哪些方法"。主程序只依赖这里的接口，
不认识任何具体算法，所以换算法不用改主程序。

Python 语法说明：
- Protocol 是"结构化子类型"（鸭子类型的正式写法）。
  实现类**不需要** `class XxxDetector(Detector)` 这样去继承，
  只要方法名和参数对得上，就自动被认为满足这个接口。
  这比传统的抽象基类(ABC)轻，加新实现时不用改动这个文件。
- `...` 是 Ellipsis 对象，在这里当作"函数体留空"用，等价于 pass。
- @runtime_checkable 让 isinstance(obj, Detector) 能用（只检查方法是否存在）。
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from .types import BatchPair, Detection, Track


@runtime_checkable
class FrameReader(Protocol):
    """读一帧图像。

    不同实现对应不同存储格式：
      - lzw_tif : 历史数据，LZW 压缩 TIF，解码约 21 ms/帧
      - raw_bin : 新数据，未压缩裸二进制，读取约 2 ms/帧
    """

    def read(self, path: str) -> np.ndarray:
        """返回 uint8 二维灰度数组，shape = (H, W)。

        约定：
        - 必须返回 uint8，不要返回 float。float64 会让后续内存带宽翻 4 倍。
        - 读失败时抛异常，不要返回 None（cv2.imread 返回 None 的行为很难排查）。
        """
        ...


@runtime_checkable
class BackgroundModel(Protocol):
    """建立背景图。

    不同实现：
      - batch_quantile : 用批次自己的 180 帧算逐像素分位数
      - external_ref   : 用专门采集的无雨背景参考批次
    """

    def build(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """返回与单帧同尺寸的 uint8 背景图。

        frames 可能是生成器，实现里只允许遍历一次。
        """
        ...


@runtime_checkable
class Detector(Protocol):
    """单帧检测：从一帧图像里找出候选雨滴。

    这是最可能被替换的模块，也是接口设计的主要考验对象。
    """

    def detect(
        self,
        img: np.ndarray,
        bg: np.ndarray,
        frame_idx: int = 0,
        debug: dict | None = None,
    ) -> list[Detection]:
        """返回该帧的候选雨滴列表。

        参数：
          img       : uint8 灰度帧
          bg        : uint8 背景图，与 img 同尺寸
          frame_idx : 帧号，原样写进每个 Detection，实现不要自己编号
          debug     : 传 None 时（主流程的默认做法）不产出任何中间结果，零开销；
                      传一个 dict 时，实现应把中间图塞进去，例如
                      debug["diff"] = diff; debug["binary"] = bw
                      中间图绝不能进返回值——否则每个实现都被迫产出，
                      而且跨线程传 3 MB 数组。

        约定：
        - 空图（没有目标）必须返回空列表 []，不是 None。
        - 不得修改传入的 img 和 bg（其他线程可能在共享它们）。
        - 返回的坐标必须是全图坐标；如果实现内部裁了 ROI，要把偏移加回去。
        """
        ...


@runtime_checkable
class Tracker(Protocol):
    """批内单路跟踪：把逐帧的检测串成轨迹。

    跟踪是有状态、顺序依赖的，所以它留在批内串行执行。
    这不影响性能——它只处理稀疏的目标列表，不碰整幅图像。
    """

    def track(self, per_frame: list[list[Detection]], cam: str) -> list[Track]:
        """输入按帧顺序排列的检测列表，返回轨迹列表。

        约定：
        - per_frame[i] 是第 i 帧的检测结果，i 就是批内顺序。
        - 必须是纯函数：同样的输入永远给出同样的输出（含 track_id 的编号顺序），
          否则并行与单线程的结果对不上，回归测试会失败。
        """
        ...


@runtime_checkable
class StereoMatcher(Protocol):
    """双路匹配：判断两个视角中哪些目标属于同一空间雨滴。

    当前是占位。双相机内外参标定完成前，这个模块不能实现真正的极线约束。
    """

    def match(
        self,
        tracks_a: list[Track],
        tracks_b: list[Track],
        batch: BatchPair,
    ) -> list[tuple[int, int, float]]:
        """返回 [(track_id_a, track_id_b, cost), ...]。

        约定：匹配失败的目标不得强行配对。它们应保留单路记录，
        由调用方标上失败原因和低置信度。
        """
        ...
