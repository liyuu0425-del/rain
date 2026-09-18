"""数据契约。

这个文件定义"模块之间传什么"。它是整套可替换接口的基础：
只要这里的字段不变，换掉任何一个算法实现，下游代码都不用改。

Python 语法说明：
- @dataclass 是一个"装饰器"，加在 class 上面，Python 会自动帮你生成
  __init__ 构造函数、__repr__ 打印格式等。省掉大量样板代码。
- 冒号后面的 `int` / `float` 是"类型注解"，只是写给人和工具看的提示，
  Python 运行时并不强制检查。但它能让编辑器帮你补全和查错。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 质量控制位标记（QC flags）
#
# 用"位运算"把多个是/否标记压进一个整数。比如一个目标既触边又疑似粘连，
# 它的 qc_flags 就是 QC_TOUCHES_BORDER | QC_MERGED == 1 | 2 == 3。
# 判断某一位是否置上：  if det.qc_flags & QC_TOUCHES_BORDER:
# ---------------------------------------------------------------------------
QC_OK = 0
QC_TOUCHES_BORDER = 1 << 0  # 触碰图像/ROI 边界，面积不完整
QC_MERGED = 1 << 1          # 疑似多滴粘连
QC_TOO_SMALL = 1 << 2       # 面积小于下限
QC_TOO_LARGE = 1 << 3       # 面积大于上限
QC_ODD_SHAPE = 1 << 4       # 形状异常（过于细长等）
QC_SATURATED = 1 << 5       # 像素饱和
QC_STATIC = 1 << 6          # 疑似附着在镜头上的静止污点

QC_NAMES = {
    QC_TOUCHES_BORDER: "touches_border",
    QC_MERGED: "merged",
    QC_TOO_SMALL: "too_small",
    QC_TOO_LARGE: "too_large",
    QC_ODD_SHAPE: "odd_shape",
    QC_SATURATED: "saturated",
    QC_STATIC: "static",
}


def qc_to_text(flags: int) -> str:
    """把 QC 位标记翻译成可读文本，写进 CSV 方便人工复核。"""
    if flags == QC_OK:
        return "ok"
    names = [name for bit, name in QC_NAMES.items() if flags & bit]
    return "|".join(names)


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """单帧里检测到的一个候选雨滴。

    这是整个项目最重要的数据契约。所有 Detector 实现都必须返回
    list[Detection]，字段含义必须完全一致。

    重要约定：新增算法如果想多输出一个量（例如拟合椭圆的长短轴），
    请在下面"可选扩展字段"区加带默认值的字段，不要改动已有字段的
    名字和含义。否则 qc / physics / sink 全都要跟着改，
    "换算法主程序不受影响"就失效了。

    单位说明：当前全部是像素。像素-毫米标定完成前，不允许出现毫米量。
    """

    frame_idx: int                      # 该帧在批次内的序号（来自文件名 frame_XXX）
    cx: float                           # 质心 x，像素（全图坐标，已加回 ROI 偏移）
    cy: float                           # 质心 y，像素
    area_px: float                      # 投影面积，像素
    bbox: tuple[int, int, int, int]     # 包围框 (x, y, w, h)，像素
    d_eq_px: float                      # 像素等效直径 = 2*sqrt(area_px/pi)
    qc_flags: int = QC_OK               # 质量位标记，见上方常量

    # --- 可选扩展字段：新算法可以填，下游必须容忍它们是 None ---
    contour: np.ndarray | None = None   # 轮廓点，Nx1x2 int32。可能很占内存
    perimeter_px: float | None = None   # 周长
    major_axis_px: float | None = None  # 拟合椭圆长轴
    minor_axis_px: float | None = None  # 拟合椭圆短轴

    @staticmethod
    def d_eq_from_area(area_px: float) -> float:
        """等效直径公式，集中在一处，避免各实现算得不一致。

        D_eq = 2 * sqrt(A / pi)
        注意：这里的 A 是像素面积，所以结果是"像素等效直径"。
        换算成毫米需要像素-毫米标定，当前尚未完成。
        """
        return 2.0 * float(np.sqrt(area_px / np.pi))


@dataclass
class Track:
    """单路图像里，一个雨滴跨若干帧的轨迹。"""

    track_id: int
    cam: str                            # "a" 或 "b"
    detections: list[Detection] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.detections)

    @property
    def frame_span(self) -> tuple[int, int]:
        """轨迹覆盖的首末帧号。"""
        idxs = [d.frame_idx for d in self.detections]
        return (min(idxs), max(idxs))

    def displacement_px_per_frame(self) -> tuple[float, float] | None:
        """平均每帧位移（像素）。

        注意：这是"像素位移/帧"，不是速度。
        逐帧真实时间戳尚未确认，禁止用 帧号差/180 换算成秒。
        速度必须等到时间基准确认后才能计算。
        """
        if len(self.detections) < 2:
            return None
        first, last = self.detections[0], self.detections[-1]
        dn = last.frame_idx - first.frame_idx
        if dn == 0:
            return None
        return ((last.cx - first.cx) / dn, (last.cy - first.cy) / dn)


@dataclass
class BatchPair:
    """一个已配对的双路批次，是并行处理的最小单元。

    paths_a / paths_b 已按帧号排好序，且长度相同、帧号一一对应。
    帧号只用于排序和双路配对，不用于计算时间。
    """

    name: str                           # 批次名，例如 Capture_20260727_232420
    paths_a: list[str]                  # 相机 a 的帧文件路径
    paths_b: list[str]                  # 相机 b 的帧文件路径
    frame_ids: list[int]                # 与上面两个列表等长的帧号
    note: str = ""                      # 配对过程中的异常说明（缺帧等）

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)


@dataclass
class BatchResult:
    """一个批次处理完的结果。

    注意这里不含任何图像数组——批次结果要跨线程传递、要写盘，
    塞进 3 MB 的 numpy 数组会让内存和队列都爆掉。
    """

    batch_name: str
    detections_a: list[Detection] = field(default_factory=list)
    detections_b: list[Detection] = field(default_factory=list)
    tracks_a: list[Track] = field(default_factory=list)
    tracks_b: list[Track] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)  # 各阶段耗时
    note: str = ""
    error: str | None = None            # 处理失败时填原因，不抛异常打断整池

    @property
    def ok(self) -> bool:
        return self.error is None
