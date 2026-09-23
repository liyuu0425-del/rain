"""ContourExtractor 实现：轮廓特征提取 + 尺寸/形状/触边 QC
（阶段2拆分自 simple_threshold.py 第5步）。

逐个轮廓算面积、包围框、质心、等效直径、周长、拟合椭圆长短轴，
同时判定触边/太小/太大/形状异常这几个 QC 标记——这部分逻辑在原 Demo
里和"找轮廓"是同一个循环，这里原样保留计算逻辑，只是把输入从
"自己调用 cv2.findContours"改成"接收上一步算好的轮廓列表"。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..registry import register_contour_extractor
from ..types import QC_ODD_SHAPE, QC_TOO_LARGE, QC_TOUCHES_BORDER, Detection


@register_contour_extractor("contour_extractor")
class ContourExtractor:
    """轮廓 -> Detection，含面积/形状/触边 QC。

    参数（与 SimpleThresholdDetector 同名同义，方便逐帧比较）：
        min_area   : 最小面积（像素），小于此值的轮廓直接丢弃、不计入返回列表
        max_area   : 最大面积（像素），超过打 QC_TOO_LARGE 标记但仍返回
        max_aspect : 长宽比上限，超过打 QC_ODD_SHAPE 标记但仍返回
    """

    def __init__(
        self,
        min_area: float = 10.0,
        max_area: float = 5000.0,
        max_aspect: float = 3.5,
    ):
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.max_aspect = float(max_aspect)

    def extract(
        self,
        contours: list[np.ndarray],
        img_shape: tuple[int, int],
        frame_idx: int,
    ) -> list[Detection]:
        H, W = img_shape[0], img_shape[1]
        detections: list[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            qc_flags = 0
            if area < self.min_area:
                # 太小的直接跳过，不记录——和原 Demo 行为一致：
                # 这类轮廓大概率是背景噪点，不是候选雨滴。
                continue
            if area > self.max_area:
                qc_flags |= QC_TOO_LARGE

            x, y, w, h = cv2.boundingRect(cnt)
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                qc_flags |= QC_TOUCHES_BORDER

            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect:
                qc_flags |= QC_ODD_SHAPE

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w / 2, y + h / 2

            d_eq = Detection.d_eq_from_area(area)
            perimeter = cv2.arcLength(cnt, closed=True)

            major, minor = None, None
            if len(cnt) >= 5:
                try:
                    (ex, ey), (ma, mi), angle = cv2.fitEllipse(cnt)
                    major, minor = max(ma, mi), min(ma, mi)
                except cv2.error:
                    pass  # 退化情况下拟合失败，保持 None

            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    cx=cx,
                    cy=cy,
                    area_px=area,
                    bbox=(x, y, w, h),
                    d_eq_px=d_eq,
                    qc_flags=qc_flags,
                    contour=cnt,
                    perimeter_px=perimeter,
                    major_axis_px=major,
                    minor_axis_px=minor,
                )
            )

        return detections
