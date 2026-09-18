"""Unicode 安全的图像读写。

## 为什么需要这个模块

OpenCV 在 Windows 上的 `cv2.imread` / `cv2.imwrite` 走的是 ASCII 文件名接口，
遇到 `D:\\毕设\\数据\\...` 这种非 ASCII 路径会失败，而且**失败方式非常隐蔽**：

    cv2.imread(中文路径)   -> 返回 None，只往 stderr 打一行 WARN，不抛异常
    cv2.imwrite(中文路径)  -> 返回 False，什么都不打印，文件根本没生成

本项目的数据路径 `D:\\毕设\\数据` 和输出路径 `D:\\Codex\\毕设相关` 都含中文，
两个坑都实测踩到过：imwrite 那次，脚本照常打印了"保存到 ..."，
但目录是空的。这类静默失败如果混进批处理，会让整批结果无声丢失。

## 解决办法

绕开 OpenCV 的文件名处理，改成：
    读：Python 读字节（正确处理 Unicode 路径）-> cv2.imdecode 从内存解码
    写：cv2.imencode 编码到内存 -> Python 写字节

性能与原生接口一致（都是同一套编解码器，只是 I/O 换成 Python 做）。

## 规则

项目里**不允许直接调用 cv2.imread / cv2.imwrite**，一律用这里的函数。
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """读图像，支持非 ASCII 路径。

    参数：
        path  : 图像路径，可含中文
        flags : 同 cv2.imread 的 flags，默认按灰度读

    返回：
        numpy 数组

    失败时**抛异常**而不是返回 None —— 让问题当场暴露，
    而不是把 None 传到下游变成 'NoneType has no attribute shape'。
    """
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"文件为空或不存在: {path}")

    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"图像解码失败（文件可能损坏或格式不支持）: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray, params: list[int] | None = None) -> None:
    """写图像，支持非 ASCII 路径。

    参数：
        path   : 输出路径，可含中文。扩展名决定编码格式（.png / .tif / .jpg）
        img    : 要保存的数组
        params : 同 cv2.imwrite 的 params，例如 [cv2.IMWRITE_PNG_COMPRESSION, 3]

    失败时抛异常。父目录不存在会自动创建。
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    ext = os.path.splitext(path)[1]
    if not ext:
        raise ValueError(f"输出路径缺少扩展名，无法确定编码格式: {path}")

    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"图像编码失败（检查扩展名 {ext} 和数组 dtype {img.dtype}）: {path}")

    buf.tofile(path)

    # 确认真的落盘了。这一步看似多余，但正是上面那个静默失败的教训：
    # 不验证就不能声称"已保存"。
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"写入后文件不存在或为空: {path}")
