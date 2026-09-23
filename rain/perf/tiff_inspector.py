"""TIFF 标签检查：直接解析 IFD，读出压缩方式等元信息，不依赖任何图像库。

阶段7B性能优化的第一步要求"先检查当前TIF的压缩方式...确认是否确实为
LZW解码瓶颈"——不能凭文件名或代码里的变量名（"lzw_tif"）就假设，
必须真正去读文件本身的 TIFF 标签。这个模块只解析 TIFF 头和 IFD 里
"我们关心的几个标签"，不是完整的 TIFF 解析器（不处理多页、不处理
BigTIFF），够用即可。

Python 语法说明：
- `struct.unpack(fmt, bytes)`：把一段原始字节按指定格式解析成数字。
  比如 `struct.unpack('<H', b'\\x05\\x00')` 按"小端序、无符号短整数"
  解析出 5。TIFF 文件头会先声明自己是"II"(小端, Intel)还是"MM"
  (大端, Motorola)，后续所有数字都要按这个字节序去读。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# TIFF 规范里的标准标签号（只列出这里用得到的几个）
_TAG_IMAGE_WIDTH = 256
_TAG_IMAGE_LENGTH = 257
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_ROWS_PER_STRIP = 278
_TAG_PREDICTOR = 317

_COMPRESSION_NAMES = {
    1: "none (uncompressed)",
    2: "CCITT_RLE",
    3: "CCITT_G3",
    4: "CCITT_G4",
    5: "LZW",
    6: "JPEG (old-style)",
    7: "JPEG",
    8: "AdobeDeflate",
    32773: "PackBits",
    32946: "Deflate",
}

_PREDICTOR_NAMES = {1: "none", 2: "horizontal_differencing", 3: "floating_point"}


@dataclass
class TiffInfo:
    """从一个 TIFF 文件的 IFD 里读出的关键信息。"""

    width: int | None
    height: int | None
    bits_per_sample: int | None
    compression_id: int | None
    compression_name: str
    predictor_id: int | None
    predictor_name: str
    rows_per_strip: int | None
    file_size_bytes: int
    raw_uncompressed_bytes: int | None  # width*height*bits_per_sample/8，用于算压缩比
    compression_ratio: float | None


def inspect_tiff(path: str) -> TiffInfo:
    """读取一个 TIFF 文件的关键标签。文件损坏或不是 TIFF 时抛异常
    （审计工具不该悄悄返回一个"看起来正常"的空结果）。
    """
    import os

    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(f"{path}: 文件太小，不是有效的 TIFF")

        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise ValueError(f"{path}: 字节序标记异常 {header[:2]!r}，不是标准 TIFF")

        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic != 42:
            raise ValueError(f"{path}: TIFF magic number 应为42，实际是{magic}")

        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        f.seek(ifd_offset)
        n_entries = struct.unpack(endian + "H", f.read(2))[0]

        tags: dict[int, int] = {}
        for _ in range(n_entries):
            entry = f.read(12)
            tag, field_type, count = struct.unpack(endian + "HHI", entry[:8])
            value_bytes = entry[8:12]
            # 只处理 SHORT(3) 和 LONG(4) 且 count==1 的情况——这几个标签
            # 在正常 TIFF 里都是这种形式，够用；其它类型不在本工具关心范围内。
            if field_type == 3 and count == 1:
                value = struct.unpack(endian + "H", value_bytes[:2])[0]
            elif field_type == 4 and count == 1:
                value = struct.unpack(endian + "I", value_bytes)[0]
            else:
                value = None
            if value is not None:
                tags[tag] = value

    width = tags.get(_TAG_IMAGE_WIDTH)
    height = tags.get(_TAG_IMAGE_LENGTH)
    bits = tags.get(_TAG_BITS_PER_SAMPLE)
    comp_id = tags.get(_TAG_COMPRESSION)
    pred_id = tags.get(_TAG_PREDICTOR)
    file_size = os.path.getsize(path)

    raw_bytes = None
    ratio = None
    if width and height and bits:
        raw_bytes = width * height * bits // 8
        if raw_bytes > 0:
            ratio = file_size / raw_bytes

    return TiffInfo(
        width=width, height=height, bits_per_sample=bits,
        compression_id=comp_id, compression_name=_COMPRESSION_NAMES.get(comp_id, f"unknown({comp_id})"),
        predictor_id=pred_id, predictor_name=_PREDICTOR_NAMES.get(pred_id, f"unknown({pred_id})"),
        rows_per_strip=tags.get(_TAG_ROWS_PER_STRIP),
        file_size_bytes=file_size, raw_uncompressed_bytes=raw_bytes, compression_ratio=ratio,
    )
