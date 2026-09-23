"""小工具：计算文件内容的 SHA-256 哈希。

用途有两个，都在阶段1里：
    1. 核对 A、B 两路目录下"内容理应相同"的 capture_metadata.json /
       capture_timing.csv 是不是真的逐字节相同，而不是只抽样比较几个字段。
    2. 核对 A、B 两路目录下同名的 .tif 图像文件，会不会出现"内容完全相同"
       的异常情况（正常情况下两台相机拍到的画面不可能逐字节相同，
       如果相同，几乎可以肯定是采集或拷贝流程出了 bug）。

单独抽成一个模块，是因为这两处用的是同一个基础操作，不应该抄两遍。
"""

from __future__ import annotations

import hashlib  # 标准库，提供 sha256 等哈希算法


def sha256_of_file(path: str) -> str:
    """返回文件内容的 sha256 十六进制摘要（64 个字符）。

    Python 语法说明：
    - `hashlib.sha256()` 造一个"哈希计算器"对象，可以反复调用 `.update(数据块)`
      往里喂数据，喂完之后 `.hexdigest()` 取出结果。
    - 之所以分块读（`while True: ... f.read(chunk_size)`）而不是一次性
      `f.read()` 整个文件，是因为图像文件有 1.3 MB，分块读可以控制内存
      占用峰值——哈希算法本来就是"边喂边算"的，不需要把整个文件都摆在内存里。
    """
    hasher = hashlib.sha256()
    chunk_size = 1024 * 1024  # 每次读 1 MB

    with open(path, "rb") as f:  # "rb" = 以二进制模式读，不做任何编码转换
        while True:
            chunk = f.read(chunk_size)
            if not chunk:  # 读到空 bytes 说明文件读完了
                break
            hasher.update(chunk)

    return hasher.hexdigest()
