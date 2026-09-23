# Python 学习指南

写给不熟悉 Python 的读者。目标不是讲完整门语言，而是讲清楚"这个项目里
反复出现的那几种写法，各自在做什么、为什么这么写"。每个阶段一小节，
配一个能独立运行的小例子——不依赖真实数据，复制粘贴到一个 `.py` 文件
就能跑，跑起来之后再回头对照项目里的正式代码，会容易理解得多。

## 怎么用这份指南

- 先看"通用语法速查"，建立一个整体印象；不需要一次全记住。
- 每个阶段的示例都是独立脚本，`python 文件名.py` 直接跑。
- 项目正式代码里，"这个写法在项目里第一次出现"的地方会有对应注释，
  指南和代码注释是互相印证的关系，不是重复的关系。

---

## 通用语法速查

| 语法 | 是什么 | 为什么项目要用它 |
|---|---|---|
| `import xxx` | 把别的文件/库里定义的东西拿到当前文件里用 | 把代码拆成小文件，各管各的 |
| `def f(x, y=1): ... return z` | 定义一个函数：输入 x、y（y 有默认值），算出 z 返回 | 把重复逻辑包起来，起个名字 |
| `class Foo:` / `__init__` / `self` | 定义一类对象的模板；`__init__` 是"造对象时执行一次"的初始化方法；`self` 是"对象自己" | 把数据和操作数据的方法绑在一起，方便替换实现 |
| `@dataclass` | 自动给 class 生成 `__init__`、打印格式 | 省掉大量样板代码，还统一了字段命名 |
| 类型注解 `x: int`、`x: int \| None` | 给人和编辑器看的"这应该是什么类型" | 提前发现传错类型的低级错误 |
| `if / elif / else` | 条件分支 | 按不同情况走不同逻辑 |
| `for x in 列表:` | 遍历一个序列 | 逐帧、逐条记录地处理数据 |
| `列表 list` `[1, 2, 3]` / `字典 dict` `{"a": 1}` | 两种最常用的容器 | 列表存"一串东西"，字典存"名字对应的值" |
| `try / except` | 先假设能成功，出错了再处理 | 应对"字段可能不存在"之类的情况 |
| `with open(...) as f:` | 打开资源，用完自动释放 | 保证文件一定被正确关闭，哪怕中途出错 |
| NumPy 数组切片/布尔索引 | `arr[a:b]`、`arr[arr > 0]` | 对整批数据做数学运算，不用手写 for 循环 |
| OpenCV 函数 | 输入 `numpy` 数组，输出也是 `numpy` 数组 | 图像处理（阶段2开始大量出现） |
| 配置文件读取（YAML/JSON） | `yaml.safe_load(...)` / `json.loads(...)` | 参数改配置文件，不改代码 |
| `if __name__ == "__main__":` | "只有直接运行这个文件才执行" | 脚本被 import 时不会意外触发整套流程 |

---

## 阶段1 示例：读元数据、算帧周期、判断时间是否正常

这个例子只用标准库（不依赖项目代码），演示阶段1四个模块背后的核心逻辑：
读 JSON、读 CSV、算相邻时间差、用 dataclass 组织结果。复制到一个新文件
（比如 `try_stage1.py`）就能跑。

```python
"""阶段1最小示例：不依赖 rain 包，独立说明每个语法点在做什么。"""

from __future__ import annotations  # 允许下面写 `int | None` 这种类型注解

import json
from dataclasses import dataclass


# --- class / @dataclass -----------------------------------------------
# @dataclass 是一个"装饰器"：写在 class 上面一行，Python 会自动帮这个类生成
# __init__ 构造函数。不加它就要自己写：
#     def __init__(self, pair_index, a_us, b_us):
#         self.pair_index = pair_index
#         self.a_us = a_us
#         self.b_us = b_us
@dataclass
class TimingRow:
    pair_index: int   # 冒号后面是"类型注解"，只是提示，不强制检查
    a_us: float
    b_us: float


def parse_rows(raw_rows: list[dict]) -> list[TimingRow]:
    """把"一串字典"转换成"一串 TimingRow 对象"。

    参数说明（def 的写法）：
        raw_rows : 参数名和类型注解。函数体里可以直接用这个名字。
    返回值：
        -> list[TimingRow] 表示这个函数会 return 一个"元素是 TimingRow 的列表"。
    """
    rows: list[TimingRow] = []          # 空列表，之后用 append 往里加
    for raw in raw_rows:                # for 循环：依次取出每个字典
        rows.append(
            TimingRow(
                pair_index=int(raw["pair_index"]),
                a_us=float(raw["a_us"]),
                b_us=float(raw["b_us"]),
            )
        )
    return rows


def find_anomalies(rows: list[TimingRow], gap_threshold_us: float) -> dict:
    """演示"相邻帧时间差"和"异常判定"的核心逻辑（阶段1 TimeQualityChecker 的简化版）。"""
    backwards = []     # 时间倒退的 pair_index
    outliers = []       # 间隔异常的 pair_index

    prev = None
    for row in rows:
        if prev is not None:
            interval = row.a_us - prev.a_us    # 和上一帧的时间差
            if interval < 0:
                backwards.append(row.pair_index)
            if abs(interval) > gap_threshold_us:
                outliers.append(row.pair_index)
        prev = row   # 更新"上一条"，供下一轮循环使用

    # 字典 {} 用来打包"多个有名字的结果"一起返回
    return {"time_backwards": backwards, "interval_outliers": outliers}


if __name__ == "__main__":
    # 用 json.loads 模拟"读 capture_metadata.json"：真实项目里是
    # `json.loads(Path(path).read_text(encoding="utf-8"))`，这里直接写死字符串
    # 省去创建文件的步骤，方便你复制粘贴就能跑。
    fake_metadata_json = '{"median_frame_period_a_us": 3200.0, "frame_gap_threshold_a_us": 4800.0}'
    metadata = json.loads(fake_metadata_json)
    print("元数据里的中位帧周期:", metadata["median_frame_period_a_us"], "us")

    # 模拟 capture_timing.csv 的几行（真实项目里用 csv.DictReader 读文件）
    fake_rows = [
        {"pair_index": 0, "a_us": 0},
        {"pair_index": 1, "a_us": 3200},
        {"pair_index": 2, "a_us": 6400},
        {"pair_index": 3, "a_us": 6300},   # 故意比上一帧还早，制造一次"时间倒退"
        {"pair_index": 4, "a_us": 60000},  # 故意跳变一大截，制造一次"间隔异常"
    ]
    # 上面每行缺了 b_us，这里统一补上（示例只关心 a_us 的异常判定）
    for r in fake_rows:
        r["b_us"] = r["a_us"]

    rows = parse_rows(fake_rows)
    result = find_anomalies(rows, gap_threshold_us=metadata["frame_gap_threshold_a_us"])

    print("时间倒退的 pair_index:", result["time_backwards"])
    print("间隔异常的 pair_index:", result["interval_outliers"])
```

运行结果应该是：

```
元数据里的中位帧周期: 3200.0 us
时间倒退的 pair_index: [3]
间隔异常的 pair_index: [4]
```

对照项目正式代码：
- `TimingRow` 对应 `rain/data/types.py` 里的 `TimingRecord`；
- `parse_rows` 对应 `rain/data/timing_reader.py` 的 `TimingReader.read`；
- `find_anomalies` 对应 `rain/data/time_quality_checker.py` 的
  `TimeQualityChecker.check`（正式版还多算了重复时间、双路漂移、
  pair_index 缺号，并且阈值优先从 metadata 取而不是硬编码）。

---

## 后续阶段

阶段2（检测拆分）开始会用到 NumPy 数组切片/布尔索引、OpenCV 函数，
届时在这里补充对应小节，与代码里的注释保持同步。

---

## 阶段2 示例：背景差分 + 阈值分割 + 找轮廓

这个例子演示阶段2六步流水线里最核心的三步（背景差分→阈值分割→连通域/
轮廓提取），用到 NumPy 数组运算和 OpenCV 函数。复制到 `try_stage2.py`
就能跑，不需要真实图像。

```python
"""阶段2最小示例：背景差分 + 固定阈值 + 找轮廓，纯 NumPy/OpenCV 演示。"""

import cv2
import numpy as np


# --- 造一张"背光板 + 一个暗色雨滴"的合成图 -------------------------------
H, W = 100, 150
background = np.full((H, W), 200, dtype=np.uint8)   # np.full: 造一个数组，每个元素都填同一个值
frame = background.copy()                            # .copy()：造一份独立拷贝，改 frame 不会连带改 background
cv2.circle(frame, center=(60, 50), radius=8, color=120, thickness=-1)  # 画一个暗圆模拟雨滴


# --- NumPy 切片：取图像的一个矩形区域 -------------------------------------
# img[y0:y1, x0:x1] —— 注意 NumPy 数组是 (行, 列) = (y, x) 的顺序，
# 和我们平时说的"横坐标x, 纵坐标y"是反的，这是图像处理里最容易踩的坑。
roi = frame[40:60, 50:70]          # 取雨滴所在的一小块区域
print("ROI 平均灰度:", roi.mean())  # 应该比背景的200暗一些，因为圆圈把这块弄暗了


# --- NumPy 布尔索引：找出所有"比背景暗15个灰阶以上"的像素 -----------------
diff = cv2.subtract(background, frame)   # 背景 - 帧：雨滴处变亮(正值)，其余接近0
is_drop_pixel = diff > 15                # 比较运算作用在整个数组上，得到一个"全是True/False"的同尺寸数组
print("疑似雨滴像素数:", is_drop_pixel.sum())  # True 当 1、False 当 0，求和就是像素数量
print("这些像素的差值范围:", diff[is_drop_pixel].min(), "~", diff[is_drop_pixel].max())
# diff[is_drop_pixel] 就是"布尔索引"：只取出 is_drop_pixel 为 True 位置对应的那些 diff 值，
# 结果是一个一维数组，长度等于 True 的个数。


# --- OpenCV 函数：阈值分割 + 找轮廓 ----------------------------------------
# cv2.threshold 输入一张灰度图和一个阈值，输出 (实际用的阈值, 二值图)。
# 二值图里像素只有 0 或 255，>阈值的地方是 255。
_, binary = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

# cv2.findContours 输入二值图，输出 (轮廓列表, 层级信息)。
# 每个轮廓是一个 Nx1x2 的整数数组，N 是这个轮廓有多少个点。
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("找到轮廓数:", len(contours))

for cnt in contours:
    area = cv2.contourArea(cnt)          # 轮廓包围的面积（像素）
    x, y, w, h = cv2.boundingRect(cnt)   # 轮廓的包围框：(左上角x, 左上角y, 宽, 高)
    print(f"轮廓面积={area:.0f}px  包围框=({x},{y},{w},{h})")
```

运行结果大致是（实际数值取决于圆的抗锯齿边缘，会有小幅浮动）：

```
ROI 平均灰度: 160.6
疑似雨滴像素数: 197
这些像素的差值范围: 80 ~ 80
找到轮廓数: 1
轮廓面积=174px  包围框=(52,42,17,17)
```

对照项目正式代码：
- `roi = frame[40:60, 50:70]` 这种切片写法在 `rain/detection/` 里不直接出现
  （检测器目前处理整幅图，没有切 ROI），但阶段2以后如果要限定检测区域，
  就是这个写法。
- `is_drop_pixel = diff > 15` 对应 `rain/detection/adaptive_threshold_segmenter.py`
  里 `cv2.threshold`/`cv2.adaptiveThreshold` 做的事情——只是那里用 OpenCV
  的阈值函数一次性对整幅图做，而不是手写布尔索引，两者效果等价，
  OpenCV 版本更快（底层是C++实现）。
- `cv2.subtract` 对应 `rain/detection/background_differencer.py`；
  `cv2.findContours` 对应 `rain/detection/connected_component_analyzer.py`；
  `cv2.contourArea`/`cv2.boundingRect` 对应
  `rain/detection/contour_extractor.py` 里算 `area_px`/`bbox` 的那几行。
