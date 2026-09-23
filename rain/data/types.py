"""阶段1数据契约：采集元数据、逐帧时间记录、帧配对结果、时间质量诊断。

和 rain/types.py 是同一类东西——规定"阶段1的几个模块之间传什么"。
单独放一份在这里，是为了不去改动 rain/types.py（那份契约服务于已有的
检测/跟踪 Demo，阶段1的新模块不应该牵动它，见任务要求"不得修改原Demo"）。

Python 语法说明（这个项目里第一次出现的写法，后面同类代码不再重复注释）：

- `from __future__ import annotations`：
  让类型注解里可以直接写 `int | None`、`list[str]` 这种"新式"写法，
  不用管当前 Python 版本是否原生支持。习惯写在文件最开头第一行代码。

- `@dataclass`：
  一个"装饰器"（写在 class 上面一行、以 @ 开头）。加了它，Python 会
  自动生成 `__init__`（构造函数，即"造一个对象要填哪些参数"）和
  `__repr__`（打印时显示的样子），不用自己手写
  `def __init__(self, x, y): self.x = x; self.y = y`。

- 类型注解 `name: str`、`value: float | None = None`：
  冒号后面是"这个字段应该是什么类型"，等号后面是默认值。Python 运行时
  不会强制检查类型对不对，这只是给人和编辑器看的提示，但能帮我们尽早
  发现"传错了类型"这种低级错误，也是自文档化的一种方式。

- `field(default_factory=list)`：
  dataclass 的字段如果默认值是列表/字典这种"可变对象"，不能直接写
  `= []`（那样所有实例会共享同一个列表对象，是 Python 的一个经典陷阱）。
  要用 `field(default_factory=list)`，让每个新对象各自新建一个空列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CaptureMetadata:
    """一次采集的元数据（来自 capture_metadata.json）。

    这份 JSON 由采集程序统一生成，A、B 两路目录下各存一份、内容应当一致
    （因为它同时记录了双路的信息）。是否一致由 MetadataReader 的调用方核对。
    """

    capture_name: str
    camera_a_sn: str
    camera_b_sn: str
    requested_frame_pairs: int
    requested_width: int
    requested_height: int
    requested_exposure_us: float
    requested_analog_gain: int
    a_trigger_mode: str
    b_trigger_mode: str
    captured_complete_frame_pairs: int
    capture_complete: bool
    missing_frame_pairs: int
    valid_frame_pairs: int
    median_frame_period_a_us: float
    median_frame_period_b_us: float
    max_frame_period_a_us: float
    max_frame_period_b_us: float
    frame_gap_threshold_a_us: float
    frame_gap_threshold_b_us: float
    median_relative_drift_us: float
    p95_abs_relative_drift_us: float
    max_abs_relative_drift_us: float
    timing_model: str
    alignment_note: str
    raw: dict  # 原始 JSON 全文，未在上面列出的字段也能在审计里查到


@dataclass
class TimingRecord:
    """capture_timing.csv 里的一行：一个已配对图像对的关键时间信息。

    只挑选"测速和时间质量检查"真正需要用到的字段。capture_timing.csv
    原表里还有很多别的列（host 时间、增益、字节数……），不在这里维护
    一份大而全的结构，需要时直接读原表。
    """

    pair_index: int
    pair_filename: str
    a_device_relative_us: float
    b_device_relative_us: float
    b_minus_a_relative_drift_us: float
    a_exposure_us: float
    b_exposure_us: float
    a_analog_gain: float
    b_analog_gain: float
    a_host_receive_wall_ns: int
    b_host_receive_wall_ns: int
    a_is_trigger_frame: int
    b_is_trigger_frame: int


@dataclass
class FramePairRecord:
    """一个 pair_index 在 A/B 两路目录下的配对结果（frame_pairing.csv 一行）。"""

    pair_index: int
    filename: str
    path_a: str | None
    path_b: str | None
    status: str  # ok | missing_a | missing_b | missing_both | extra_a | extra_b
    note: str = ""


@dataclass
class PairTimingQC:
    """单个图像对的时间质量诊断（timing_quality.csv 一行）。"""

    pair_index: int
    a_device_relative_us: float
    b_device_relative_us: float
    interval_a_us: float | None  # 与上一个 pair_index 的 a 设备时间差
    interval_b_us: float | None
    drift_us: float  # b_minus_a_relative_drift_us，原样带过来方便复核
    is_time_backwards_a: bool
    is_time_backwards_b: bool
    is_duplicate_time_a: bool
    is_duplicate_time_b: bool
    is_interval_outlier_a: bool
    is_interval_outlier_b: bool
    is_drift_outlier: bool

    @property
    def flags_text(self) -> str:
        """把上面几个布尔标记拼成一行可读文本，写进 CSV 方便人工复核。

        `@property` 是另一个装饰器：加了它，`qc.flags_text` 可以像访问
        字段一样访问（不用写成 `qc.flags_text()` 这种带括号的方法调用），
        但它的值是每次访问时临时算出来的，不是存好的字段。
        """
        names = []
        if self.is_time_backwards_a:
            names.append("time_backwards_a")
        if self.is_time_backwards_b:
            names.append("time_backwards_b")
        if self.is_duplicate_time_a:
            names.append("duplicate_time_a")
        if self.is_duplicate_time_b:
            names.append("duplicate_time_b")
        if self.is_interval_outlier_a:
            names.append("interval_outlier_a")
        if self.is_interval_outlier_b:
            names.append("interval_outlier_b")
        if self.is_drift_outlier:
            names.append("drift_outlier")
        return "|".join(names) if names else "ok"


@dataclass
class TimingQualityReport:
    """整批时间质量检查的汇总结果。"""

    n_records: int
    measured_frame_period_median_us: float
    measured_frame_period_p95_us: float
    measured_frame_period_max_us: float
    n_time_backwards: int
    n_duplicate_time: int
    n_interval_outliers: int
    n_drift_outliers: int
    missing_pair_indices: list[int] = field(default_factory=list)
    per_pair: list[PairTimingQC] = field(default_factory=list)

    @property
    def n_missing_pair_index(self) -> int:
        return len(self.missing_pair_indices)

    @property
    def n_flagged_total(self) -> int:
        return sum(1 for p in self.per_pair if p.flags_text != "ok")


@dataclass
class ImageFileReport:
    """单张 TIF 文件的解码/格式检查结果（阶段1最终验收新增）。"""

    camera: str  # "a" | "b"
    filename: str
    path: str
    ok: bool
    error: str | None
    width: int | None
    height: int | None
    dtype: str | None  # 例如 "uint8"
    n_channels: int | None
    file_size_bytes: int
    sha256: str


@dataclass
class ImageIntegritySummary:
    """整批 TIF 完整性扫描的汇总结果。"""

    n_total: int
    n_ok: int
    n_corrupt_or_empty: int
    resolution_counts: dict[str, int] = field(default_factory=dict)
    dtype_channel_counts: dict[str, int] = field(default_factory=dict)
    identical_ab_filenames: list[str] = field(default_factory=list)
    per_file: list[ImageFileReport] = field(default_factory=list)

    @property
    def n_identical_ab_pairs(self) -> int:
        return len(self.identical_ab_filenames)
