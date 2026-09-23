"""阶段5/6数据契约：单条轨迹的线性测速结果 + 统计模块结果。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VelocityFitResult:
    """LinearVelocityEstimator.estimate() 的返回值：一条轨迹的测速结果。

    z(t) = z0 + v*t 里的 z 取轨迹点的纵坐标 cy（像素，向下为正）——
    雨滴下落是竖直方向的主运动，这是物理上有意义的"速度"；水平方向的
    拟合结果放在 vx_px_per_s，只作参考（比如判断轨迹是否明显斜飞，
    不是主测量对象）。

    velocity_px_per_s 为 None 时，表示这条轨迹根本没有拟合出速度
    （点数不足），status 和 reason 说明原因；不是"速度为0"。
    """

    track_id: int
    camera: str
    n_points: int
    duration_s: float | None
    velocity_px_per_s: float | None       # z 方向（竖直，cy）拟合斜率
    z0_px: float | None                    # 拟合截距
    vx_px_per_s: float | None              # x 方向（水平，cx）拟合斜率，参考量
    residual_rms_px: float | None          # z 方向拟合的均方根残差（像素）
    r_squared: float | None                # z 方向拟合的 R²，2点拟合时为 None（必然完美过两点，无统计意义）
    time_quality: str                      # ok | non_monotonic | duplicate_timestamps | degenerate_time_span
    spatial_calibration_status: str        # 来自调用方配置，默认 "spatial_calibration_unavailable"
    unit: str                              # "px/s"，没有速度时是空字符串
    confidence: str                        # high | medium | low | none
    status: str                            # fitted | fitted_two_point | insufficient_points | rejected
    reason: str


# ---------------------------------------------------------------------------
# 阶段6：数量浓度、粒径分布、时间窗口统计
#
# 590.4 cm³ 是任务书明确的**双路共同观测体积**：两台正交相机各自拍摄，
# 只有双路都确认存在（双路匹配成功）的目标，才真正处于这块共同空间里。
# 单路（A 或 B 单独）检测到的目标只是"这一路相机视场里出现过的候选"，
# 不能证明它落在双路共同体积内——直接拿单路目标数去除以 590.4 cm³，
# 是把"单路视场"当成了"双路共同体积"，数值上能算出来，但物理意义是错的。
#
# 所以阶段6的统计结果拆成两条完全独立的产出线：
#   SingleCameraProxyStat   : 单路调试统计。字段名和数值算法都不变，
#                             但明确标注"非正式、不可作为最终浓度"，
#                             只用于单路调参/快速核对流水线通不通。
#   StereoCommonVolumeStat  : 双路共同体积统计。N 必须来自双路匹配成功
#                             的目标数（阶段4的 StereoMatcher），
#                             没有标定文件时这个数永远是 0，
#                             concentration_status 明确标"unavailable"，
#                             不是"算出来等于0"。
# ---------------------------------------------------------------------------


@dataclass
class QualityVerdict:
    """StatisticsQualityController.evaluate() 的返回值：一条轨迹是否够格
    进入"高可信统计"。

    is_high_confidence=False 不代表这条轨迹被丢弃——它仍然会被记进
    excluded_target_count，只是不参与 valid_target_count 和浓度计算。
    """

    is_high_confidence: bool
    reasons: list[str] = field(default_factory=list)  # 不合格时的具体原因，合格时为空列表


@dataclass
class ConcentrationResult:
    """NumberConcentrationEstimator.estimate() 的返回值。"""

    n_valid_targets: int
    volume_m3: float
    concentration_m3: float  # = n_valid_targets / volume_m3，单位 m^-3
    volume_status: str       # 来自调用方配置，默认 "nominal_unvalidated"
    unit: str                # 固定值 "m^-3"（这是量纲单位，不是可信度状态，不需要配置化）


@dataclass
class DropSizeDistributionResult:
    """DropSizeDistributionEstimator.estimate() 的返回值：像素等效直径分布。

    这不是正式的（毫米）雨滴谱——没有像素-毫米标定，bin 的边界和计数
    单位都是像素，只能叫"像素等效直径分布"。size_distribution.csv 和
    报告里必须保留这个措辞，不能简化成"雨滴谱"。
    """

    camera: str
    bin_edges_px: list[float]
    counts: list[int]
    n_total: int
    spatial_calibration_status: str  # 来自调用方配置，默认 "spatial_calibration_unavailable"
    note: str = "像素等效直径分布，非正式毫米雨滴谱（无像素-毫米标定）"


@dataclass
class SingleCameraProxyStat:
    """window_statistics.csv 里 category="single_camera_proxy" 的一行。

    这是单路（A 或 B 各自独立）的调试统计——检测/跟踪流水线通不通、
    单路目标数量级对不对，看这个就够了。但 590.4 cm³ 是双路共同体积，
    不是单路视场体积，所以这里算出来的"浓度"数值不能代表真实降雨浓度，
    is_single_camera_proxy 和 usable_for_final_concentration 这两个字段
    就是把这条警告焊死在数据结构里，不依赖看报告的人自己记住。
    """

    camera: str
    window_index: int
    window_start_s: float
    window_end_s: float
    sample_count: int              # 窗口内原始检测总数（不去重，逐帧累加）
    valid_target_count: int        # 窗口内质量合格、按track去重的目标数（首次出现时刻落在本窗口）
    excluded_target_count: int     # 窗口内质量不合格、按track去重的目标数
    size_bin_edges_px: list[float]  # 粒径分箱边界（像素），随结果一起落盘方便复核
    nominal_concentration_m3: float  # = valid_target_count / volume_m3（非正式，仅供调试参考）
    volume_status: str             # 来自调用方配置，默认 "nominal_unvalidated"
    time_status: str               # ok | degraded（窗口内有任何帧被阶段1标记时间异常）
    spatial_calibration_status: str  # 来自调用方配置，默认 "spatial_calibration_unavailable"
    quality_status: str            # empty | low_confidence | nominal
    is_single_camera_proxy: bool = True         # 恒为True：明确这是单路代理统计
    usable_for_final_concentration: bool = False  # 恒为False：不得当作最终浓度使用


# 向后兼容别名：阶段6最初的实现叫 WindowStatRow，语义上就是
# SingleCameraProxyStat（当时还没区分单路/双路）。保留别名避免
# 已经写好的调用点（如果有）需要同步改名；新代码一律用新名字。
WindowStatRow = SingleCameraProxyStat


@dataclass
class StereoCommonVolumeStat:
    """window_statistics.csv 里 category="stereo_common_volume" 的一行。

    这才是 590.4 cm³ 名义体积真正对应的统计对象：N 只能来自双路匹配
    成功（阶段4 StereoMatcher 判定 status="matched"）的目标数。

    没有标定文件时（当前状态），n_matched_pairs 恒为0，
    concentration_status 恒为 "unavailable"——不是"浓度为0"，是"这个量
    现在没法算"，两者含义完全不同，绝不能把 unavailable 情况下的0
    误读成"确认无雨"。
    """

    window_index: int
    window_start_s: float
    window_end_s: float
    n_matched_pairs: int            # 双路匹配成功的目标数（已经是"共同空间"意义上的去重计数）
    valid_target_count: int         # 质量门限通过的 matched pairs 数；calibration不可用时恒为0
    concentration_m3: float | None  # concentration_status="computed" 时才有数值，否则 None
    concentration_status: str       # "unavailable" | "computed"
    volume_status: str              # 来自调用方配置，默认 "nominal_unvalidated"
    stereo_match_status: str        # "calibration_unavailable" | "ok"
    spatial_calibration_status: str  # 来自调用方配置，默认 "spatial_calibration_unavailable"
    quality_status: str             # "unavailable" | "empty" | "nominal"
