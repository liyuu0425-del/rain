"""注册表与工厂。

作用：把 config.yaml 里的字符串名字，变成真正的算法对象。
这是"换算法不改主程序"的机关所在。

用法（在实现文件里）：

    @register_detector("simple_threshold")
    class SimpleThresholdDetector:
        def __init__(self, thresh=20): ...
        def detect(self, img, bg, frame_idx=0, debug=None): ...

用法（在主程序里）：

    detector = build_detector(cfg["detector"])

config.yaml 里：

    detector:
      name: simple_threshold
      params: {thresh: 20}

改 name 就换算法，主程序一个字不动。

Python 语法说明：
- 下面的 register_xxx 是"带参数的装饰器"。它本身是个函数，
  调用后返回真正的装饰器 deco，deco 再接收类对象、登记到字典、原样返回。
  所以 @register_detector("abc") 等价于：
      class Foo: ...
      Foo = register_detector("abc")(Foo)
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

# 五张注册表，每种模块一张。键是配置里写的名字，值是类对象（不是实例）。
READERS: dict[str, type] = {}
BACKGROUNDS: dict[str, type] = {}
DETECTORS: dict[str, type] = {}
TRACKERS: dict[str, type] = {}
STEREO: dict[str, type] = {}

# 阶段2新增：检测流程拆成六个子步骤后，每个子步骤也单独注册，
# 这样组合式 Detector（rain/detectors/modular_pipeline.py）才能像
# build_detector 一样，从配置字符串构造出子模块实例，而不是在
# Pipeline 代码里 new 出具体的类——那样就违反了"Pipeline只负责调用顺序"。
NORMALIZERS: dict[str, type] = {}
DIFFERENCERS: dict[str, type] = {}
SEGMENTERS: dict[str, type] = {}
MORPHOLOGIES: dict[str, type] = {}
CC_ANALYZERS: dict[str, type] = {}
CONTOUR_EXTRACTORS: dict[str, type] = {}

# 阶段3新增：跨帧跟踪拆成三个独立算法模块，同样各自注册一张表。
# HungarianSolver 单独一张表（不是 TrackAssociator 的内部细节写死），
# 是因为"候选冲突时用什么最优分配算法"本身也应该可替换。
KALMAN_PREDICTORS: dict[str, type] = {}
TRACK_ASSOCIATORS: dict[str, type] = {}
TRACK_MANAGERS: dict[str, type] = {}
HUNGARIAN_SOLVERS: dict[str, type] = {}

# 阶段4新增：双路匹配拆成五个独立算法模块。HungarianSolver 复用阶段3
# 已经注册的 HUNGARIAN_SOLVERS 表，不重新建一张——同一个算法（给定代价矩阵
# 求最优分配）在跨帧关联和双路匹配里是同一件事，没有理由实现两遍。
EPIPOLAR_GATES: dict[str, type] = {}
MATCH_GATES: dict[str, type] = {}
COST_MATRIX_BUILDERS: dict[str, type] = {}
NN_SOLVERS: dict[str, type] = {}
STEREO_MATCHERS: dict[str, type] = {}

# 阶段5新增：线性测速独立一张表。
VELOCITY_ESTIMATORS: dict[str, type] = {}

# 阶段6新增：统计质量门限、数量浓度、粒径分布、时间窗口统计各一张表。
STATS_QUALITY_CONTROLLERS: dict[str, type] = {}
CONCENTRATION_ESTIMATORS: dict[str, type] = {}
SIZE_DISTRIBUTION_ESTIMATORS: dict[str, type] = {}
WINDOW_STATISTICS: dict[str, type] = {}

# 双路共同体积统计单独一张表——和 WindowStatistics（单路代理统计）
# 输入输出形状完全不同，不能塞进同一张表互相替换。
STEREO_COMMON_VOLUME_STATISTICS: dict[str, type] = {}

_ALL_TABLES = {
    "reader": READERS,
    "background": BACKGROUNDS,
    "detector": DETECTORS,
    "tracker": TRACKERS,
    "stereo": STEREO,
    "normalizer": NORMALIZERS,
    "differencer": DIFFERENCERS,
    "segmenter": SEGMENTERS,
    "morphology": MORPHOLOGIES,
    "cc_analyzer": CC_ANALYZERS,
    "contour_extractor": CONTOUR_EXTRACTORS,
    "kalman_predictor": KALMAN_PREDICTORS,
    "track_associator": TRACK_ASSOCIATORS,
    "track_manager": TRACK_MANAGERS,
    "hungarian_solver": HUNGARIAN_SOLVERS,
    "epipolar_gate": EPIPOLAR_GATES,
    "match_gate": MATCH_GATES,
    "cost_matrix_builder": COST_MATRIX_BUILDERS,
    "nn_solver": NN_SOLVERS,
    "stereo_matcher": STEREO_MATCHERS,
    "velocity_estimator": VELOCITY_ESTIMATORS,
    "stats_quality_controller": STATS_QUALITY_CONTROLLERS,
    "concentration_estimator": CONCENTRATION_ESTIMATORS,
    "size_distribution_estimator": SIZE_DISTRIBUTION_ESTIMATORS,
    "window_statistics": WINDOW_STATISTICS,
    "stereo_common_volume_statistics": STEREO_COMMON_VOLUME_STATISTICS,
}


def _make_register(table: dict[str, type], kind: str) -> Callable:
    def register(name: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            if name in table:
                raise ValueError(
                    f"{kind} 名字重复注册: {name!r}（已有 {table[name].__name__}，"
                    f"又来了 {cls.__name__}）"
                )
            table[name] = cls
            return cls

        return deco

    return register


register_reader = _make_register(READERS, "reader")
register_background = _make_register(BACKGROUNDS, "background")
register_detector = _make_register(DETECTORS, "detector")
register_tracker = _make_register(TRACKERS, "tracker")
register_stereo = _make_register(STEREO, "stereo")
register_normalizer = _make_register(NORMALIZERS, "normalizer")
register_differencer = _make_register(DIFFERENCERS, "differencer")
register_segmenter = _make_register(SEGMENTERS, "segmenter")
register_morphology = _make_register(MORPHOLOGIES, "morphology")
register_cc_analyzer = _make_register(CC_ANALYZERS, "cc_analyzer")
register_contour_extractor = _make_register(CONTOUR_EXTRACTORS, "contour_extractor")
register_kalman_predictor = _make_register(KALMAN_PREDICTORS, "kalman_predictor")
register_track_associator = _make_register(TRACK_ASSOCIATORS, "track_associator")
register_track_manager = _make_register(TRACK_MANAGERS, "track_manager")
register_hungarian_solver = _make_register(HUNGARIAN_SOLVERS, "hungarian_solver")
register_epipolar_gate = _make_register(EPIPOLAR_GATES, "epipolar_gate")
register_match_gate = _make_register(MATCH_GATES, "match_gate")
register_cost_matrix_builder = _make_register(COST_MATRIX_BUILDERS, "cost_matrix_builder")
register_nn_solver = _make_register(NN_SOLVERS, "nn_solver")
register_stereo_matcher = _make_register(STEREO_MATCHERS, "stereo_matcher")
register_velocity_estimator = _make_register(VELOCITY_ESTIMATORS, "velocity_estimator")
register_stats_quality_controller = _make_register(STATS_QUALITY_CONTROLLERS, "stats_quality_controller")
register_concentration_estimator = _make_register(CONCENTRATION_ESTIMATORS, "concentration_estimator")
register_size_distribution_estimator = _make_register(SIZE_DISTRIBUTION_ESTIMATORS, "size_distribution_estimator")
register_window_statistics = _make_register(WINDOW_STATISTICS, "window_statistics")
register_stereo_common_volume_statistics = _make_register(
    STEREO_COMMON_VOLUME_STATISTICS, "stereo_common_volume_statistics"
)


def _build(table: dict[str, type], kind: str, cfg: dict[str, Any] | None) -> Any:
    """通用工厂：按 cfg["name"] 查表，用 cfg["params"] 构造实例。"""
    if cfg is None:
        raise ValueError(f"配置里缺少 {kind} 段")
    name = cfg.get("name")
    if not name:
        raise ValueError(f"{kind} 配置缺少 name 字段：{cfg!r}")
    if name not in table:
        available = ", ".join(sorted(table)) or "(空——实现模块没被 import)"
        raise KeyError(f"未知的 {kind}: {name!r}。当前已注册: {available}")
    params = cfg.get("params") or {}
    try:
        return table[name](**params)
    except TypeError as exc:
        raise TypeError(
            f"构造 {kind} {name!r} 失败，检查 config.yaml 里的 params：{exc}"
        ) from exc


def build_reader(cfg):
    return _build(READERS, "reader", cfg)


def build_background(cfg):
    return _build(BACKGROUNDS, "background", cfg)


def build_detector(cfg):
    return _build(DETECTORS, "detector", cfg)


def build_tracker(cfg):
    return _build(TRACKERS, "tracker", cfg)


def build_stereo(cfg):
    return _build(STEREO, "stereo", cfg)


def build_normalizer(cfg):
    return _build(NORMALIZERS, "normalizer", cfg)


def build_differencer(cfg):
    return _build(DIFFERENCERS, "differencer", cfg)


def build_segmenter(cfg):
    return _build(SEGMENTERS, "segmenter", cfg)


def build_morphology(cfg):
    return _build(MORPHOLOGIES, "morphology", cfg)


def build_cc_analyzer(cfg):
    return _build(CC_ANALYZERS, "cc_analyzer", cfg)


def build_contour_extractor(cfg):
    return _build(CONTOUR_EXTRACTORS, "contour_extractor", cfg)


def build_kalman_predictor(cfg):
    return _build(KALMAN_PREDICTORS, "kalman_predictor", cfg)


def build_track_associator(cfg):
    return _build(TRACK_ASSOCIATORS, "track_associator", cfg)


def build_track_manager(cfg):
    return _build(TRACK_MANAGERS, "track_manager", cfg)


def build_hungarian_solver(cfg):
    return _build(HUNGARIAN_SOLVERS, "hungarian_solver", cfg)


def build_epipolar_gate(cfg):
    return _build(EPIPOLAR_GATES, "epipolar_gate", cfg)


def build_match_gate(cfg):
    return _build(MATCH_GATES, "match_gate", cfg)


def build_cost_matrix_builder(cfg):
    return _build(COST_MATRIX_BUILDERS, "cost_matrix_builder", cfg)


def build_nn_solver(cfg):
    return _build(NN_SOLVERS, "nn_solver", cfg)


def build_stereo_matcher(cfg):
    return _build(STEREO_MATCHERS, "stereo_matcher", cfg)


def build_velocity_estimator(cfg):
    return _build(VELOCITY_ESTIMATORS, "velocity_estimator", cfg)


def build_stats_quality_controller(cfg):
    return _build(STATS_QUALITY_CONTROLLERS, "stats_quality_controller", cfg)


def build_concentration_estimator(cfg):
    return _build(CONCENTRATION_ESTIMATORS, "concentration_estimator", cfg)


def build_size_distribution_estimator(cfg):
    return _build(SIZE_DISTRIBUTION_ESTIMATORS, "size_distribution_estimator", cfg)


def build_window_statistics(cfg):
    return _build(WINDOW_STATISTICS, "window_statistics", cfg)


def build_stereo_common_volume_statistics(cfg):
    return _build(STEREO_COMMON_VOLUME_STATISTICS, "stereo_common_volume_statistics", cfg)


def describe() -> str:
    """列出所有已注册实现，给 bench.py 和排错用。"""
    lines = []
    for kind, table in _ALL_TABLES.items():
        names = ", ".join(sorted(table)) if table else "(无)"
        lines.append(f"  {kind:11s}: {names}")
    return "\n".join(lines)
