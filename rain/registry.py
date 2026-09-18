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

_ALL_TABLES = {
    "reader": READERS,
    "background": BACKGROUNDS,
    "detector": DETECTORS,
    "tracker": TRACKERS,
    "stereo": STEREO,
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


def describe() -> str:
    """列出所有已注册实现，给 bench.py 和排错用。"""
    lines = []
    for kind, table in _ALL_TABLES.items():
        names = ", ".join(sorted(table)) if table else "(无)"
        lines.append(f"  {kind:11s}: {names}")
    return "\n".join(lines)
