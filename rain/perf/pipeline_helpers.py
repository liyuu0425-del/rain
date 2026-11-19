"""Shared configuration helpers for the performance pipeline scripts."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_source_path(source_path: str, config_path: Path) -> Path:
    candidate = Path(source_path)
    if candidate.is_absolute():
        return candidate
    project_candidate = config_path.parent.parent / candidate
    if project_candidate.exists():
        return project_candidate
    return config_path.parent / candidate


def load_observation_volume(stats_cfg: dict, config_path: Path) -> dict[str, object]:
    """Load the configured observation volume using stage 6's rules."""
    source_cfg = stats_cfg.get("volume_source") or {}
    source_path_text = source_cfg.get("path")
    source_key = source_cfg.get("key", "common_box.safe_volume_cm3")
    if not source_path_text:
        params = stats_cfg["concentration_estimator"]["params"]
        volume_cm3 = float(params["volume_cm3"])
        return {
            "volume_cm3": volume_cm3,
            "volume_m3": volume_cm3 * 1e-6,
            "source_path": "pipeline.yaml::statistics.concentration_estimator.params",
            "source_key": "volume_cm3",
        }

    source_path = resolve_source_path(str(source_path_text), config_path)
    value = json.loads(source_path.read_text(encoding="utf-8"))
    for key_part in str(source_key).split("."):
        if not isinstance(value, dict) or key_part not in value:
            raise KeyError(f"观测体积键 {source_key!r} 不在 {source_path} 中。")
        value = value[key_part]

    raw_value = float(value)
    leaf_key = str(source_key).rsplit(".", 1)[-1]
    if leaf_key.endswith("_cm3"):
        volume_cm3 = raw_value
    elif leaf_key.endswith("_m3"):
        volume_cm3 = raw_value * 1e6
    elif leaf_key.endswith("_liter"):
        volume_cm3 = raw_value * 1000.0
    else:
        raise ValueError(
            f"无法判断观测体积单位：键 {source_key!r} 应以 _cm3、_m3 或 _liter 结尾。"
        )
    if volume_cm3 <= 0:
        raise ValueError(f"观测体积必须大于0，读取到 {volume_cm3} cm^3。")
    return {
        "volume_cm3": volume_cm3,
        "volume_m3": volume_cm3 * 1e-6,
        "source_path": str(source_path),
        "source_key": str(source_key),
    }


def apply_observation_volume(cfg: dict, config_path: Path) -> dict[str, object]:
    info = load_observation_volume(cfg["statistics"], config_path)
    cfg["statistics"]["concentration_estimator"]["params"].update(
        volume_cm3=info["volume_cm3"], volume_m3=info["volume_m3"]
    )
    return info


def configured_reader_params(cfg: dict) -> dict:
    return dict(cfg.get("reader", {}).get("params") or {})


def configured_frame_count(cfg: dict) -> int:
    return int(cfg["detection"]["n_frames"])
