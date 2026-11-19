"""阶段5 CSV入口：验证真实观测、设备时间和轨迹级配对的完整数据流。"""

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.stage5_velocity import _join_stereo_matches, _read_tracks_csv, main
from rain.measurement.linear_velocity_estimator import LinearVelocityEstimator


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _point(cam, tid, frame, t, y, method="gated_nn", flags=0):
    return {
        "track_id": tid, "camera_id": cam, "pair_index": frame,
        "device_time_s": t, "detected_cx_px": 10, "detected_cy_px": y,
        "detected_area_px": 50, "detected_d_eq_px": 8,
        "association_method": method, "quality_flags": flags,
    }


def _match(a=1, b=1, status="matched"):
    return {
        "track_a_id": a, "track_b_id": b, "n_common_frames": 3,
        "status": status, "confidence": "low", "reason": "fixed_delay_uncalibrated",
    }


def test_csv_pipeline_fits_device_time_and_joins_only_accepted_pairs(tmp_path, monkeypatch, capsys):
    tracks = tmp_path / "tracks.csv"
    rows = [_point(cam, 1, i * 2, t, 100 + v * t)
            for cam, v in (("a", 3000), ("b", 2500))
            for i, t in enumerate((0.0, 0.013, 0.04))]
    # 即使预测行带有坐标，也不能污染拟合。
    rows.append(_point("a", 1, 1, 0.01, 99999, "predicted_only", 256))
    rows.append(_point("a", 2, 5, 0.05, 500, "new_track", 512))
    _write_csv(tracks, rows)
    matches = tmp_path / "stereo_matches.csv"
    _write_csv(matches, [_match(), _match(999, 999, "rejected"), _match("", "", "unmatched_a")])
    output = tmp_path / "out"
    config = tmp_path / "pipeline.yaml"
    # 无图像目录、reader、detection或tracking配置也必须能完成阶段5。
    config.write_text(yaml.safe_dump({"measurement": {
        "tracks_csv": str(tracks), "stereo_matches_csv": str(matches),
        "spatial_calibration_path": str(Path(__file__).parent.parent / "config" / "practical_common_space.json"),
        "output_dir": str(output), "n_plot_samples": 0,
        "linear_velocity_estimator": {"name": "linear_velocity_estimator", "params": {}},
    }}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["stage5_velocity.py", "--config", str(config)])

    assert main() == 0
    terminal = capsys.readouterr().out
    assert "符合测速条件的轨迹速度" not in terminal
    assert "track_id=" not in terminal
    assert "阶段5步骤耗时" in terminal
    assert "阶段5总体耗时" in terminal
    with (output / "velocities.csv").open(encoding="utf-8-sig", newline="") as stream:
        velocities = list(csv.DictReader(stream))
    assert len(velocities) == 3
    by_key = {(r["camera_id"], r["track_id"]): r for r in velocities}
    for cam, expected, scale in (
        ("a", 3000, 0.03578243706605337),
        ("b", 2500, 0.03571829135234379),
    ):
        result = by_key[(cam, "1")]
        assert float(result["velocity_px_per_s"]) == pytest.approx(expected)
        assert float(result["velocity_mm_per_s"]) == pytest.approx(expected * scale)
        assert float(result["velocity_m_per_s"]) == pytest.approx(expected * scale / 1000)
        assert result["n_points"] == "3"
        assert result["confidence"] == "medium"
        assert result["spatial_calibration_status"] == "spatial_calibration_applied"
    assert by_key[("a", "2")]["velocity_px_per_s"] == ""
    with (output / "matched_velocities.csv").open(encoding="utf-8-sig", newline="") as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 1
    assert paired[0]["match_confidence"] == "low"
    assert paired[0]["a_confidence"] == paired[0]["b_confidence"] == "medium"
    assert float(paired[0]["a_velocity_px_per_s"]) == pytest.approx(3000)
    assert float(paired[0]["b_velocity_px_per_s"]) == pytest.approx(2500)
    report = json.loads((output / "stage5_report.json").read_text(encoding="utf-8"))
    assert report["input"]["predicted_only_skipped"] == 1
    assert report["input"]["real_points_kept"] == 7
    assert report["stereo_association"]["n_matched_track_pairs"] == 1
    assert report["stage_timings_ms"]["images_read"] == 0


def test_csv_preserves_time_reversal_when_restoring_frame_order(tmp_path):
    path = tmp_path / "tracks.csv"
    _write_csv(path, [
        _point("a", 1, 2, 0.03, 190),
        _point("a", 1, 0, 0.02, 160),
        _point("a", 1, 1, 0.01, 130),
    ])
    records, _ = _read_tracks_csv(path)
    result = LinearVelocityEstimator().estimate(records["a"][0])
    assert result.time_quality == "non_monotonic"
    assert result.confidence == "low"


def test_duplicate_real_observation_is_rejected(tmp_path):
    path = tmp_path / "tracks.csv"
    _write_csv(path, [_point("a", 1, 0, 0, 100), _point("a", 1, 0, 0, 101)])
    with pytest.raises(ValueError, match="重复检测"):
        _read_tracks_csv(path)


def test_match_with_missing_track_is_rejected(tmp_path):
    path = tmp_path / "stereo_matches.csv"
    _write_csv(path, [_match()])
    with pytest.raises(ValueError, match="核对输入批次"):
        _join_stereo_matches(path, [])


def test_matching_requires_one_to_one_tracks(tmp_path):
    tracks = tmp_path / "tracks.csv"
    _write_csv(tracks, [_point(cam, 1, i, i * 0.01, 100 + i * 30)
                        for cam in ("a", "b") for i in range(2)])
    records, _ = _read_tracks_csv(tracks)
    fits = [LinearVelocityEstimator().estimate(r) for recs in records.values() for r in recs]
    matches = tmp_path / "stereo_matches.csv"
    _write_csv(matches, [_match(), _match()])
    with pytest.raises(ValueError, match="不是一对一"):
        _join_stereo_matches(matches, fits)


def test_detection_level_matches_are_not_accepted_as_track_pairs(tmp_path):
    path = tmp_path / "stereo_matches.csv"
    _write_csv(path, [{"pair_index": 0, "a_detection_index": 1, "status": "matched"}])
    with pytest.raises(ValueError, match="轨迹级"):
        _join_stereo_matches(path, [])
