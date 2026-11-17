import csv
from pathlib import Path

import numpy as np

from pipeline.stage4_stereo_matching import _write_track_match_csv
from rain.stereo.common_space_track_matcher import (
    CameraTrack,
    CommonSpaceCalibration,
    CommonSpaceTrackMatcher,
    MatcherSettings,
    TrackObservation,
)


def test_match_csv_uses_timestamped_fallback_when_default_is_locked(tmp_path, monkeypatch, capsys):
    target = tmp_path / "stereo_matches.csv"
    target.write_text("existing locked file", encoding="utf-8")
    original_open = Path.open

    def open_with_locked_target(path, *args, **kwargs):
        if path == target and args and args[0] == "w":
            raise PermissionError("file is locked")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_locked_target)

    result_path = _write_track_match_csv(
        [{"track_a_id": 12, "track_b_id": 34, "status": "matched"}],
        target,
    )

    assert result_path != target
    assert result_path.exists()
    assert target.read_text(encoding="utf-8") == "existing locked file"
    with result_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["track_a_id"] == "12"
    assert "另存为" in capsys.readouterr().out


def _calibration():
    return CommonSpaceCalibration(
        mask_a=np.full((300, 300), 255, dtype=np.uint8),
        mask_b=np.full((300, 300), 255, dtype=np.uint8),
        tilt_u=-0.023339870947248074,
        tilt_v=0.9970250892026022,
        tilt_intercept=21.362934633680343,
        rmse_px=2.0,
        max_error_px=3.0,
        source_path="test-calibration",
    )


def _track(camera, track_id, heights, diameter=10.0, x=100.0):
    track = CameraTrack(track_id=track_id, camera=camera)
    for frame, height in enumerate(heights):
        if camera == "a":
            y = height
        else:
            y = (height - _calibration().tilt_intercept
                 - _calibration().tilt_u * x) / _calibration().tilt_v
        area = np.pi * (diameter / 2.0) ** 2
        track.observations[frame] = TrackObservation(
            pair_index=frame,
            cx=x,
            cy=y,
            area_px=area,
            d_eq_px=diameter,
        )
    return track


def test_common_space_track_match_uses_height_and_downward_motion():
    matcher = CommonSpaceTrackMatcher(_calibration(), MatcherSettings())
    matched, rows, result = matcher.match(
        [_track("a", 1, [100, 120, 140])],
        [_track("b", 7, [100, 120, 140])],
    )
    assert [(item.track_a_id, item.track_b_id) for item in matched] == [(1, 7)]
    assert result["n_matched_track_pairs"] == 1
    assert next(row for row in rows if row["status"] == "matched")["method"] == "nearest_neighbor"


def test_common_space_track_rejects_large_height_offset():
    matcher = CommonSpaceTrackMatcher(_calibration(), MatcherSettings())
    matched, rows, result = matcher.match(
        [_track("a", 1, [100, 120, 140])],
        [_track("b", 7, [200, 220, 240])],
    )
    assert not matched
    assert result["n_track_candidates"] == 0
    assert any(row["reason"] == "height_offset_exceeded" for row in rows)


def test_global_assignment_matches_each_track_at_most_once():
    matcher = CommonSpaceTrackMatcher(_calibration(), MatcherSettings())
    matched, rows, result = matcher.match(
        [
            _track("a", 1, [100, 120, 140], diameter=10),
            _track("a", 2, [101, 121, 141], diameter=20),
        ],
        [_track("b", 7, [100, 120, 140], diameter=10)],
    )
    assert len(matched) == 1
    assert matched[0].track_a_id == 1
    assert matched[0].track_b_id == 7
    assert result["n_conflict_components"] == 1
    assert next(row for row in rows if row["status"] == "matched")["method"] == "hungarian"
