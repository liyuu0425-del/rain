"""低开销双路轨迹匹配：复用阶段3轨迹和实测共同空间标定。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..imageio import imread_unicode
from ..tracking.hungarian_solver import HungarianSolver
from ..tracking.track_associator import _connected_components


@dataclass(frozen=True)
class CommonSpaceCalibration:
    mask_a: np.ndarray
    mask_b: np.ndarray
    tilt_u: float
    tilt_v: float
    tilt_intercept: float
    rmse_px: float
    max_error_px: float
    source_path: str


@dataclass(frozen=True)
class TrackObservation:
    pair_index: int
    cx: float
    cy: float
    area_px: float
    d_eq_px: float


@dataclass
class CameraTrack:
    track_id: int
    camera: str
    observations: dict[int, TrackObservation] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherSettings:
    min_track_points: int = 2
    min_common_frames: int = 2
    min_consecutive_frames: int = 2
    disqualifying_qc_mask: int = 127
    min_downward_px_per_frame: float = 0.5
    min_motion_delta_px: float = 4.0
    min_net_displacement_px: float = 8.0
    min_height_gate_px: float = 12.0
    min_height_fluctuation_gate_px: float = 6.0
    max_normalized_motion_error: float = 0.42
    max_net_motion_error: float = 0.38
    diameter_log_scale: float = 0.95
    area_log_scale: float = 1.8
    max_total_cost: float = 0.98
    min_assignment_margin: float = 0.045
    cost_weights: dict = field(default_factory=lambda: {
        "height_offset": 0.20,
        "height_fluctuation": 0.15,
        "motion": 0.30,
        "net_motion": 0.15,
        "diameter": 0.12,
        "area": 0.08,
    })


@dataclass(frozen=True)
class Candidate:
    track_a_id: int
    track_b_id: int
    a_index: int
    b_index: int
    common_frames: tuple[int, ...]
    median_height_residual_px: float
    height_mad_px: float
    normalized_motion_error: float
    net_motion_error: float
    diameter_log_error: float
    area_log_error: float
    total_cost: float


def load_common_space_calibration(
    report_path: str | Path,
    mask_a_path: str | Path,
    mask_b_path: str | Path,
) -> CommonSpaceCalibration:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tilt = report["tilt_correction"]
    expected_wh = report["camera_A_mask"]["image_size_wh"]
    mask_a = imread_unicode(str(mask_a_path))
    mask_b = imread_unicode(str(mask_b_path))
    expected_shape = (int(expected_wh[1]), int(expected_wh[0]))
    if mask_a.shape != expected_shape or mask_b.shape != expected_shape:
        raise ValueError(
            f"共同视场掩膜尺寸必须为 {expected_shape}，"
            f"实际 A={mask_a.shape}, B={mask_b.shape}"
        )
    values = [
        float(tilt["coefficient_u"]),
        float(tilt["coefficient_v"]),
        float(tilt["intercept_px"]),
        float(tilt["rmse_px"]),
        float(tilt.get("maximum_abs_error_px", 0.0)),
    ]
    if not all(math.isfinite(v) for v in values) or values[3] <= 0:
        raise ValueError("共同空间倾斜修正系数或误差无效")
    return CommonSpaceCalibration(
        mask_a=mask_a,
        mask_b=mask_b,
        tilt_u=values[0],
        tilt_v=values[1],
        tilt_intercept=values[2],
        rmse_px=values[3],
        max_error_px=values[4],
        source_path=str(report_path),
    )


def _inside_mask(x: float, y: float, mask: np.ndarray) -> bool:
    ix, iy = int(round(x)), int(round(y))
    return 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix] > 0


def read_tracks_csv(
    tracks_path: str | Path,
    calibration: CommonSpaceCalibration,
    settings: MatcherSettings,
) -> tuple[dict[str, list[CameraTrack]], dict[str, int]]:
    tracks: dict[str, dict[int, CameraTrack]] = {"a": {}, "b": {}}
    counts = {
        "rows_read": 0,
        "predicted_only_skipped": 0,
        "qc_skipped": 0,
        "outside_common_mask_skipped": 0,
        "real_points_kept": 0,
    }
    with Path(tracks_path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "track_id", "camera_id", "pair_index", "detected_cx_px", "detected_cy_px",
            "detected_area_px", "detected_d_eq_px", "quality_flags",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"tracks.csv 缺少必要列: {sorted(required - set(reader.fieldnames or []))}")

        for row in reader:
            counts["rows_read"] += 1
            camera = row["camera_id"].strip().lower()
            if camera not in tracks:
                raise ValueError(f"tracks.csv camera_id 必须是 a/b，收到 {camera!r}")
            if not row["detected_cx_px"] or not row["detected_cy_px"]:
                counts["predicted_only_skipped"] += 1
                continue

            flags = int(row["quality_flags"] or 0)
            if (flags & 0xFF) & settings.disqualifying_qc_mask:
                counts["qc_skipped"] += 1
                continue

            x, y = float(row["detected_cx_px"]), float(row["detected_cy_px"])
            mask = calibration.mask_a if camera == "a" else calibration.mask_b
            if not _inside_mask(x, y, mask):
                counts["outside_common_mask_skipped"] += 1
                continue

            tid, pidx = int(row["track_id"]), int(row["pair_index"])
            track = tracks[camera].setdefault(tid, CameraTrack(track_id=tid, camera=camera))
            if pidx in track.observations:
                raise ValueError(f"轨迹 {camera}:{tid} 在 pair_index={pidx} 有重复记录")
            track.observations[pidx] = TrackObservation(
                pair_index=pidx,
                cx=x,
                cy=y,
                area_px=float(row["detected_area_px"]),
                d_eq_px=float(row["detected_d_eq_px"]),
            )
            counts["real_points_kept"] += 1

    output: dict[str, list[CameraTrack]] = {"a": [], "b": []}
    tracks_before_min_length = sum(len(camera_tracks) for camera_tracks in tracks.values())
    for camera in ("a", "b"):
        output[camera] = sorted(
            (track for track in tracks[camera].values()
             if len(track.observations) >= settings.min_track_points),
            key=lambda track: track.track_id,
        )
    counts["tracks_a_after_filter"] = len(output["a"])
    counts["tracks_b_after_filter"] = len(output["b"])
    counts["tracks_too_short_skipped"] = tracks_before_min_length - len(output["a"]) - len(output["b"])
    return output, counts


class CommonSpaceTrackMatcher:
    """以共视场Z坐标和跨帧运动为主证据，做轨迹级一对一匹配。"""

    def __init__(self, calibration: CommonSpaceCalibration, settings: MatcherSettings):
        self.calibration = calibration
        self.settings = settings
        self.height_gate_px = max(
            settings.min_height_gate_px,
            3.0 * calibration.rmse_px,
            calibration.max_error_px,
        )
        self.height_fluctuation_gate_px = max(
            settings.min_height_fluctuation_gate_px,
            calibration.rmse_px,
        )
        self.solver = HungarianSolver()

    def match(self, tracks_a: list[CameraTrack], tracks_b: list[CameraTrack]):
        b_by_frame: dict[int, list[int]] = {}
        for j, track in enumerate(tracks_b):
            for frame in track.observations:
                b_by_frame.setdefault(frame, []).append(j)

        candidates: dict[tuple[int, int], Candidate] = {}
        pair_rows: list[dict] = []
        rejection_counts: dict[str, int] = {}
        evaluated_pairs = 0

        for i, track_a in enumerate(tracks_a):
            possible_b: set[int] = set()
            for frame in track_a.observations:
                possible_b.update(b_by_frame.get(frame, ()))
            for j in sorted(possible_b):
                evaluated_pairs += 1
                track_b = tracks_b[j]
                candidate, reason, detail = self._evaluate(track_a, track_b, i, j)
                if candidate is None:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    pair_rows.append({
                        "track_a_id": track_a.track_id,
                        "track_b_id": track_b.track_id,
                        **detail,
                        "total_cost": None,
                        "status": "rejected",
                        "method": "none",
                        "confidence": "none",
                        "reason": reason,
                    })
                    continue
                candidates[(i, j)] = candidate
                pair_rows.append(self._candidate_row(candidate, "candidate", "awaiting_assignment"))

        assigned: list[Candidate] = []
        assignment_rejections: dict[tuple[int, int], str] = {}
        assignment_methods: dict[tuple[int, int], str] = {}
        n_conflict_components = 0
        if candidates:
            matrix = np.full((len(tracks_a), len(tracks_b)), np.inf, dtype=np.float64)
            candidate_edges = np.zeros((len(tracks_a), len(tracks_b)), dtype=bool)
            for (i, j), candidate in candidates.items():
                matrix[i, j] = candidate.total_cost
                candidate_edges[i, j] = True
            components = _connected_components(candidate_edges)
            for a_idxs, b_idxs in components:
                if len(a_idxs) > 1 or len(b_idxs) > 1:
                    n_conflict_components += 1
                sub_cost = matrix[np.ix_(a_idxs, b_idxs)]
                if len(a_idxs) == 1 and len(b_idxs) == 1:
                    component_assignments = [(0, 0)]
                    method = "nearest_neighbor"
                else:
                    component_assignments = self.solver.solve(sub_cost)
                    method = "hungarian"
                for local_i, local_j in component_assignments:
                    i, j = a_idxs[local_i], b_idxs[local_j]
                    candidate = candidates.get((i, j))
                    if candidate is None:
                        continue
                    assignment_methods[(i, j)] = method
                    if candidate.total_cost >= self.settings.max_total_cost:
                        assignment_rejections[(i, j)] = "total_cost_exceeded"
                        continue
                    row_costs = sorted(c.total_cost for (ri, _), c in candidates.items() if ri == i)
                    col_costs = sorted(c.total_cost for (_, cj), c in candidates.items() if cj == j)
                    row_margin = self._margin(row_costs, candidate.total_cost)
                    col_margin = self._margin(col_costs, candidate.total_cost)
                    if min(row_margin, col_margin) < self.settings.min_assignment_margin:
                        assignment_rejections[(i, j)] = "ambiguous_assignment"
                        continue
                    assigned.append(candidate)

        assigned_keys = {(c.a_index, c.b_index) for c in assigned}
        for row in pair_rows:
            if row["status"] != "candidate":
                continue
            key = (row["_a_index"], row["_b_index"])
            row.pop("_a_index", None)
            row.pop("_b_index", None)
            if key in assigned_keys:
                row.update(
                    status="matched", method=assignment_methods[key],
                    confidence="low", reason="ok",
                )
            else:
                reason = assignment_rejections.get(key, "lost_in_conflict_resolution")
                row.update(status="unmatched_candidate", method="none", confidence="none", reason=reason)
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        matched_a = {c.track_a_id for c in assigned}
        matched_b = {c.track_b_id for c in assigned}
        candidate_a = {c.track_a_id for c in candidates.values()}
        candidate_b = {c.track_b_id for c in candidates.values()}
        for track in tracks_a:
            if track.track_id not in matched_a:
                reason = "lost_in_conflict_resolution" if track.track_id in candidate_a else "no_candidate"
                pair_rows.append(self._unmatched_row("a", track.track_id, reason))
        for track in tracks_b:
            if track.track_id not in matched_b:
                reason = "lost_in_conflict_resolution" if track.track_id in candidate_b else "no_candidate"
                pair_rows.append(self._unmatched_row("b", track.track_id, reason))

        diagnostics = {
            "n_track_pairs_evaluated": evaluated_pairs,
            "n_track_candidates": len(candidates),
            "n_matched_track_pairs": len(assigned),
            "n_conflict_components": n_conflict_components,
            "rejection_counts": rejection_counts,
        }
        return assigned, pair_rows, diagnostics

    def _evaluate(self, track_a, track_b, i, j):
        common = sorted(set(track_a.observations).intersection(track_b.observations))
        detail = {"n_common_frames": len(common), "first_common_frame": "", "last_common_frame": ""}
        if len(common) < self.settings.min_common_frames:
            return None, "temporal_overlap_too_short", detail
        detail.update(first_common_frame=common[0], last_common_frame=common[-1])
        longest = self._longest_run(common)
        if longest < self.settings.min_consecutive_frames:
            return None, "no_consecutive_overlap", detail

        a_obs = [track_a.observations[f] for f in common]
        b_obs = [track_b.observations[f] for f in common]
        z_a = np.asarray([p.cy for p in a_obs], dtype=np.float64)
        z_b = np.asarray([
            self.calibration.tilt_u * p.cx
            + self.calibration.tilt_v * p.cy
            + self.calibration.tilt_intercept
            for p in b_obs
        ], dtype=np.float64)
        residuals = z_b - z_a
        median_residual = float(np.median(residuals))
        height_mad = float(np.median(np.abs(residuals - median_residual)))
        detail.update(
            median_height_residual_px=median_residual,
            height_mad_px=height_mad,
        )
        if abs(median_residual) > self.height_gate_px:
            return None, "height_offset_exceeded", detail
        if height_mad > self.height_fluctuation_gate_px:
            return None, "height_fluctuation_exceeded", detail

        da, db = [], []
        for k in range(1, len(common)):
            if common[k] != common[k - 1] + 1:
                continue
            delta_a = z_a[k] - z_a[k - 1]
            delta_b = z_b[k] - z_b[k - 1]
            da.append(float(delta_a))
            db.append(float(delta_b))
        if not da:
            return None, "no_consecutive_motion", detail
        if (float(np.median(da)) < self.settings.min_downward_px_per_frame
                or float(np.median(db)) < self.settings.min_downward_px_per_frame):
            return None, "not_downward_motion", detail
        if (abs(float(np.median(da))) < self.settings.min_motion_delta_px
                or abs(float(np.median(db))) < self.settings.min_motion_delta_px):
            return None, "motion_too_small", detail

        motion_error = float(np.median([
            abs(x - y) / max(abs(x), abs(y), self.settings.min_motion_delta_px)
            for x, y in zip(da, db)
        ]))
        net_a = float(z_a[-1] - z_a[0])
        net_b = float(z_b[-1] - z_b[0])
        net_error = abs(net_a - net_b) / max(abs(net_a), abs(net_b), self.settings.min_net_displacement_px)
        detail.update(normalized_motion_error=motion_error, net_motion_error=net_error)
        if motion_error > self.settings.max_normalized_motion_error:
            return None, "motion_inconsistent", detail
        if net_error > self.settings.max_net_motion_error:
            return None, "net_motion_inconsistent", detail

        diameter_log_error = float(np.median([
            abs(math.log(max(pb.d_eq_px, 1.0e-6) / max(pa.d_eq_px, 1.0e-6)))
            for pa, pb in zip(a_obs, b_obs)
        ]))
        area_log_error = float(np.median([
            abs(math.log(max(pb.area_px, 1.0e-6) / max(pa.area_px, 1.0e-6)))
            for pa, pb in zip(a_obs, b_obs)
        ]))
        z_center_term = min(abs(median_residual) / max(self.height_gate_px, 1.0), 2.0)
        z_mad_term = min(height_mad / max(self.height_fluctuation_gate_px, 1.0), 2.0)
        motion_term = min(motion_error / max(self.settings.max_normalized_motion_error, 1.0e-6), 2.0)
        net_term = min(net_error / max(self.settings.max_net_motion_error, 1.0e-6), 2.0)
        diameter_term = min(diameter_log_error / max(self.settings.diameter_log_scale, 1.0e-6), 2.0)
        area_term = min(area_log_error / max(self.settings.area_log_scale, 1.0e-6), 2.0)
        weights = self.settings.cost_weights
        weight_sum = sum(float(weights.get(key, 0.0)) for key in (
            "height_offset", "height_fluctuation", "motion",
            "net_motion", "diameter", "area",
        ))
        if weight_sum <= 0:
            raise ValueError("cost_weights 权重和必须大于0")
        total = sum(
            float(weights.get(key, 0.0)) * term
            for key, term in (
                ("height_offset", z_center_term),
                ("height_fluctuation", z_mad_term),
                ("motion", motion_term),
                ("net_motion", net_term),
                ("diameter", diameter_term),
                ("area", area_term),
            )
        ) / weight_sum
        return Candidate(
            track_a_id=track_a.track_id,
            track_b_id=track_b.track_id,
            a_index=i,
            b_index=j,
            common_frames=tuple(common),
            median_height_residual_px=median_residual,
            height_mad_px=height_mad,
            normalized_motion_error=motion_error,
            net_motion_error=net_error,
            diameter_log_error=diameter_log_error,
            area_log_error=area_log_error,
            total_cost=float(total),
        ), "candidate", detail

    def _candidate_row(self, candidate: Candidate, status: str, reason: str) -> dict:
        return {
            "track_a_id": candidate.track_a_id,
            "track_b_id": candidate.track_b_id,
            "n_common_frames": len(candidate.common_frames),
            "first_common_frame": candidate.common_frames[0],
            "last_common_frame": candidate.common_frames[-1],
            "median_height_residual_px": candidate.median_height_residual_px,
            "height_mad_px": candidate.height_mad_px,
            "normalized_motion_error": candidate.normalized_motion_error,
            "net_motion_error": candidate.net_motion_error,
            "diameter_log_error": candidate.diameter_log_error,
            "area_log_error": candidate.area_log_error,
            "total_cost": candidate.total_cost,
            "status": status,
            "method": "none",
            "confidence": "none",
            "reason": reason,
            "_a_index": candidate.a_index,
            "_b_index": candidate.b_index,
        }

    @staticmethod
    def _unmatched_row(camera: str, track_id: int, reason: str) -> dict:
        return {
            "track_a_id": track_id if camera == "a" else "",
            "track_b_id": track_id if camera == "b" else "",
            "n_common_frames": 0,
            "first_common_frame": "",
            "last_common_frame": "",
            "median_height_residual_px": "",
            "height_mad_px": "",
            "normalized_motion_error": "",
            "net_motion_error": "",
            "diameter_log_error": "",
            "area_log_error": "",
            "total_cost": "",
            "status": f"unmatched_{camera}",
            "method": "none",
            "confidence": "none",
            "reason": reason,
        }

    @staticmethod
    def _margin(values: list[float], chosen: float) -> float:
        alternatives = []
        removed_chosen = False
        for value in values:
            if not removed_chosen and abs(value - chosen) <= 1.0e-9:
                removed_chosen = True
            else:
                alternatives.append(value)
        return min(alternatives) - chosen if alternatives else math.inf

    @staticmethod
    def _longest_run(frames: list[int]) -> int:
        longest = current = 1
        for previous, current_frame in zip(frames[:-1], frames[1:]):
            current = current + 1 if current_frame == previous + 1 else 1
            longest = max(longest, current)
        return longest

def settings_from_config(raw: dict) -> MatcherSettings:
    allowed = MatcherSettings.__dataclass_fields__
    return MatcherSettings(**{key: value for key, value in raw.items() if key in allowed})
