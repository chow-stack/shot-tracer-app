"""Versioned trajectory documents with explicit point provenance."""

from __future__ import annotations

import copy
import math
from typing import Any


PROVENANCE_VALUES = {"observed", "estimated_gap", "inferred"}


def _timestamp_for_frame(document: dict, frame_index: int) -> float:
    video = document.get("video", {})
    timestamps = video.get("timestamps_seconds")
    if isinstance(timestamps, list) and frame_index < len(timestamps):
        return float(timestamps[frame_index])
    fps = float(video.get("fps", 0.0))
    if fps <= 0:
        raise ValueError("Trajectory needs exact timestamps or a positive video fps")
    return frame_index / fps


def point_provenance(point: dict) -> str:
    provenance = point.get("provenance")
    if provenance in PROVENANCE_VALUES:
        return str(provenance)
    return "estimated_gap" if point.get("estimated") else "observed"


def _validate_point(point: dict, position: int) -> None:
    frame_index = point.get("frame_index")
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(f"trajectory[{position}].frame_index is invalid")
    for key in ("timestamp_seconds", "x", "y"):
        value = point.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"trajectory[{position}].{key} must be finite")
    if point_provenance(point) not in PROVENANCE_VALUES:
        raise ValueError(f"trajectory[{position}].provenance is invalid")


def build_segments(points: list[dict]) -> list[dict]:
    """Group adjacent points of the same provenance for schema v2 exports."""
    segments: list[dict] = []
    for point in points:
        kind = point_provenance(point)
        if not segments or segments[-1]["kind"] != kind:
            segments.append({
                "kind": kind,
                "start_frame": point["frame_index"],
                "end_frame": point["frame_index"],
                "points": [copy.deepcopy(point)],
            })
        else:
            segments[-1]["end_frame"] = point["frame_index"]
            segments[-1]["points"].append(copy.deepcopy(point))
    return segments


def upgrade_trajectory_document(document: dict) -> dict:
    """Return a validated v2 copy while retaining the legacy flat trajectory."""
    upgraded = copy.deepcopy(document)
    values = upgraded.get("trajectory")
    if not isinstance(values, list) or not values:
        raise ValueError("Trajectory document must contain a non-empty trajectory array")

    points: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for position, original in enumerate(sorted(values, key=lambda item: item.get("frame_index", -1))):
        if not isinstance(original, dict):
            raise ValueError(f"trajectory[{position}] must be an object")
        point = dict(original)
        frame_index = point.get("frame_index")
        if not isinstance(frame_index, int) or frame_index in seen_frames:
            raise ValueError("Trajectory frame indexes must be unique integers")
        seen_frames.add(frame_index)
        point["timestamp_seconds"] = float(
            point.get("timestamp_seconds", _timestamp_for_frame(upgraded, frame_index))
        )
        point["provenance"] = point_provenance(point)
        point["estimated"] = point["provenance"] != "observed"
        point.setdefault("uncertainty_px", None)
        point.setdefault("source", "association" if point["provenance"] == "observed" else "short_gap_interpolation")
        _validate_point(point, position)
        points.append(point)

    observed = [point for point in points if point["provenance"] == "observed"]
    if not observed:
        raise ValueError("Trajectory document contains no observed points")
    upgraded["trajectory_schema_version"] = 2
    upgraded.setdefault("continuation_mode", "observed_only")
    upgraded["observation_end_frame"] = observed[-1]["frame_index"]
    upgraded["trajectory"] = points
    upgraded["observations"] = observed
    upgraded["segments"] = build_segments(points)
    return upgraded


def observed_points(document: dict) -> list[dict]:
    upgraded = upgrade_trajectory_document(document)
    return [point for point in upgraded["trajectory"] if point["provenance"] == "observed"]
