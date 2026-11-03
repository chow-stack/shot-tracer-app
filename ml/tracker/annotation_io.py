"""Shared I/O and validation for dense golf-ball video annotations."""

from __future__ import annotations

import hashlib
import json
import os

from tracker.video_io import atomic_write_json, probe_video_timestamps


SCHEMA_VERSION = 1
VISIBILITY_VALUES = {
    "unlabeled",
    "visible",
    "blurred",
    "occluded",
    "not_visible",
    "out_of_frame",
}
SHOT_STATE_VALUES = {"unknown", "address", "flight", "landed", "finished"}
POINT_VISIBILITIES = {"visible", "blurred"}


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def create_annotation_document(
    video_path: str,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    timestamps_seconds: list[float] | None = None,
) -> dict:
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError("Video metadata must contain positive dimensions, fps, and frame count")

    absolute_video_path = os.path.abspath(video_path)
    if timestamps_seconds is not None and len(timestamps_seconds) != frame_count:
        raise ValueError(
            f"Expected {frame_count} timestamps but received {len(timestamps_seconds)}"
        )
    frames = [
        {
            "frame_index": index,
            "timestamp_seconds": (
                float(timestamps_seconds[index])
                if timestamps_seconds is not None
                else index / fps
            ),
            "x": None,
            "y": None,
            "visibility": "unlabeled",
            "shot_state": "unknown",
            "uncertain": False,
            "blur": None,
        }
        for index in range(frame_count)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "video": {
            "path": os.path.basename(absolute_video_path),
            "sha256": sha256_file(absolute_video_path),
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "frame_count": int(frame_count),
        },
        "frames": frames,
    }


def validate_annotation_document(document: dict) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    video = document.get("video")
    frames = document.get("frames")
    if not isinstance(video, dict):
        return errors + ["video must be an object"]
    if not isinstance(frames, list):
        return errors + ["frames must be an array"]

    for key in ("path", "sha256", "width", "height", "fps", "frame_count"):
        if key not in video:
            errors.append(f"video.{key} is required")

    frame_count = video.get("frame_count")
    width = video.get("width")
    height = video.get("height")
    if not isinstance(video.get("path"), str) or not video.get("path"):
        errors.append("video.path must be a non-empty string")
    sha256 = video.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        errors.append("video.sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(width, int) or width <= 0:
        errors.append("video.width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        errors.append("video.height must be a positive integer")
    if not isinstance(frame_count, int) or frame_count <= 0:
        errors.append("video.frame_count must be a positive integer")
    fps = video.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        errors.append("video.fps must be positive")
    if isinstance(frame_count, int) and len(frames) != frame_count:
        errors.append(f"frames has {len(frames)} records but video.frame_count is {frame_count}")

    seen: set[int] = set()
    previous_timestamp = -1.0
    for position, frame in enumerate(frames):
        prefix = f"frames[{position}]"
        if not isinstance(frame, dict):
            errors.append(f"{prefix} must be an object")
            continue
        index = frame.get("frame_index")
        if not isinstance(index, int) or index < 0:
            errors.append(f"{prefix}.frame_index must be a non-negative integer")
        elif index in seen:
            errors.append(f"{prefix}.frame_index duplicates {index}")
        else:
            seen.add(index)
        if index != position:
            errors.append(f"{prefix}.frame_index must equal its array position")

        visibility = frame.get("visibility")
        state = frame.get("shot_state")
        if visibility not in VISIBILITY_VALUES:
            errors.append(f"{prefix}.visibility is invalid: {visibility!r}")
        if state not in SHOT_STATE_VALUES:
            errors.append(f"{prefix}.shot_state is invalid: {state!r}")
        if not isinstance(frame.get("uncertain"), bool):
            errors.append(f"{prefix}.uncertain must be boolean")

        timestamp = frame.get("timestamp_seconds")
        if not isinstance(timestamp, (int, float)) or timestamp < 0:
            errors.append(f"{prefix}.timestamp_seconds must be non-negative")
        elif timestamp < previous_timestamp:
            errors.append(f"{prefix}.timestamp_seconds must be monotonic")
        else:
            previous_timestamp = float(timestamp)

        x, y = frame.get("x"), frame.get("y")
        if visibility in POINT_VISIBILITIES:
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                errors.append(f"{prefix} requires x/y for {visibility}")
            elif (
                isinstance(width, int)
                and isinstance(height, int)
                and not (0 <= x < width and 0 <= y < height)
            ):
                errors.append(f"{prefix} point ({x}, {y}) is outside the video")
        elif x is not None or y is not None:
            errors.append(f"{prefix} must not contain x/y for {visibility}")

        blur = frame.get("blur")
        if blur is not None:
            if not isinstance(blur, dict):
                errors.append(f"{prefix}.blur must be an object or null")
            elif not isinstance(blur.get("orientation_degrees"), (int, float)) or not isinstance(
                blur.get("length_pixels"), (int, float)
            ) or blur.get("length_pixels", -1) < 0:
                errors.append(f"{prefix}.blur requires numeric orientation and non-negative length")

    return errors


def load_annotation_document(path: str | os.PathLike[str]) -> dict:
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    errors = validate_annotation_document(document)
    if errors:
        raise ValueError("Invalid annotation document:\n- " + "\n- ".join(errors[:20]))
    return document
