"""Annotation-free video metadata and JSON I/O helpers."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2


def probe_video_timestamps(video_path: str | os.PathLike[str]) -> list[float]:
    """Return exact presentation timestamps for every decodable video frame."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "csv=p=0",
                os.fspath(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required to preserve exact video timestamps") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffprobe failed: {error.stderr.strip()}") from error

    timestamps: list[float] = []
    for line in result.stdout.splitlines():
        value = line.split(",", 1)[0].strip()
        if value:
            timestamps.append(float(value))
    if not timestamps:
        raise RuntimeError("ffprobe returned no video-frame timestamps")
    return timestamps


def probe_video_metadata(video_path: str | os.PathLike[str]) -> dict:
    """Probe dimensions/fps and use timestamps as the exact frame-count truth."""
    timestamps = probe_video_timestamps(video_path)
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata for: {video_path}")
    return {
        "path": os.path.abspath(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(timestamps),
        "timestamps_seconds": timestamps,
    }


def atomic_write_json(path: str | os.PathLike[str], document: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, destination)
        os.chmod(destination, 0o644)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
