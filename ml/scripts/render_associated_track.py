"""Render a clip-level associated golf-ball trajectory for visual QA.

Observed portions of the path are cyan. Bounded gap estimates are rendered as
short amber dashes. Inferred continuation is rendered as magenta dashes so the
video never presents interpolation or forecasting as a detection.
The renderer reads the source sequentially; this is important for variable-rate
iPhone MOV files, where OpenCV's random frame seeking can return the wrong
decoded frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


OBSERVED_COLOR = (255, 230, 35)  # BGR: cyan
ESTIMATED_COLOR = (0, 185, 255)  # BGR: amber
INFERRED_COLOR = (210, 80, 255)  # BGR: magenta


def load_trajectory(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    points = sorted(document.get("trajectory", []), key=lambda point: point["frame_index"])
    if not points:
        raise ValueError(f"No trajectory points in {path}")
    return points


def provenance(point: dict) -> str:
    value = point.get("provenance")
    if value in {"observed", "estimated_gap", "inferred"}:
        return value
    return "estimated_gap" if point.get("estimated") else "observed"


def draw_segment(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int],
                 color: tuple[int, int, int], width: int, dashed: bool) -> None:
    if not dashed:
        cv2.line(canvas, start, end, color, width, cv2.LINE_AA)
        return

    delta = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        cv2.circle(canvas, start, max(1, width // 2), color, -1, cv2.LINE_AA)
        return
    direction = delta / length
    dash_length = max(8.0, width * 2.5)
    gap_length = max(5.0, width * 1.5)
    distance = 0.0
    while distance < length:
        dash_end = min(length, distance + dash_length)
        p1 = np.asarray(start, dtype=np.float32) + direction * distance
        p2 = np.asarray(start, dtype=np.float32) + direction * dash_end
        cv2.line(canvas, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)),
                 color, width, cv2.LINE_AA)
        distance += dash_length + gap_length


def draw_trace(frame: np.ndarray, points: list[dict], current_frame: int) -> np.ndarray:
    visible = [point for point in points if point["frame_index"] <= current_frame]
    if not visible:
        return frame

    scale = min(frame.shape[0], frame.shape[1]) / 1080.0
    inner_width = max(3, round(5 * scale))
    glow_width = max(inner_width + 4, round(13 * scale))
    glow = np.zeros_like(frame)
    line = np.zeros_like(frame)

    for previous, point in zip(visible, visible[1:]):
        start = (round(previous["x"]), round(previous["y"]))
        end = (round(point["x"]), round(point["y"]))
        kinds = {provenance(previous), provenance(point)}
        inferred = "inferred" in kinds
        estimated = "estimated_gap" in kinds
        color = INFERRED_COLOR if inferred else ESTIMATED_COLOR if estimated else OBSERVED_COLOR
        if inferred:
            opacity = min(float(previous.get("opacity", 1.0)), float(point.get("opacity", 1.0)))
            color = tuple(round(channel * opacity) for channel in color)
        dashed = inferred or estimated
        draw_segment(glow, start, end, color, glow_width, dashed)
        draw_segment(line, start, end, color, inner_width, dashed)

    head = visible[-1]
    head_point = (round(head["x"]), round(head["y"]))
    head_kind = provenance(head)
    head_color = (
        INFERRED_COLOR
        if head_kind == "inferred"
        else ESTIMATED_COLOR
        if head_kind == "estimated_gap"
        else OBSERVED_COLOR
    )
    if head_kind == "inferred":
        opacity = float(head.get("opacity", 1.0))
        head_color = tuple(round(channel * opacity) for channel in head_color)
    cv2.circle(glow, head_point, max(7, round(11 * scale)), head_color, -1, cv2.LINE_AA)
    cv2.circle(line, head_point, max(3, round(5 * scale)), (255, 255, 255), -1, cv2.LINE_AA)

    rendered = cv2.addWeighted(frame, 1.0, glow, 0.22, 0.0)
    rendered = cv2.addWeighted(rendered, 1.0, line, 0.92, 0.0)
    return rendered


def render(video_path: Path, trajectory_path: Path, output_path: Path) -> int:
    points = load_trajectory(trajectory_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {output_path}")

    frame_index = 0
    progress = tqdm(total=expected_frames or None, unit="frame", desc=video_path.stem)
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            writer.write(draw_trace(frame, points, frame_index))
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frame_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame_count = render(args.video, args.trajectory, args.output)
    print(f"Rendered {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
