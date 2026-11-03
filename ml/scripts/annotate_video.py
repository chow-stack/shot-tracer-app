"""Interactive dense frame annotator for golf-shot videos.

The tool autosaves after every edit and resumes at the first unlabeled frame.
Coordinates are stored in original video pixels even when the display is scaled.

Usage:
    ml/.venv/bin/python ml/scripts/annotate_video.py ml/videos/curve-iron.MOV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.annotation_io import (  # noqa: E402
    POINT_VISIBILITIES,
    atomic_write_json,
    create_annotation_document,
    load_annotation_document,
    probe_video_timestamps,
    sha256_file,
)


WINDOW_NAME = "Shot Tracer Annotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Video to annotate")
    parser.add_argument(
        "--output",
        help="Annotation JSON path (default: ml/annotations/<video>.annotations.json)",
    )
    parser.add_argument("--start-frame", type=int, help="Frame to open instead of first unlabeled")
    parser.add_argument("--display-width", type=int, default=1440)
    parser.add_argument("--display-height", type=int, default=900)
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=360,
        help="Original-pixel square shown after right-clicking to zoom (default: 360)",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Re-hash the video when resuming and reject a mismatched file",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create/validate the annotation file and exit without opening a window",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Re-read video metadata/timestamps while preserving existing labels",
    )
    return parser.parse_args()


def default_output_path(video_path: str) -> str:
    script_dir = Path(__file__).resolve().parent
    return str(script_dir.parent / "annotations" / f"{Path(video_path).stem}.annotations.json")


def read_video_metadata(video_path: str) -> tuple[int, int, float, int, list[float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    timestamps = probe_video_timestamps(video_path)
    frame_count = len(timestamps)
    return width, height, fps, frame_count, timestamps


def open_or_create_document(
    video_path: str, output_path: str, verify_hash: bool, refresh_metadata: bool
) -> dict:
    existing = None
    if os.path.exists(output_path):
        existing = load_annotation_document(output_path)
        document = existing
        metadata = document["video"]
        if verify_hash and sha256_file(video_path) != metadata["sha256"]:
            raise ValueError("Video SHA-256 does not match the existing annotation document")
        if not refresh_metadata:
            return document

    width, height, fps, frame_count, timestamps = read_video_metadata(video_path)
    document = create_annotation_document(
        video_path, width, height, fps, frame_count, timestamps_seconds=timestamps
    )
    if existing is not None:
        label_fields = ("x", "y", "visibility", "shot_state", "uncertain", "blur")
        for old_frame, new_frame in zip(existing["frames"], document["frames"]):
            for field in label_fields:
                new_frame[field] = old_frame[field]
    atomic_write_json(output_path, document)
    return document


def first_unlabeled_frame(document: dict) -> int:
    for frame in document["frames"]:
        if frame["visibility"] == "unlabeled":
            return frame["frame_index"]
    return 0


class VideoFrameReader:
    """Exact frame reader that never trusts keyframe-based OpenCV seeking.

    These iPhone MOV files can report a successful ``CAP_PROP_POS_FRAMES`` seek
    while returning a frame 5–6 positions later than requested. An annotation
    made on that frame is then silently saved under the wrong index. Decode
    forward sequentially, restarting only for backward navigation, so the frame
    shown always matches the annotation/evaluator index.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = self._open()
        self.frame_index = -1
        self.frame = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        return cap

    def _restart(self) -> None:
        self.cap.release()
        self.cap = self._open()
        self.frame_index = -1
        self.frame = None

    def _read_next(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {self.frame_index + 1}")
        self.frame_index += 1
        self.frame = frame
        return frame

    def get(self, requested_index: int):
        if requested_index == self.frame_index and self.frame is not None:
            return self.frame
        if requested_index < self.frame_index:
            self._restart()
        while self.frame_index < requested_index:
            self._read_next()
        return self.frame

    def release(self) -> None:
        self.cap.release()


def fit_display(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame.copy(), scale
    return cv2.resize(frame, (round(width * scale), round(height * scale))), scale


def zoom_crop(frame, center: tuple[float, float], size: int):
    """Return a clamped square crop and its origin in original coordinates."""
    height, width = frame.shape[:2]
    crop_width = min(size, width)
    crop_height = min(size, height)
    center_x, center_y = center
    x1 = max(0, min(round(center_x - crop_width / 2), width - crop_width))
    y1 = max(0, min(round(center_y - crop_height / 2), height - crop_height))
    return frame[y1 : y1 + crop_height, x1 : x1 + crop_width], x1, y1


def put_hud(display, lines: list[str]) -> None:
    line_height = 24
    panel_height = 12 + line_height * len(lines)
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (12, 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if args.zoom_size <= 0:
        raise SystemExit("--zoom-size must be positive")
    video_path = os.path.abspath(args.video)
    output_path = os.path.abspath(args.output or default_output_path(video_path))
    document = open_or_create_document(
        video_path, output_path, args.verify_hash, args.refresh_metadata
    )
    metadata = document["video"]
    frames = document["frames"]

    if args.init_only:
        labeled_count = sum(frame["visibility"] != "unlabeled" for frame in frames)
        print(
            f"Ready: {output_path} ({metadata['frame_count']} frames, "
            f"{labeled_count} labeled)"
        )
        return

    reader = VideoFrameReader(video_path)

    frame_index = args.start_frame if args.start_frame is not None else first_unlabeled_frame(document)
    frame_index = max(0, min(frame_index, metadata["frame_count"] - 1))
    active_visibility = "visible"
    active_shot_state = frames[frame_index]["shot_state"]
    auto_advance = True
    mouse_state: dict[str, object] = {
        "click": None,
        "zoom_request": None,
        "scale": 1.0,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }
    zoom_center: tuple[float, float] | None = None

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            scale = float(mouse_state["scale"])
            original_point = (
                float(mouse_state["origin_x"]) + x / scale,
                float(mouse_state["origin_y"]) + y / scale,
            )
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_state["click"] = original_point
            else:
                mouse_state["zoom_request"] = original_point

    def save() -> None:
        atomic_write_json(output_path, document)

    def advance(amount: int = 1) -> None:
        nonlocal frame_index
        frame_index = max(0, min(frame_index + amount, metadata["frame_count"] - 1))

    def label_without_point(visibility: str) -> None:
        record = frames[frame_index]
        record["x"] = None
        record["y"] = None
        record["visibility"] = visibility
        record["shot_state"] = active_shot_state
        record["blur"] = None
        save()
        if auto_advance:
            advance()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"Annotations: {output_path}")
    print("Controls are also shown in the video window. Every edit is autosaved.")

    try:
        while True:
            frame = reader.get(frame_index)
            if zoom_center is None:
                shown_frame, origin_x, origin_y = frame, 0, 0
            else:
                shown_frame, origin_x, origin_y = zoom_crop(frame, zoom_center, args.zoom_size)
            display, scale = fit_display(shown_frame, args.display_width, args.display_height)
            mouse_state["scale"] = scale
            mouse_state["origin_x"] = float(origin_x)
            mouse_state["origin_y"] = float(origin_y)
            record = frames[frame_index]

            if record["visibility"] in POINT_VISIBILITIES:
                point = (
                    round((record["x"] - origin_x) * scale),
                    round((record["y"] - origin_y) * scale),
                )
                if 0 <= point[0] < display.shape[1] and 0 <= point[1] < display.shape[0]:
                    cv2.circle(display, point, 9, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(display, point, (0, 255, 255), cv2.MARKER_CROSS, 18, 1)

            labeled_count = sum(f["visibility"] != "unlabeled" for f in frames)
            put_hud(
                display,
                [
                    f"Frame {frame_index + 1}/{metadata['frame_count']} | labeled {labeled_count} | "
                    f"saved: {record['visibility']} / {record['shot_state']}"
                    f"{' / uncertain' if record['uncertain'] else ''}",
                    f"Mode: click={active_visibility}, state={active_shot_state}, "
                    f"auto-advance={'on' if auto_advance else 'off'}",
                    f"View: {'ZOOM (z=full)' if zoom_center else 'FULL (right-click=zoom)'} | left-click=point",
                    "v visible | b blurred | o occluded | n not visible | x out of frame",
                    "1 address | 2 flight | 3 landed | 4 finished | u uncertain | c clear",
                    "j/l prev/next | [/ ] jump 10 | t auto-advance | s save | q save+quit",
                ],
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKeyEx(30)
            click = mouse_state.pop("click", None)
            mouse_state["click"] = None
            zoom_request = mouse_state.pop("zoom_request", None)
            mouse_state["zoom_request"] = None
            if zoom_request is not None:
                zoom_center = zoom_request
                continue
            if click is not None:
                x, y = click
                x = min(max(float(x), 0.0), metadata["width"] - 1.0)
                y = min(max(float(y), 0.0), metadata["height"] - 1.0)
                record["x"] = x
                record["y"] = y
                record["visibility"] = active_visibility
                record["shot_state"] = active_shot_state
                record["blur"] = None
                save()
                if auto_advance:
                    advance()
                zoom_center = None
                continue

            if key == -1:
                continue
            key &= 0xFF
            if key == ord("q"):
                save()
                break
            if key == ord("s"):
                save()
            elif key == ord("v"):
                active_visibility = "visible"
                if record["x"] is not None:
                    record["visibility"] = "visible"
                    save()
            elif key == ord("b"):
                active_visibility = "blurred"
                if record["x"] is not None:
                    record["visibility"] = "blurred"
                    save()
            elif key == ord("o"):
                label_without_point("occluded")
            elif key == ord("n"):
                label_without_point("not_visible")
            elif key == ord("x"):
                label_without_point("out_of_frame")
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                active_shot_state = {
                    ord("1"): "address",
                    ord("2"): "flight",
                    ord("3"): "landed",
                    ord("4"): "finished",
                }[key]
                record["shot_state"] = active_shot_state
                save()
            elif key == ord("u"):
                record["uncertain"] = not record["uncertain"]
                save()
            elif key == ord("c"):
                record.update(
                    x=None,
                    y=None,
                    visibility="unlabeled",
                    shot_state="unknown",
                    uncertain=False,
                    blur=None,
                )
                save()
            elif key == ord("j"):
                advance(-1)
            elif key in (ord("l"), ord(" "), 13):
                advance()
            elif key == ord("["):
                advance(-10)
            elif key == ord("]"):
                advance(10)
            elif key == ord("t"):
                auto_advance = not auto_advance
            elif key == ord("z"):
                zoom_center = None
    finally:
        save()
        reader.release()
        cv2.destroyAllWindows()

    print(f"Saved annotations to: {output_path}")


if __name__ == "__main__":
    main()
