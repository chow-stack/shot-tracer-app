"""Generate top-K golf-ball candidates with overlapping full-frame tiles.

This is an annotation-free supporting detector, not a tracker. It deliberately
removes the Kalman crop so a previous mistake cannot exclude the ball.

Usage:
    ml/.venv/bin/python ml/scripts/detect_tiled_candidates.py \
        ml/videos/curve-iron.MOV
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.video_io import atomic_write_json, probe_video_metadata  # noqa: E402


DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "golf_ball_yolov8n", "weights", "best.pt"
)


def axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def tile_rectangles(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    return [
        (x, y, x + tile_width, y + tile_height)
        for y in axis_starts(height, tile_height, overlap)
        for x in axis_starts(width, tile_width, overlap)
    ]


def box_iou(first: dict, second: dict) -> float:
    intersection_width = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"]))
    intersection_height = max(0.0, min(first["y2"], second["y2"]) - max(first["y1"], second["y1"]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
    second_area = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def merge_cross_tile_duplicates(candidates: list[dict], iou_threshold: float) -> list[dict]:
    """Confidence-ordered NMS after tile coordinates are mapped to the frame."""
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(box_iou(candidate, existing) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def default_output_path(video_path: str) -> str:
    ml_root = Path(__file__).resolve().parent.parent
    return str(ml_root / "output" / f"{Path(video_path).stem}.tiled-candidates.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--cross-tile-iou", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def detect_tiled_candidates(
    video_path: str,
    search_indexes: set[int],
    output_path: str,
    *,
    metadata: dict | None = None,
    model_path: str = DEFAULT_MODEL,
    tile_size: int = 640,
    overlap: float = 0.25,
    confidence: float = 0.01,
    cross_tile_iou: float = 0.3,
    top_k: int = 5,
    device: str = "mps",
) -> dict:
    """Run tiled YOLO on an automatically chosen supporting search schedule."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    metadata = metadata or probe_video_metadata(video_path)
    frame_count = int(metadata["frame_count"])
    search_indexes = {index for index in search_indexes if 0 <= index < frame_count}
    if not search_indexes:
        raise ValueError("search_indexes must contain at least one decodable frame")
    rectangles = tile_rectangles(
        metadata["width"], metadata["height"], tile_size, overlap
    )
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_candidates: list[dict] = []
    processed_frames = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    frame_index = 0
    maximum_search_frame = max(search_indexes)

    print(
        f"Processing {len(search_indexes)} supporting frames with {len(rectangles)} "
        f"overlapping {tile_size}px tiles per frame on {device}..."
    )
    while frame_index <= maximum_search_frame:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended before annotated frame {frame_index}")
        if frame_index in search_indexes:
            tiles = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in rectangles]
            inference_started = time.perf_counter()
            results = model.predict(
                tiles,
                conf=confidence,
                imgsz=tile_size,
                device=device,
                verbose=False,
            )
            inference_seconds += time.perf_counter() - inference_started

            frame_candidates: list[dict] = []
            for tile_index, (result, rectangle) in enumerate(zip(results, rectangles)):
                offset_x, offset_y, _, _ = rectangle
                for box in result.boxes:
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    detection_confidence = float(box.conf[0])
                    global_box = {
                        "frame_index": frame_index,
                        "x1": offset_x + x1,
                        "y1": offset_y + y1,
                        "x2": offset_x + x2,
                        "y2": offset_y + y2,
                        "x": offset_x + (x1 + x2) / 2.0,
                        "y": offset_y + (y1 + y2) / 2.0,
                        "confidence": detection_confidence,
                        "tile_index": tile_index,
                    }
                    frame_candidates.append(global_box)

            merged = merge_cross_tile_duplicates(frame_candidates, cross_tile_iou)
            for rank, candidate in enumerate(merged[:top_k], start=1):
                candidate["rank"] = rank
                candidate["source"] = "overlapping_tiled_yolo"
                output_candidates.append(candidate)

            processed_frames += 1
            if processed_frames % 25 == 0 or processed_frames == len(search_indexes):
                print(
                    f"  {processed_frames}/{len(search_indexes)} frames, "
                    f"{len(output_candidates)} retained candidates"
                )
        frame_index += 1

    cap.release()
    total_seconds = time.perf_counter() - started
    document = {
        "schema_version": 1,
        "detector": "overlapping_tiled_yolo",
        "video": {
            "path": os.path.abspath(video_path),
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "timestamps_seconds": metadata["timestamps_seconds"],
        },
        "settings": {
            "model": os.path.abspath(model_path),
            "tile_size": tile_size,
            "overlap": overlap,
            "confidence_threshold": confidence,
            "cross_tile_iou_threshold": cross_tile_iou,
            "top_k": top_k,
            "device": device,
            "scope": "automatic_motion_track_window_plus_periodic_fallback",
            "search_indexes": sorted(search_indexes),
        },
        "performance": {
            "processed_frames": processed_frames,
            "tiles_per_frame": len(rectangles),
            "total_tile_inferences": processed_frames * len(rectangles),
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
            "mean_inference_ms_per_frame": 1000.0 * inference_seconds / processed_frames,
        },
        "candidates": output_candidates,
    }
    atomic_write_json(output_path, document)
    print(f"Saved {len(output_candidates)} candidates to: {output_path}")
    print(
        f"Inference: {inference_seconds:.2f}s total, "
        f"{document['performance']['mean_inference_ms_per_frame']:.1f}ms/searched frame"
    )
    return document


def main() -> None:
    args = parse_args()
    metadata = probe_video_metadata(args.video)
    search_indexes = set(range(metadata["frame_count"]))
    output_path = os.path.abspath(args.output or default_output_path(args.video))
    detect_tiled_candidates(
        args.video,
        search_indexes,
        output_path,
        metadata=metadata,
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        confidence=args.confidence,
        cross_tile_iou=args.cross_tile_iou,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
