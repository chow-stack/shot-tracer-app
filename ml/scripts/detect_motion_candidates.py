"""Generate top-K golf-ball candidates from motion without a learned detector.

This is an annotation-free candidate generator, not a tracker. It sequentially
decodes the complete clip and uses only the address-ball tap supplied by the
user.

The primary signal is a three-frame intersection: pixels that change from the
previous frame to the current frame *and* from the current frame to the next
frame. A median address-period background provides a lower-priority fallback
for slower or overlapping motion.

Usage:
    ml/.venv/bin/python ml/scripts/detect_motion_candidates.py \
        ml/videos/curve-iron.MOV --tap-x 576 --tap-y 1659.7333
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.video_io import atomic_write_json, probe_video_metadata  # noqa: E402


def parse_thresholds(value: str) -> tuple[int, ...]:
    try:
        thresholds = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated integers") from error
    if not thresholds or any(threshold <= 0 or threshold > 255 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be integers in [1, 255]")
    return thresholds


def frame_change(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the strongest absolute BGR-channel change per pixel."""
    difference = cv2.absdiff(first, second)
    # OpenCV's vectorized uint8 maxima are exactly equivalent to NumPy's
    # channel reduction here and avoid a large temporary reduction buffer for
    # every full-resolution frame comparison.
    return cv2.max(cv2.max(difference[:, :, 0], difference[:, :, 1]), difference[:, :, 2])


def three_frame_response(
    previous: np.ndarray, current: np.ndarray, following: np.ndarray
) -> np.ndarray:
    """Isolate pixels belonging to an object present in the middle frame."""
    return np.minimum(frame_change(previous, current), frame_change(current, following))


def percentile90_uint8(values: np.ndarray) -> float:
    """Exact NumPy-linear 90th percentile for a non-empty uint8 vector.

    Motion-component arrays contain at most ``max_area`` values. Avoiding the
    generic quantile dispatcher matters because this function runs hundreds of
    thousands of times per clip.
    """
    if values.ndim != 1 or values.size == 0 or values.dtype != np.uint8:
        raise ValueError("percentile90_uint8 requires a non-empty uint8 vector")
    rank = 0.9 * (values.size - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(np.partition(values, lower)[lower])
    partitioned = np.partition(values, (lower, upper))
    low_value = float(partitioned[lower])
    high_value = float(partitioned[upper])
    weight = rank - lower
    # NumPy's linear interpolation changes direction above 0.5 to limit
    # floating-point error. Matching that detail preserves exported JSON bits.
    if weight < 0.5:
        return low_value + (high_value - low_value) * weight
    return high_value - (high_value - low_value) * (1.0 - weight)


def choose_background_indexes(background_end: int, sample_count: int) -> set[int]:
    """Sample the early address portion without needing an impact label."""
    if sample_count <= 0 or background_end <= 0:
        return set()
    count = min(sample_count, background_end)
    return {int(value) for value in np.linspace(0, background_end - 1, count)}


def component_candidates(
    response: np.ndarray,
    thresholds: tuple[int, ...],
    *,
    frame_index: int,
    source: str,
    source_bonus: float,
    min_area: int,
    max_area: int,
    max_dimension: int,
    max_aspect_ratio: float,
    maximum_y: float | None,
) -> list[dict]:
    """Extract permissive small/streak-like connected components."""
    candidates: list[dict] = []
    for threshold in thresholds:
        # CMP_GE preserves the previous inclusive threshold semantics while
        # keeping the operation inside OpenCV's optimized uint8 path.
        binary = cv2.compare(response, threshold, cv2.CMP_GE)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if not min_area <= area <= max_area:
                continue
            if max(width, height) > max_dimension or min(width, height) <= 0:
                continue
            if max(width, height) / min(width, height) > max_aspect_ratio:
                continue

            component_mask = labels[y : y + height, x : x + width] == label
            values = response[y : y + height, x : x + width][component_mask]
            if values.size == 0:
                continue
            weights = values.astype(np.float64)
            local_y, local_x = np.nonzero(component_mask)
            center_x = x + float(np.average(local_x, weights=weights))
            center_y = y + float(np.average(local_y, weights=weights))
            if maximum_y is not None and center_y > maximum_y:
                continue

            peak = percentile90_uint8(values)
            mean = float(np.mean(values))
            size_score = min(1.0, math.sqrt(area / 8.0)) * math.exp(-area / max_area)
            confidence = min(
                1.0,
                source_bonus + 0.55 * (peak / 255.0) + 0.25 * (mean / 255.0) + 0.20 * size_score,
            )
            candidates.append(
                {
                    "frame_index": frame_index,
                    "x": center_x,
                    "y": center_y,
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x + width),
                    "y2": float(y + height),
                    "width": width,
                    "height": height,
                    "area": area,
                    "response_peak": peak,
                    "response_mean": mean,
                    "threshold": threshold,
                    "confidence": confidence,
                    "source": source,
                }
            )
    return candidates


def merge_nearby_candidates(candidates: list[dict], radius: float) -> list[dict]:
    """Remove the same response extracted at several thresholds/sources."""
    kept: list[dict] = []
    if radius == 0:
        by_position: dict[tuple[float, float], dict] = {}
        for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            key = (candidate["x"], candidate["y"])
            duplicate = by_position.get(key)
            if duplicate is None:
                by_position[key] = candidate
                kept.append(candidate)
            else:
                sources = set(duplicate.get("supporting_sources", [duplicate["source"]]))
                sources.add(candidate["source"])
                duplicate["supporting_sources"] = sorted(sources)
                duplicate["confidence"] = min(
                    1.0,
                    duplicate["confidence"]
                    + (0.05 if candidate["source"] != duplicate["source"] else 0.0),
                )
        return sorted(kept, key=lambda item: item["confidence"], reverse=True)
    grid: dict[tuple[int, int], list[int]] = {}
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        cell = (math.floor(candidate["x"] / radius), math.floor(candidate["y"] / radius))
        nearby_indexes: list[int] = []
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                nearby_indexes.extend(grid.get((cell[0] + offset_x, cell[1] + offset_y), ()))
        duplicate_index = next(
            (
                index
                for index in sorted(nearby_indexes)
                if math.hypot(
                    candidate["x"] - kept[index]["x"],
                    candidate["y"] - kept[index]["y"],
                )
                <= radius
            ),
            None,
        )
        if duplicate_index is None:
            grid.setdefault(cell, []).append(len(kept))
            kept.append(candidate)
        else:
            duplicate = kept[duplicate_index]
            sources = set(duplicate.get("supporting_sources", [duplicate["source"]]))
            sources.add(candidate["source"])
            duplicate["supporting_sources"] = sorted(sources)
            duplicate["confidence"] = min(
                1.0,
                duplicate["confidence"] + (0.05 if candidate["source"] != duplicate["source"] else 0.0),
            )
    # Cross-source support can increase confidence after insertion, so restore
    # confidence order before top-K truncation.
    return sorted(kept, key=lambda item: item["confidence"], reverse=True)


def default_output_path(video_path: str) -> str:
    ml_root = Path(__file__).resolve().parent.parent
    return str(ml_root / "output" / f"{Path(video_path).stem}.motion-candidates.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--tap-x", required=True, type=float)
    parser.add_argument("--tap-y", required=True, type=float)
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--temporal-thresholds", type=parse_thresholds, default=(10, 18, 30))
    parser.add_argument("--background-thresholds", type=parse_thresholds, default=(20, 32, 48))
    parser.add_argument("--background-samples", type=int, default=15)
    parser.add_argument("--background-end-frame", type=int, default=120)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=800)
    parser.add_argument("--max-dimension", type=int, default=100)
    parser.add_argument("--max-aspect-ratio", type=float, default=20.0)
    parser.add_argument("--merge-radius", type=float, default=8.0)
    parser.add_argument("--below-address-margin", type=float, default=120.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.background_samples < 0:
        raise SystemExit("--background-samples must be non-negative")
    if args.background_end_frame <= 0:
        raise SystemExit("--background-end-frame must be positive")
    if args.min_area <= 0 or args.max_area < args.min_area:
        raise SystemExit("area bounds are invalid")
    if args.max_dimension <= 0 or args.max_aspect_ratio < 1:
        raise SystemExit("component shape bounds are invalid")
    if args.merge_radius < 0 or args.below_address_margin < 0:
        raise SystemExit("radius/margin values must be non-negative")


def detect_motion_candidates(
    video_path: str,
    tap: tuple[float, float],
    output_path: str,
    *,
    metadata: dict | None = None,
    top_k: int = 100,
    temporal_thresholds: tuple[int, ...] = (10, 18, 30),
    background_thresholds: tuple[int, ...] = (20, 32, 48),
    background_samples: int = 15,
    background_end_frame: int = 120,
    min_area: int = 2,
    max_area: int = 800,
    max_dimension: int = 100,
    max_aspect_ratio: float = 20.0,
    merge_radius: float = 8.0,
    below_address_margin: float = 120.0,
) -> dict:
    """Generate full-clip candidates and return the exported document."""
    metadata = metadata or probe_video_metadata(video_path)
    frame_count = int(metadata["frame_count"])
    if frame_count < 3:
        raise ValueError("Motion extraction requires at least three decodable frames")
    background_end_frame = min(background_end_frame, frame_count - 2)
    maximum_y = tap[1] + below_address_margin
    background_indexes = choose_background_indexes(background_end_frame, background_samples)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, current = cap.read()
    if not ok:
        raise RuntimeError("Could not decode frame 0")
    previous: np.ndarray | None = None
    background_frames: list[np.ndarray] = []
    background: np.ndarray | None = None
    output_candidates: list[dict] = []
    raw_component_count = 0
    processed_frames = 0
    frame_index = 0
    started = time.perf_counter()

    print(
        f"Processing {frame_count} full-clip frames with motion components; "
        f"temporal thresholds={temporal_thresholds}, "
        f"background thresholds={background_thresholds}"
    )
    while frame_index < frame_count - 1:
        ok, following = cap.read()
        if not ok:
            raise RuntimeError(
                f"Video ended at look-ahead frame {frame_index + 1}; "
                f"ffprobe reported {frame_count} frames"
            )

        if frame_index in background_indexes:
            background_frames.append(current.copy())
        if background is None and frame_index >= background_end_frame:
            if background_frames:
                background = np.median(np.stack(background_frames), axis=0).astype(np.uint8)
            background_frames.clear()

        if previous is not None:
            temporal = three_frame_response(previous, current, following)
            frame_candidates = component_candidates(
                temporal,
                temporal_thresholds,
                frame_index=frame_index,
                source="three_frame_motion",
                source_bonus=0.15,
                min_area=min_area,
                max_area=max_area,
                max_dimension=max_dimension,
                max_aspect_ratio=max_aspect_ratio,
                maximum_y=maximum_y,
            )
            if background is not None:
                background_change = frame_change(background, current)
                frame_candidates.extend(
                    component_candidates(
                        background_change,
                        background_thresholds,
                        frame_index=frame_index,
                        source="median_background_motion",
                        source_bonus=0.0,
                        min_area=min_area,
                        max_area=max_area,
                        max_dimension=max_dimension,
                        max_aspect_ratio=max_aspect_ratio,
                        maximum_y=maximum_y,
                    )
                )
            raw_component_count += len(frame_candidates)
            merged = merge_nearby_candidates(frame_candidates, merge_radius)
            for rank, candidate in enumerate(merged[:top_k], start=1):
                candidate["rank"] = rank
                output_candidates.append(candidate)
            processed_frames += 1
            if processed_frames % 25 == 0 or processed_frames == frame_count - 2:
                print(
                    f"  {processed_frames}/{frame_count - 2} searchable frames, "
                    f"{len(output_candidates)} retained candidates"
                )

        previous, current = current, following
        frame_index += 1

    cap.release()
    total_seconds = time.perf_counter() - started
    document = {
        "schema_version": 1,
        "detector": "motion_components",
        "video": {
            "path": os.path.abspath(video_path),
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "timestamps_seconds": metadata["timestamps_seconds"],
        },
        "initialization": {
            "address_point": {"x": tap[0], "y": tap[1]} if tap else None,
            "source": "explicit user-provided address tap" if tap else None,
        },
        "settings": {
            "temporal_thresholds": list(temporal_thresholds),
            "background_thresholds": list(background_thresholds),
            "background_samples_requested": background_samples,
            "background_samples_used": len(background_indexes),
            "background_end_frame": background_end_frame,
            "min_area": min_area,
            "max_area": max_area,
            "max_dimension": max_dimension,
            "max_aspect_ratio": max_aspect_ratio,
            "merge_radius": merge_radius,
            "below_address_margin": below_address_margin,
            "top_k": top_k,
            "scope": "complete_decodable_clip",
        },
        "performance": {
            "processed_frames": processed_frames,
            "raw_components": raw_component_count,
            "total_seconds": total_seconds,
            "mean_ms_per_frame": 1000.0 * total_seconds / max(1, processed_frames),
        },
        "candidates": output_candidates,
    }
    atomic_write_json(output_path, document)
    print(f"Saved {len(output_candidates)} candidates to: {output_path}")
    print(
        f"Processing: {total_seconds:.2f}s total, "
        f"{document['performance']['mean_ms_per_frame']:.1f}ms/searchable frame"
    )
    return document


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_path = os.path.abspath(args.output or default_output_path(args.video))
    detect_motion_candidates(
        args.video,
        (args.tap_x, args.tap_y),
        output_path,
        top_k=args.top_k,
        temporal_thresholds=args.temporal_thresholds,
        background_thresholds=args.background_thresholds,
        background_samples=args.background_samples,
        background_end_frame=args.background_end_frame,
        min_area=args.min_area,
        max_area=args.max_area,
        max_dimension=args.max_dimension,
        max_aspect_ratio=args.max_aspect_ratio,
        merge_radius=args.merge_radius,
        below_address_margin=args.below_address_margin,
    )


if __name__ == "__main__":
    main()
