"""Deterministic manifests for temporal tiled golf-ball training."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tracker.annotation_io import POINT_VISIBILITIES, load_annotation_document


MANIFEST_SCHEMA_VERSION = 1


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


def tile_rectangles(
    width: int, height: int, tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    return [
        (x, y, x + tile_width, y + tile_height)
        for y in axis_starts(height, tile_height, overlap)
        for x in axis_starts(width, tile_width, overlap)
    ]


def context_indexes(center: int, frame_count: int, radius: int) -> list[int]:
    if frame_count <= 0 or radius < 0 or not 0 <= center < frame_count:
        raise ValueError("invalid center, frame_count, or radius")
    return [min(frame_count - 1, max(0, center + offset)) for offset in range(-radius, radius + 1)]


def point_in_rectangle(x: float, y: float, rectangle: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rectangle
    return x1 <= x < x2 and y1 <= y < y2


def deterministic_tile_indexes(
    video_id: str, frame_index: int, count: int, tile_count: int
) -> list[int]:
    """Select negatives repeatably without depending on Python's hash seed."""
    if count <= 0 or tile_count <= 0:
        return []
    digest = hashlib.sha256(f"{video_id}:{frame_index}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % tile_count
    step = 1 + int.from_bytes(digest[8:16], "big") % max(1, tile_count - 1)
    selected: list[int] = []
    cursor = start
    while len(selected) < min(count, tile_count):
        if cursor not in selected:
            selected.append(cursor)
        cursor = (cursor + step) % tile_count
        if len(selected) < min(count, tile_count) and cursor in selected:
            cursor = (cursor + 1) % tile_count
    return selected


def build_video_samples(
    annotation_path: str,
    video_path: str,
    *,
    tile_size: int = 640,
    overlap: float = 0.25,
    context_radius: int = 1,
    negative_tiles_per_frame: int = 2,
    negative_tiles_per_visible_frame: int = 2,
) -> tuple[dict, list[dict]]:
    document = load_annotation_document(annotation_path)
    metadata = document["video"]
    video_id = Path(metadata["path"]).stem
    rectangles = tile_rectangles(metadata["width"], metadata["height"], tile_size, overlap)
    samples: list[dict] = []

    for frame in document["frames"]:
        if frame["shot_state"] != "flight" or frame["visibility"] == "unlabeled":
            continue
        frame_index = frame["frame_index"]
        visible = frame["visibility"] in POINT_VISIBILITIES
        if visible:
            positive_tile_indexes = [
                index
                for index, rectangle in enumerate(rectangles)
                if point_in_rectangle(float(frame["x"]), float(frame["y"]), rectangle)
            ]
            available_negative_indexes = [
                index for index in range(len(rectangles)) if index not in positive_tile_indexes
            ]
            selected_positions = deterministic_tile_indexes(
                f"{video_id}:visible-negative",
                frame_index,
                negative_tiles_per_visible_frame,
                len(available_negative_indexes),
            )
            tile_targets = [(index, True) for index in positive_tile_indexes]
            tile_targets.extend(
                (available_negative_indexes[position], False) for position in selected_positions
            )
        else:
            tile_indexes = deterministic_tile_indexes(
                video_id, frame_index, negative_tiles_per_frame, len(rectangles)
            )
            tile_targets = [(index, False) for index in tile_indexes]

        for tile_index, target_visible in tile_targets:
            x1, y1, x2, y2 = rectangles[tile_index]
            target = {"visible": target_visible, "x": None, "y": None}
            if target_visible:
                target["x"] = float(frame["x"]) - x1
                target["y"] = float(frame["y"]) - y1
            samples.append(
                {
                    "sample_id": f"{video_id}:{frame_index:06d}:t{tile_index:02d}",
                    "video_id": video_id,
                    "target_frame_index": frame_index,
                    "context_frame_indexes": context_indexes(
                        frame_index, metadata["frame_count"], context_radius
                    ),
                    "tile_index": tile_index,
                    "tile": [x1, y1, x2, y2],
                    "visibility": frame["visibility"],
                    "uncertain": frame["uncertain"],
                    "target": target,
                }
            )

    video = {
        "video_id": video_id,
        "video_path": str(Path(video_path).resolve()),
        "annotation_path": str(Path(annotation_path).resolve()),
        "sha256": metadata["sha256"],
        "width": metadata["width"],
        "height": metadata["height"],
        "fps": metadata["fps"],
        "frame_count": metadata["frame_count"],
        "tile_rectangles": [list(rectangle) for rectangle in rectangles],
    }
    return video, samples


def build_leave_one_video_out_manifests(
    sources: list[tuple[str, str]],
    *,
    tile_size: int = 640,
    overlap: float = 0.25,
    context_radius: int = 1,
    negative_tiles_per_frame: int = 2,
    negative_tiles_per_visible_frame: int = 2,
) -> list[dict]:
    if len(sources) < 2:
        raise ValueError("At least two complete videos are required for video-level splits")
    videos: list[dict] = []
    samples: list[dict] = []
    for annotation_path, video_path in sources:
        video, video_samples = build_video_samples(
            annotation_path,
            video_path,
            tile_size=tile_size,
            overlap=overlap,
            context_radius=context_radius,
            negative_tiles_per_frame=negative_tiles_per_frame,
            negative_tiles_per_visible_frame=negative_tiles_per_visible_frame,
        )
        videos.append(video)
        samples.extend(video_samples)
    video_ids = [video["video_id"] for video in videos]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video_id values must be unique")

    manifests: list[dict] = []
    for validation_id in sorted(video_ids):
        train_ids = sorted(video_id for video_id in video_ids if video_id != validation_id)
        fold_samples = []
        for sample in samples:
            split = "validation" if sample["video_id"] == validation_id else "train"
            fold_samples.append({**sample, "split": split})
        manifests.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "settings": {
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "context_radius": context_radius,
                    "context_frames": context_radius * 2 + 1,
                    "negative_tiles_per_invisible_frame": negative_tiles_per_frame,
                    "negative_tiles_per_visible_frame": negative_tiles_per_visible_frame,
                    "scope": "labeled_flight_frames",
                },
                "fold": {
                    "name": f"validation-{validation_id}",
                    "train_video_ids": train_ids,
                    "validation_video_ids": [validation_id],
                },
                "videos": videos,
                "samples": fold_samples,
            }
        )
    return manifests


def validate_no_video_leakage(manifest: dict) -> None:
    train_ids = set(manifest["fold"]["train_video_ids"])
    validation_ids = set(manifest["fold"]["validation_video_ids"])
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"Videos occur in both splits: {sorted(overlap)}")
    for sample in manifest["samples"]:
        expected = "validation" if sample["video_id"] in validation_ids else "train"
        if sample["split"] != expected:
            raise ValueError(f"Sample {sample['sample_id']} violates its video-level split")
