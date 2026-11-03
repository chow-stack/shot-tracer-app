"""Predecode lossless temporal tiles so video seeking cannot starve training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.annotation_io import atomic_write_json  # noqa: E402


def required_tiles(manifest: dict) -> dict[str, dict[int, set[int]]]:
    required: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for sample in manifest["samples"]:
        for frame_index in sample["context_frame_indexes"]:
            required[sample["video_id"]][frame_index].add(sample["tile_index"])
    return required


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", default="ml/cache/temporal_640")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.input_size <= 0:
        raise SystemExit("--input-size must be positive")
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    videos = {video["video_id"]: video for video in manifest["videos"]}
    required = required_tiles(manifest)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_count = sum(
        len(tile_indexes)
        for frames in required.values()
        for tile_indexes in frames.values()
    )
    metadata_path = output_dir / "metadata.json"
    data_path = output_dir / "tiles.uint8"
    if metadata_path.exists() and data_path.exists() and not args.rebuild:
        with open(metadata_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            existing.get("format") == "uint8_memmap"
            and existing.get("input_size") == args.input_size
            and existing.get("tile_count") == expected_count
        ):
            print(f"Cache already complete: {output_dir}", flush=True)
            return
        raise RuntimeError(
            f"An incompatible cache exists at {output_dir}; pass --rebuild to replace it"
        )

    ordered_keys = sorted(
        (video_id, frame_index, tile_index)
        for video_id, frames in required.items()
        for frame_index, tile_indexes in frames.items()
        for tile_index in tile_indexes
    )
    index = {
        f"{video_id}/{frame_index:06d}/t{tile_index:02d}": row
        for row, (video_id, frame_index, tile_index) in enumerate(ordered_keys)
    }
    row_lookup = {key: row for row, key in enumerate(ordered_keys)}
    required_bytes = expected_count * args.input_size * args.input_size * 3
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < required_bytes * 1.1:
        raise RuntimeError(
            f"Cache requires {required_bytes / 2**30:.2f} GiB but only "
            f"{free_bytes / 2**30:.2f} GiB is free"
        )
    print(
        f"Building exact uint8 cache: {expected_count} tiles, "
        f"{required_bytes / 2**30:.2f} GiB",
        flush=True,
    )
    temporary_data_path = output_dir / ".tiles.uint8.tmp"
    cache = np.memmap(
        temporary_data_path,
        dtype=np.uint8,
        mode="w+",
        shape=(expected_count, args.input_size, args.input_size, 3),
    )
    progress = tqdm(total=expected_count, desc="Caching exact tiles", unit="tile")

    for video_id in sorted(required):
        video = videos[video_id]
        cap = cv2.VideoCapture(video["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video['video_path']}")
        maximum_frame = max(required[video_id])
        rectangles = video["tile_rectangles"]
        for frame_index in range(maximum_frame + 1):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Could not decode frame {frame_index} from {video['video_path']}"
                )
            for tile_index in sorted(required[video_id].get(frame_index, ())):
                x1, y1, x2, y2 = rectangles[tile_index]
                tile = frame[y1:y2, x1:x2]
                if tile.shape[:2] != (args.input_size, args.input_size):
                    tile = cv2.resize(
                        tile,
                        (args.input_size, args.input_size),
                        interpolation=cv2.INTER_AREA,
                    )
                cache[row_lookup[(video_id, frame_index, tile_index)]] = tile
                progress.update(1)
        cap.release()
    progress.close()
    cache.flush()
    del cache
    os.replace(temporary_data_path, data_path)
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "format": "uint8_memmap",
            "data_file": data_path.name,
            "input_size": args.input_size,
            "tile_count": expected_count,
            "shape": [expected_count, args.input_size, args.input_size, 3],
            "index": index,
            "video_sha256": {
                video_id: videos[video_id]["sha256"] for video_id in sorted(required)
            },
        },
    )
    print(f"Cache ready: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
