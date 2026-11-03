"""PyTorch dataset for tiled adjacent-frame golf-ball examples."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def gaussian_heatmap(
    width: int,
    height: int,
    center_x: float | None,
    center_y: float | None,
    sigma: float,
) -> torch.Tensor:
    if width <= 0 or height <= 0 or sigma <= 0:
        raise ValueError("width, height, and sigma must be positive")
    if center_x is None or center_y is None:
        return torch.zeros((1, height, width), dtype=torch.float32)
    y_grid, x_grid = np.ogrid[:height, :width]
    squared_distance = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    heatmap = np.exp(-squared_distance / (2.0 * sigma * sigma)).astype(np.float32)
    return torch.from_numpy(heatmap).unsqueeze(0)


class VideoFrameCache:
    """Small LRU cache with exact-index decode and a sequential fallback."""

    def __init__(self, maximum_frames: int = 18):
        self.maximum_frames = maximum_frames
        self.frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    @staticmethod
    def _decode(path: str, frame_index: int) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        decoded_index = round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if ok and decoded_index == frame_index:
            cap.release()
            return frame

        cap.release()
        cap = cv2.VideoCapture(path)
        frame = None
        for _ in range(frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
        cap.release()
        assert frame is not None
        return frame

    def get(self, path: str, frame_index: int) -> np.ndarray:
        key = (path, frame_index)
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        frame = self._decode(path, frame_index)
        self.frames[key] = frame
        while len(self.frames) > self.maximum_frames:
            self.frames.popitem(last=False)
        return frame


class TemporalTileDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        *,
        input_size: int = 640,
        output_stride: int = 4,
        heatmap_sigma: float = 1.5,
        cache_frames: int = 18,
        cache_dir: str | None = None,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if input_size <= 0 or output_stride <= 0 or input_size % output_stride:
            raise ValueError("input_size must be positive and divisible by output_stride")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.samples = [sample for sample in manifest["samples"] if sample["split"] == split]
        self.videos = {video["video_id"]: video for video in manifest["videos"]}
        self.input_size = input_size
        self.output_stride = output_stride
        self.heatmap_sigma = heatmap_sigma
        self.frame_cache = VideoFrameCache(cache_frames)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self._cache_array: np.memmap | None = None
        self.cache_metadata: dict | None = None
        if self.cache_dir is not None:
            metadata_path = self.cache_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Temporal cache metadata not found: {metadata_path}. "
                    "Run build_temporal_cache.py first."
                )
            with open(metadata_path, encoding="utf-8") as handle:
                cache_metadata = json.load(handle)
            if cache_metadata.get("format") != "uint8_memmap":
                raise ValueError(
                    f"Unsupported temporal cache format: {cache_metadata.get('format')!r}"
                )
            if cache_metadata.get("input_size") != self.input_size:
                raise ValueError(
                    f"Cache input_size={cache_metadata.get('input_size')} does not match "
                    f"requested input_size={self.input_size}"
                )
            data_path = self.cache_dir / cache_metadata["data_file"]
            if not data_path.exists():
                raise FileNotFoundError(f"Temporal cache data not found: {data_path}")
            self.cache_metadata = cache_metadata

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Each DataLoader worker opens its own read-only mapping after spawn.
        state["_cache_array"] = None
        return state

    def _cached_tile(self, video_id: str, frame_index: int, tile_index: int) -> np.ndarray:
        assert self.cache_dir is not None and self.cache_metadata is not None
        if self._cache_array is None:
            self._cache_array = np.memmap(
                self.cache_dir / self.cache_metadata["data_file"],
                dtype=np.uint8,
                mode="r",
                shape=tuple(self.cache_metadata["shape"]),
            )
        key = f"{video_id}/{frame_index:06d}/t{tile_index:02d}"
        try:
            row = self.cache_metadata["index"][key]
        except KeyError as error:
            raise RuntimeError(f"Temporal cache is missing key: {key}") from error
        return np.asarray(self._cache_array[row])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        video = self.videos[sample["video_id"]]
        x1, y1, x2, y2 = sample["tile"]
        channels: list[np.ndarray] = []
        for frame_index in sample["context_frame_indexes"]:
            if self.cache_dir is not None:
                resized = self._cached_tile(
                    sample["video_id"], frame_index, sample["tile_index"]
                )
            else:
                frame = self.frame_cache.get(video["video_path"], frame_index)
                tile = frame[y1:y2, x1:x2]
                resized = cv2.resize(
                    tile,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_AREA,
                )
            channels.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        stacked = np.concatenate(channels, axis=2)
        image = torch.from_numpy(stacked).permute(2, 0, 1).float().div_(255.0)

        output_size = self.input_size // self.output_stride
        target = sample["target"]
        tile_width, tile_height = x2 - x1, y2 - y1
        if target["visible"]:
            center_x = target["x"] * self.input_size / tile_width / self.output_stride
            center_y = target["y"] * self.input_size / tile_height / self.output_stride
        else:
            center_x = center_y = None
        heatmap = gaussian_heatmap(
            output_size, output_size, center_x, center_y, self.heatmap_sigma
        )
        return {
            "image": image,
            "heatmap": heatmap,
            "visibility": torch.tensor(float(target["visible"]), dtype=torch.float32),
            "weight": torch.tensor(0.5 if sample["uncertain"] else 1.0, dtype=torch.float32),
            "sample_id": sample["sample_id"],
            "target_frame_index": sample["target_frame_index"],
        }
