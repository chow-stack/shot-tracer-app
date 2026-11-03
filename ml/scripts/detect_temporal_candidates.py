"""Export top-K candidates from a trained temporal heatmap checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from temporal.manifest import tile_rectangles  # noqa: E402
from temporal.model import TemporalHeatmapNet  # noqa: E402
from tracker.annotation_io import atomic_write_json, load_annotation_document  # noqa: E402


def suppress_nearby_points(candidates: list[dict], distance: float) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(
            (candidate["x"] - existing["x"]) ** 2
            + (candidate["y"] - existing["y"]) ** 2
            > distance * distance
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def prepare_tile_tensor(
    frames: list[np.ndarray], rectangle: tuple[int, int, int, int], input_size: int
) -> torch.Tensor:
    x1, y1, x2, y2 = rectangle
    channels = []
    for frame in frames:
        tile = frame[y1:y2, x1:x2]
        resized = cv2.resize(tile, (input_size, input_size), interpolation=cv2.INTER_AREA)
        channels.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    stacked = np.concatenate(channels, axis=2)
    return torch.from_numpy(stacked).permute(2, 0, 1).float().div_(255.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--duplicate-distance", type=float, default=12.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.01)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.top_k <= 0:
        raise SystemExit("--batch and --top-k must be positive")
    annotations = load_annotation_document(args.annotations)
    metadata = annotations["video"]
    flight_indexes = [
        frame["frame_index"]
        for frame in annotations["frames"]
        if frame["shot_state"] == "flight"
    ]
    if args.max_frames is not None:
        flight_indexes = flight_indexes[: args.max_frames]
    if not flight_indexes:
        raise SystemExit("No flight frames selected")
    selected = set(flight_indexes)

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_settings = checkpoint.get("model", {})
    if model_settings.get("context_frames", 3) != 3:
        raise SystemExit("This inference path currently requires a three-frame checkpoint")
    input_size = int(checkpoint["training"]["input_size"])
    model = TemporalHeatmapNet(context_frames=3).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rectangles = tile_rectangles(
        metadata["width"], metadata["height"], args.tile_size, args.overlap
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    frame_buffer: dict[int, np.ndarray] = {}
    output_candidates: list[dict] = []
    inference_seconds = 0.0
    processed = 0
    maximum_needed = min(metadata["frame_count"] - 1, max(flight_indexes) + 1)
    started = time.perf_counter()
    progress = tqdm(total=len(flight_indexes), desc="Temporal inference", unit="frame")

    for decoded_index in range(maximum_needed + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not decode frame {decoded_index}")
        frame_buffer[decoded_index] = frame
        center = decoded_index - 1
        if center in selected:
            context = [frame_buffer[center - 1], frame_buffer[center], frame_buffer[center + 1]]
            tensors = [prepare_tile_tensor(context, rectangle, input_size) for rectangle in rectangles]
            frame_candidates: list[dict] = []
            inference_started = time.perf_counter()
            with torch.no_grad():
                for offset in range(0, len(tensors), args.batch):
                    batch = torch.stack(tensors[offset : offset + args.batch]).to(device)
                    heatmap_logits, visibility_logits = model(batch)
                    heatmaps = torch.sigmoid(heatmap_logits).cpu()
                    visibilities = torch.sigmoid(visibility_logits).cpu()
                    for local_index in range(heatmaps.shape[0]):
                        tile_index = offset + local_index
                        heatmap = heatmaps[local_index, 0]
                        flat_index = int(torch.argmax(heatmap))
                        heatmap_y, heatmap_x = divmod(flat_index, heatmap.shape[1])
                        peak = float(heatmap[heatmap_y, heatmap_x])
                        visibility = float(visibilities[local_index])
                        confidence = peak * visibility
                        if confidence < args.minimum_confidence:
                            continue
                        x1, y1, x2, y2 = rectangles[tile_index]
                        x = x1 + (heatmap_x + 0.5) * (x2 - x1) / heatmap.shape[1]
                        y = y1 + (heatmap_y + 0.5) * (y2 - y1) / heatmap.shape[0]
                        frame_candidates.append(
                            {
                                "frame_index": center,
                                "x": x,
                                "y": y,
                                "confidence": confidence,
                                "heatmap_peak": peak,
                                "visibility_confidence": visibility,
                                "tile_index": tile_index,
                                "source": "temporal_heatmap_3_frame",
                            }
                        )
            inference_seconds += time.perf_counter() - inference_started
            merged = suppress_nearby_points(frame_candidates, args.duplicate_distance)
            for rank, candidate in enumerate(merged[: args.top_k], start=1):
                candidate["rank"] = rank
                output_candidates.append(candidate)
            processed += 1
            progress.update(1)
            progress.set_postfix(candidates=len(output_candidates))
        for old_index in list(frame_buffer):
            if old_index < decoded_index - 2:
                del frame_buffer[old_index]
    cap.release()
    progress.close()

    output_path = Path(
        args.output
        or Path("ml/output") / f"{Path(args.video).stem}.temporal-candidates.json"
    ).resolve()
    document = {
        "schema_version": 1,
        "detector": "temporal_heatmap_3_frame",
        "video": {
            "path": str(Path(args.video).resolve()),
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
        },
        "settings": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "input_size": input_size,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "top_k": args.top_k,
            "device": str(device),
            "scope": "annotated_flight_frames",
        },
        "performance": {
            "processed_frames": processed,
            "tiles_per_frame": len(rectangles),
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
            "mean_inference_ms_per_frame": 1000.0 * inference_seconds / processed,
        },
        "candidates": output_candidates,
    }
    atomic_write_json(output_path, document)
    print(f"Saved {len(output_candidates)} candidates to: {output_path}")
    print(f"Mean inference: {document['performance']['mean_inference_ms_per_frame']:.1f}ms/frame")


if __name__ == "__main__":
    main()
