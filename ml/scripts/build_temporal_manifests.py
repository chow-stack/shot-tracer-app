"""Build deterministic leave-one-video-out manifests for temporal training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from temporal.manifest import (  # noqa: E402
    build_leave_one_video_out_manifests,
    validate_no_video_leakage,
)
from tracker.annotation_io import atomic_write_json  # noqa: E402


def parse_source(value: str) -> tuple[str, str]:
    try:
        annotation, video = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use ANNOTATION=VIDEO") from error
    if not annotation or not video:
        raise argparse.ArgumentTypeError("Use ANNOTATION=VIDEO")
    return annotation, video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        metavar="ANNOTATION=VIDEO",
        help="Repeat once per completely labeled shot video.",
    )
    parser.add_argument("--output-dir", default="ml/temporal_manifests")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--negative-tiles-per-frame", type=int, default=2)
    parser.add_argument("--negative-tiles-per-visible-frame", type=int, default=2)
    args = parser.parse_args()

    manifests = build_leave_one_video_out_manifests(
        args.source,
        tile_size=args.tile_size,
        overlap=args.overlap,
        context_radius=args.context_radius,
        negative_tiles_per_frame=args.negative_tiles_per_frame,
        negative_tiles_per_visible_frame=args.negative_tiles_per_visible_frame,
    )
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        validate_no_video_leakage(manifest)
        path = destination / f"{manifest['fold']['name']}.json"
        atomic_write_json(path, manifest)
        train_count = sum(sample["split"] == "train" for sample in manifest["samples"])
        validation_count = len(manifest["samples"]) - train_count
        print(f"{path}: {train_count} train, {validation_count} validation samples")


if __name__ == "__main__":
    main()
