"""Upgrade a track to schema v2 and optionally append an inferred tail."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.trajectory_forecast import (  # noqa: E402
    assisted_continuation,
    cinematic_continuation,
)
from physics.projection import physics_template_continuation  # noqa: E402
from tracker.trajectory_io import build_segments, upgrade_trajectory_document  # noqa: E402
from tracker.video_io import atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("observed_only", "automatic_cinematic", "physics_template", "assisted_landing"),
        default="observed_only",
    )
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--landing-x", type=float)
    parser.add_argument("--landing-y", type=float)
    parser.add_argument("--shot-profile", choices=("wood", "iron", "wedge"))
    parser.add_argument("--horizon-y", type=float)
    return parser.parse_args()


def continue_document(
    source: dict,
    mode: str,
    duration_seconds: float,
    landing: tuple[float, float] | None = None,
    shot_profile: str | None = None,
    shot_profile_source: str = "explicit_user_input",
    horizon_y: float | None = None,
    horizon_source: str = "not_provided",
) -> dict:
    document = upgrade_trajectory_document(source)
    if mode == "observed_only":
        return document
    maximum_duration = 8.0 if mode == "physics_template" else 3.0
    if duration_seconds <= 0 or duration_seconds > maximum_duration:
        raise ValueError(f"Continuation duration must be between 0 and {maximum_duration:g} seconds")
    video = document.get("video", {})
    all_timestamps = video.get("timestamps_seconds")
    if not isinstance(all_timestamps, list):
        raise ValueError("Continuation requires exact video timestamps")
    observed = document["observations"]
    first_frame = document["observation_end_frame"] + 1
    end_time = float(observed[-1]["timestamp_seconds"]) + duration_seconds
    frame_indexes = [
        index
        for index in range(first_frame, len(all_timestamps))
        if float(all_timestamps[index]) <= end_time
    ]
    timestamps = [float(all_timestamps[index]) for index in frame_indexes]
    if not frame_indexes:
        raise ValueError("No video frames remain after the final observation")

    if mode == "automatic_cinematic":
        initialization = document.get("initialization", {})
        address_y = initialization.get("address_y", initialization.get("tap_y"))
        inferred, calculation = cinematic_continuation(
            observed,
            frame_indexes,
            timestamps,
            (int(video["width"]), int(video["height"])),
            float(address_y) if isinstance(address_y, (int, float)) else None,
        )
        document["continuation"] = {
            "mode": mode,
            "quality_status": "cinematic_only",
            "model": {
                "family": "constrained_cubic_hermite_2d",
                "fit_end_frame": document["observation_end_frame"],
            },
            **calculation,
            "disclaimer": "Plausible screen-space completion; not a measured landing.",
        }
    elif mode == "physics_template":
        if shot_profile is None:
            raise ValueError("physics_template requires a shot profile")
        initialization = document.get("initialization", {})
        address_x = initialization.get("address_x", initialization.get("tap_x"))
        address_y = initialization.get("address_y", initialization.get("tap_y"))
        if not isinstance(address_x, (int, float)) or not isinstance(address_y, (int, float)):
            raise ValueError("physics_template requires an address point")
        inferred, continuation = physics_template_continuation(
            observed,
            frame_indexes,
            timestamps,
            (int(video["width"]), int(video["height"])),
            (float(address_x), float(address_y)),
            shot_profile,
            shot_profile_source,
            horizon_y,
            horizon_source,
        )
        document["continuation"] = continuation
    elif mode == "assisted_landing":
        if landing is None:
            raise ValueError("Assisted landing requires landing x/y")
        inferred = assisted_continuation(
            observed[-1], observed[-2], landing, frame_indexes, timestamps
        )
        document["continuation"] = {
            "mode": mode,
            "quality_status": "user_constrained",
            "model": {"family": "cubic_hermite_2d"},
            "endpoint": {"source": "user_tap", "x": landing[0], "y": landing[1]},
        }
    else:
        raise ValueError(f"Unknown continuation mode: {mode}")

    document["continuation_mode"] = mode
    document["trajectory"] = document["trajectory"] + inferred
    document["segments"] = build_segments(document["trajectory"])
    return document


def main() -> None:
    args = parse_args()
    with args.trajectory.open(encoding="utf-8") as handle:
        source = json.load(handle)
    landing = None
    if args.mode == "assisted_landing":
        if args.landing_x is None or args.landing_y is None:
            raise SystemExit("--landing-x and --landing-y are required for assisted landing")
        landing = (args.landing_x, args.landing_y)
    if args.mode == "physics_template" and args.shot_profile is None:
        raise SystemExit("--shot-profile is required for physics_template")
    duration_seconds = args.duration_seconds
    if duration_seconds is None:
        duration_seconds = 6.0 if args.mode == "physics_template" else 2.0
    output = continue_document(
        source,
        args.mode,
        duration_seconds,
        landing,
        args.shot_profile,
        "explicit_user_input",
        args.horizon_y,
        "explicit_user_input" if args.horizon_y is not None else "not_provided",
    )
    atomic_write_json(args.output, output)
    inferred_count = sum(point["provenance"] == "inferred" for point in output["trajectory"])
    print(f"Saved schema v2 trajectory with {inferred_count} inferred points to {args.output}")


if __name__ == "__main__":
    main()
