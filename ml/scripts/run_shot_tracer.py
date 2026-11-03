"""Run the complete offline shot tracer without ground-truth annotations."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from associate_candidates import associate_candidate_files
from detect_motion_candidates import detect_motion_candidates
from detect_tiled_candidates import DEFAULT_MODEL, detect_tiled_candidates
from render_associated_track import render

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.video_io import atomic_write_json, probe_video_metadata  # noqa: E402


def default_video_output(video_path: str) -> Path:
    return Path(__file__).resolve().parent.parent / "output" / f"{Path(video_path).stem}.annotation-free.mp4"


def artifact_path(output_video: Path, suffix: str) -> Path:
    return output_video.with_name(f"{output_video.stem}.{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--tap-x", required=True, type=float)
    parser.add_argument("--tap-y", required=True, type=float)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--motion-top-k", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=600)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--yolo-end-margin", type=int, default=30)
    parser.add_argument("--fallback-stride", type=int, default=30)
    parser.add_argument("--skip-yolo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    metadata = probe_video_metadata(args.video)
    tap = (args.tap_x, args.tap_y)
    if not (0 <= tap[0] < metadata["width"] and 0 <= tap[1] < metadata["height"]):
        raise SystemExit("The address-ball tap is outside the video frame")
    if args.motion_top_k <= 0 or args.beam_width <= 0:
        raise SystemExit("Candidate count and beam width must be positive")
    if args.yolo_end_margin < 0 or args.fallback_stride <= 0:
        raise SystemExit("YOLO margin must be non-negative and fallback stride positive")

    output_video = (args.output or default_video_output(args.video)).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    motion_path = artifact_path(output_video, "motion-candidates.json")
    motion_track_path = artifact_path(output_video, "motion-track.json")
    tiled_path = artifact_path(output_video, "tiled-candidates.json")
    track_path = artifact_path(output_video, "track.json")
    run_path = artifact_path(output_video, "run.json")

    stage_times: dict[str, float] = {}
    stage_started = time.perf_counter()
    detect_motion_candidates(
        args.video,
        tap,
        str(motion_path),
        metadata=metadata,
        top_k=args.motion_top_k,
    )
    stage_times["motion_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    motion_track = associate_candidate_files(
        [str(motion_path)],
        tap,
        str(track_path if args.skip_yolo else motion_track_path),
        beam_width=args.beam_width,
        max_gap=args.max_gap,
    )
    stage_times["motion_association_seconds"] = time.perf_counter() - stage_started

    candidate_paths = [str(motion_path)]
    final_track = motion_track
    if not args.skip_yolo:
        launch = int(motion_track["launch_frame"])
        termination = int(motion_track["termination_frame"])
        last_searchable = metadata["frame_count"] - 2
        window_end = min(last_searchable, termination + args.yolo_end_margin)
        search_indexes = set(range(max(1, launch - 1), window_end + 1))
        search_indexes.update(range(window_end + 1, last_searchable + 1, args.fallback_stride))
        stage_started = time.perf_counter()
        detect_tiled_candidates(
            args.video,
            search_indexes,
            str(tiled_path),
            metadata=metadata,
            model_path=args.model,
            device=args.device,
        )
        stage_times["tiled_yolo_seconds"] = time.perf_counter() - stage_started
        candidate_paths.append(str(tiled_path))

        stage_started = time.perf_counter()
        final_track = associate_candidate_files(
            candidate_paths,
            tap,
            str(track_path),
            beam_width=args.beam_width,
            max_gap=args.max_gap,
        )
        stage_times["combined_association_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    rendered_frames = render(Path(args.video), track_path, output_video)
    stage_times["render_seconds"] = time.perf_counter() - stage_started
    stage_times["total_seconds"] = time.perf_counter() - started

    run_document = {
        "schema_version": 1,
        "pipeline": "annotation_free_offline_shot_tracer",
        "video": metadata,
        "initialization": {"tap_x": tap[0], "tap_y": tap[1], "source": "explicit_user_input"},
        "settings": {
            "motion_top_k": args.motion_top_k,
            "beam_width": args.beam_width,
            "max_gap": args.max_gap,
            "tiled_yolo_enabled": not args.skip_yolo,
            "model": os.path.abspath(args.model) if not args.skip_yolo else None,
            "device": args.device if not args.skip_yolo else None,
            "yolo_end_margin": args.yolo_end_margin,
            "fallback_stride": args.fallback_stride,
        },
        "result": {
            "launch_frame": final_track["launch_frame"],
            "termination_frame": final_track["termination_frame"],
            "trajectory_points": len(final_track["trajectory"]),
            "rendered_frames": rendered_frames,
        },
        "artifacts": {
            "motion_candidates": str(motion_path),
            "tiled_candidates": str(tiled_path) if not args.skip_yolo else None,
            "trajectory": str(track_path),
            "video": str(output_video),
        },
        "performance": stage_times,
    }
    atomic_write_json(run_path, run_document)
    print(
        f"Completed annotation-free trace: launch={final_track['launch_frame']}, "
        f"termination={final_track['termination_frame']}, output={output_video}"
    )


if __name__ == "__main__":
    main()
