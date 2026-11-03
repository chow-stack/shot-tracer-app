"""Evaluate tracker diagnostics against dense frame annotations.

Examples:
    ml/.venv/bin/python ml/scripts/evaluate_tracking.py \
        ml/annotations/curve-iron.annotations.json \
        ml/output/curve-iron.diagnostics.json

    # Candidate-generation recall using at most five peaks per frame
    ml/.venv/bin/python ml/scripts/evaluate_tracking.py annotations.json \
        diagnostics.json --series candidates --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.annotation_io import POINT_VISIBILITIES, load_annotation_document  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", help="Dense ground-truth annotation JSON")
    parser.add_argument("predictions", help="Tracker diagnostics/prediction JSON")
    parser.add_argument(
        "--series",
        choices=("trajectory", "observations", "candidates", "frames"),
        default="trajectory",
        help="Prediction array to evaluate (default: trajectory)",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Candidates retained per frame")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="Maximum center error in original pixels for a correct frame",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Evaluate every labeled state instead of flight frames only",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args()


def load_predictions(path: str, series: str) -> tuple[dict[int, list[dict]], dict]:
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    values = document.get(series)
    if not isinstance(values, list):
        available = ", ".join(
            key for key in ("trajectory", "observations", "candidates", "frames") if key in document
        )
        raise ValueError(f"Prediction file has no {series!r} array; available: {available or 'none'}")

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for position, prediction in enumerate(values):
        if not isinstance(prediction, dict):
            raise ValueError(f"{series}[{position}] must be an object")
        frame_index = prediction.get("frame_index")
        x, y = prediction.get("x"), prediction.get("y")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"{series}[{position}].frame_index is invalid")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"{series}[{position}] requires numeric x/y")
        by_frame[frame_index].append(prediction)

    for predictions in by_frame.values():
        predictions.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return by_frame, document


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def longest_contiguous_run(frame_indexes: set[int]) -> int:
    longest = current = 0
    previous = None
    for frame_index in sorted(frame_indexes):
        if previous is not None and frame_index == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = frame_index
    return longest


def evaluate(
    annotations: dict,
    predictions_by_frame: dict[int, list[dict]],
    prediction_document: dict,
    top_k: int,
    tolerance: float,
    all_states: bool,
    series: str,
) -> dict:
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    labeled_scope = [
        frame
        for frame in annotations["frames"]
        if frame["visibility"] != "unlabeled"
        and (all_states or frame["shot_state"] == "flight")
    ]
    if not labeled_scope:
        qualifier = "labeled" if all_states else "labeled flight-state"
        raise ValueError(f"Annotation file contains no {qualifier} frames")

    target_frames = [
        frame for frame in labeled_scope if frame["visibility"] in POINT_VISIBILITIES
    ]
    if not target_frames:
        raise ValueError("Evaluation scope contains no visible/blurred ground-truth points")

    scope_indexes = {frame["frame_index"] for frame in labeled_scope}
    target_by_index = {frame["frame_index"]: frame for frame in target_frames}
    correct_indexes: set[int] = set()
    center_errors: list[float] = []
    predicted_scope_frames = 0
    correct_predicted_frames = 0
    invisible_prediction_frames = 0

    for frame in labeled_scope:
        frame_index = frame["frame_index"]
        candidates = predictions_by_frame.get(frame_index, [])[:top_k]
        if candidates:
            predicted_scope_frames += 1
        if frame_index not in target_by_index:
            if candidates:
                invisible_prediction_frames += 1
            continue
        if not candidates:
            continue

        error = min(
            math.hypot(float(candidate["x"]) - frame["x"], float(candidate["y"]) - frame["y"])
            for candidate in candidates
        )
        center_errors.append(error)
        if error <= tolerance:
            correct_indexes.add(frame_index)
            correct_predicted_frames += 1

    target_count = len(target_frames)
    correct_count = len(correct_indexes)
    precision = correct_predicted_frames / predicted_scope_frames if predicted_scope_frames else 0.0
    recall = correct_count / target_count

    gt_launch_frames = [
        frame["frame_index"]
        for frame in annotations["frames"]
        if frame["shot_state"] == "flight" and frame["visibility"] != "unlabeled"
    ]
    predicted_indexes = sorted(index for index in predictions_by_frame if index in scope_indexes)
    declared_launch = prediction_document.get("launch_frame")
    if isinstance(declared_launch, int) and gt_launch_frames:
        launch_error = declared_launch - min(gt_launch_frames)
    else:
        launch_error = None

    last_visible_gt = max(frame["frame_index"] for frame in target_frames)
    termination_error = predicted_indexes[-1] - last_visible_gt if predicted_indexes else None

    return {
        "schema_version": 1,
        "series": series,
        "settings": {
            "top_k": top_k,
            "tolerance_pixels": tolerance,
            "flight_only": not all_states,
        },
        "counts": {
            "labeled_scope_frames": len(labeled_scope),
            "visible_or_blurred_gt_frames": target_count,
            "predicted_scope_frames": predicted_scope_frames,
            "correct_frames": correct_count,
            "invisible_frames_with_prediction": invisible_prediction_frames,
        },
        "metrics": {
            "recall_at_k": recall,
            "precision_by_frame": precision,
            "false_track_rate": 1.0 - precision if predicted_scope_frames else 0.0,
            "flight_coverage": recall,
            "median_center_error_pixels": statistics.median(center_errors) if center_errors else None,
            "p95_center_error_pixels": percentile(center_errors, 95),
            "longest_contiguous_correct_run": longest_contiguous_run(correct_indexes),
            "launch_frame_error": launch_error,
            "termination_frame_error": termination_error,
        },
    }


def format_optional(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_report(report: dict) -> None:
    counts = report["counts"]
    metrics = report["metrics"]
    settings = report["settings"]
    print(f"Series: {report['series']} (top-K={settings['top_k']}, tolerance={settings['tolerance_pixels']}px)")
    print(
        f"Scope: {counts['labeled_scope_frames']} labeled frames, "
        f"{counts['visible_or_blurred_gt_frames']} visible/blurred targets"
    )
    print(f"Recall@K / visible-flight coverage: {metrics['recall_at_k']:.3%}")
    print(f"Frame precision: {metrics['precision_by_frame']:.3%}")
    print(f"False-track rate: {metrics['false_track_rate']:.3%}")
    print(f"Median center error: {format_optional(metrics['median_center_error_pixels'])} px")
    print(f"95th-percentile center error: {format_optional(metrics['p95_center_error_pixels'])} px")
    print(f"Longest contiguous correct run: {metrics['longest_contiguous_correct_run']} frames")
    print(f"Launch-frame error: {metrics['launch_frame_error'] if metrics['launch_frame_error'] is not None else 'n/a'}")
    print(
        "Termination-frame error: "
        f"{metrics['termination_frame_error'] if metrics['termination_frame_error'] is not None else 'n/a'}"
    )


def main() -> None:
    args = parse_args()
    annotations = load_annotation_document(args.annotations)
    predictions, prediction_document = load_predictions(args.predictions, args.series)
    report = evaluate(
        annotations,
        predictions,
        prediction_document,
        args.top_k,
        args.tolerance,
        args.all_states,
        args.series,
    )
    print_report(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
