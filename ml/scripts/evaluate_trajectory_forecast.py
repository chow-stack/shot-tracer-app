"""Evaluate the two fixed trajectory baselines on withheld visible tails."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.annotation_io import POINT_VISIBILITIES, load_annotation_document  # noqa: E402
from tracker.trajectory_forecast import MODEL_NAMES, fit_model, predict  # noqa: E402
from tracker.video_io import atomic_write_json  # noqa: E402


WITHHELD_FRACTIONS = (0.10, 0.20, 0.30)


def annotation_points(document: dict) -> list[dict]:
    return [
        {
            "frame_index": frame["frame_index"],
            "timestamp_seconds": float(frame["timestamp_seconds"]),
            "x": float(frame["x"]),
            "y": float(frame["y"]),
        }
        for frame in document["frames"]
        if frame["shot_state"] == "flight" and frame["visibility"] in POINT_VISIBILITIES
    ]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np_percentile(values, 100.0 * fraction))


def np_percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_points(points: list[dict], clip_name: str) -> list[dict]:
    if len(points) < 12:
        raise ValueError(f"{clip_name} needs at least 12 visible/blurred points")
    cases: list[dict] = []
    for fraction in WITHHELD_FRACTIONS:
        withheld_count = max(1, round(len(points) * fraction))
        train, target = points[:-withheld_count], points[-withheld_count:]
        if len(train) < 8:
            continue
        target_times = [point["timestamp_seconds"] for point in target]
        for model_name in MODEL_NAMES:
            model = fit_model(train, model_name)
            predicted = predict(model, target_times)
            errors = [
                math.hypot(prediction["x"] - truth["x"], prediction["y"] - truth["y"])
                for prediction, truth in zip(predicted, target)
            ]
            cases.append({
                "clip": clip_name,
                "model": model_name,
                "withheld_fraction": fraction,
                "training_points": len(train),
                "withheld_points": len(target),
                "max_horizon_seconds": target_times[-1] - train[-1]["timestamp_seconds"],
                "median_error_px": statistics.median(errors),
                "p95_error_px": np_percentile(errors, 95.0),
                "final_error_px": errors[-1],
            })
    return cases


def build_report(annotation_paths: list[Path]) -> dict:
    cases: list[dict] = []
    for path in annotation_paths:
        document = load_annotation_document(path)
        cases.extend(evaluate_points(annotation_points(document), path.stem.replace(".annotations", "")))
    summaries: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        model_cases = [case for case in cases if case["model"] == model_name]
        medians = [case["median_error_px"] for case in model_cases]
        finals = [case["final_error_px"] for case in model_cases]
        summaries[model_name] = {
            "case_count": len(model_cases),
            "median_of_case_median_error_px": statistics.median(medians),
            "p95_all_case_final_error_px": np_percentile(finals, 95.0),
        }
    winner = min(MODEL_NAMES, key=lambda name: summaries[name]["median_of_case_median_error_px"])
    return {
        "schema_version": 1,
        "evaluation": "withheld_visible_tail",
        "withheld_fractions": list(WITHHELD_FRACTIONS),
        "cases": cases,
        "summary": summaries,
        "selected_baseline": winner,
        "limitations": "This evaluates visible withheld points, not the true invisible landing.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.annotations)
    for name, summary in report["summary"].items():
        print(
            f"{name}: median(case medians)={summary['median_of_case_median_error_px']:.2f}px, "
            f"p95(final errors)={summary['p95_all_case_final_error_px']:.2f}px"
        )
    print(f"Selected baseline: {report['selected_baseline']}")
    if args.output:
        atomic_write_json(args.output, report)
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
