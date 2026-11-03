from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "scripts"))

from annotate_video import zoom_crop  # noqa: E402
from evaluate_tracking import evaluate, load_predictions  # noqa: E402
from tracker.annotation_io import (  # noqa: E402
    atomic_write_json,
    create_annotation_document,
    load_annotation_document,
    validate_annotation_document,
)


class AnnotationIOTests(unittest.TestCase):
    def test_zoom_crop_clamps_to_frame_edges(self):
        import numpy as np

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop, x1, y1 = zoom_crop(frame, (195.0, 5.0), 60)
        self.assertEqual(crop.shape, (60, 60, 3))
        self.assertEqual((x1, y1), (140, 0))

    def test_create_write_and_load_dense_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "clip.mov"
            video.write_bytes(b"synthetic-video-placeholder")
            document = create_annotation_document(str(video), 1920, 1080, 60.0, 4)
            self.assertEqual(len(document["frames"]), 4)
            self.assertEqual(document["frames"][3]["timestamp_seconds"], 0.05)
            self.assertEqual(validate_annotation_document(document), [])

            output = root / "annotations.json"
            atomic_write_json(output, document)
            loaded = load_annotation_document(output)
            self.assertEqual(loaded["video"]["sha256"], document["video"]["sha256"])

    def test_visible_frame_requires_point(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = Path(temporary_directory) / "clip.mov"
            video.write_bytes(b"video")
            document = create_annotation_document(str(video), 100, 100, 30.0, 1)
            document["frames"][0]["visibility"] = "visible"
            errors = validate_annotation_document(document)
            self.assertTrue(any("requires x/y" in error for error in errors))


class EvaluationTests(unittest.TestCase):
    def test_observations_series_can_exclude_estimated_trajectory_points(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "trajectory": [
                            {"frame_index": 1, "x": 10.0, "y": 10.0, "estimated": False},
                            {"frame_index": 2, "x": 20.0, "y": 20.0, "estimated": True},
                        ],
                        "observations": [
                            {"frame_index": 1, "x": 10.0, "y": 10.0, "estimated": False}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            observations, _ = load_predictions(str(path), "observations")
            self.assertEqual(set(observations), {1})

    def test_flight_metrics_include_misses_and_invisible_false_tracks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "clip.mov"
            video.write_bytes(b"video")
            annotations = create_annotation_document(str(video), 100, 100, 10.0, 5)
            frames = annotations["frames"]
            frames[1].update(x=10.0, y=10.0, visibility="visible", shot_state="flight")
            frames[2].update(x=20.0, y=20.0, visibility="blurred", shot_state="flight")
            frames[3].update(visibility="not_visible", shot_state="flight")

            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "launch_frame": 1,
                        "trajectory": [
                            {"frame_index": 1, "x": 10.0, "y": 10.0},
                            {"frame_index": 2, "x": 50.0, "y": 20.0},
                            {"frame_index": 3, "x": 60.0, "y": 20.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            predictions, prediction_document = load_predictions(
                str(predictions_path), "trajectory"
            )
            report = evaluate(
                annotations,
                predictions,
                prediction_document,
                top_k=1,
                tolerance=20.0,
                all_states=False,
                series="trajectory",
            )

            self.assertEqual(report["counts"]["correct_frames"], 1)
            self.assertEqual(report["metrics"]["recall_at_k"], 0.5)
            self.assertAlmostEqual(report["metrics"]["precision_by_frame"], 1 / 3)
            self.assertAlmostEqual(report["metrics"]["false_track_rate"], 2 / 3)
            self.assertEqual(report["metrics"]["median_center_error_pixels"], 15.0)
            self.assertEqual(report["metrics"]["longest_contiguous_correct_run"], 1)
            self.assertEqual(report["metrics"]["launch_frame_error"], 0)
            self.assertEqual(report["metrics"]["termination_frame_error"], 1)


if __name__ == "__main__":
    unittest.main()
