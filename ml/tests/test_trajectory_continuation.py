from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "scripts"))

from continue_trajectory import continue_document  # noqa: E402
from evaluate_trajectory_forecast import evaluate_points  # noqa: E402
from tracker.trajectory_forecast import MODEL_NAMES, fit_model, predict  # noqa: E402
from tracker.trajectory_io import upgrade_trajectory_document  # noqa: E402
from physics.templates import TEMPLATE_BANK_VERSION, template_bank, template_specs  # noqa: E402


def legacy_document(count: int = 20) -> dict:
    timestamps = [index * 0.1 for index in range(40)]
    trajectory = []
    for index in range(count):
        trajectory.append({
            "frame_index": index,
            "x": 10.0 + 3.0 * index,
            "y": 100.0 - 5.0 * index + 0.2 * index * index,
            "estimated": index == 5,
        })
    return {
        "schema_version": 1,
        "video": {"width": 200, "height": 200, "fps": 10.0, "frame_count": 40, "timestamps_seconds": timestamps},
        "trajectory": trajectory,
    }


class TrajectorySchemaTests(unittest.TestCase):
    def test_v1_upgrade_preserves_flat_points_and_marks_provenance(self):
        source = legacy_document()
        upgraded = upgrade_trajectory_document(source)
        self.assertEqual(upgraded["trajectory_schema_version"], 2)
        self.assertEqual(len(upgraded["trajectory"]), len(source["trajectory"]))
        self.assertEqual(upgraded["trajectory"][5]["provenance"], "estimated_gap")
        self.assertEqual(upgraded["observation_end_frame"], 19)
        self.assertNotIn("provenance", source["trajectory"][0])

    def test_observed_only_does_not_change_coordinates(self):
        source = legacy_document()
        output = continue_document(source, "observed_only", 1.2)
        original = [(point["frame_index"], point["x"], point["y"]) for point in source["trajectory"]]
        upgraded = [(point["frame_index"], point["x"], point["y"]) for point in output["trajectory"]]
        self.assertEqual(original, upgraded)


class ForecastTests(unittest.TestCase):
    def test_template_bank_is_small_fixed_and_complete(self):
        specs = template_specs()
        self.assertEqual(len(specs), 18)
        self.assertEqual({spec.profile for spec in specs}, {"wood", "iron", "wedge"})
        self.assertEqual({spec.shape for spec in specs}, {"straight", "draw", "fade"})
        self.assertTrue(all(value.simulation.terminated_at_ground for value in template_bank()))

    def test_models_predict_finite_future_points(self):
        points = upgrade_trajectory_document(legacy_document())["observations"]
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = fit_model(points, name)
                prediction = predict(model, [2.0, 2.1])
                self.assertEqual(len(prediction), 2)
                self.assertGreater(prediction[1]["uncertainty_px"], prediction[0]["uncertainty_px"])

    def test_automatic_continuation_is_separate_and_inferred(self):
        output = continue_document(legacy_document(), "automatic_cinematic", 2.0)
        inferred = [point for point in output["trajectory"] if point["provenance"] == "inferred"]
        self.assertGreater(len(inferred), 0)
        self.assertEqual(output["observation_end_frame"], 19)
        self.assertTrue(all(point["frame_index"] > 19 for point in inferred))
        endpoint = output["continuation"]["endpoint"]
        self.assertFalse(endpoint["is_measured_landing"])
        self.assertGreaterEqual(endpoint["x"], 0.0)
        self.assertLessEqual(endpoint["x"], 200.0)
        self.assertGreaterEqual(endpoint["y"], 0.0)
        self.assertLessEqual(endpoint["y"], 200.0)

    def test_cinematic_completion_does_not_cross_center_without_observed_crossing(self):
        document = legacy_document()
        for point in document["trajectory"]:
            point["x"] += 80.0
        output = continue_document(document, "automatic_cinematic", 2.0)
        self.assertGreaterEqual(output["continuation"]["endpoint"]["x"], 112.0)

    def test_rising_track_keeps_rising_before_later_apex(self):
        document = legacy_document(20)
        for index, point in enumerate(document["trajectory"]):
            point["y"] = 140.0 - 2.0 * index
        output = continue_document(document, "automatic_cinematic", 2.0)
        inferred = [point for point in output["trajectory"] if point["provenance"] == "inferred"]
        observed_y = output["observations"][-1]["y"]
        early = inferred[:3]
        self.assertTrue(all(point["y"] < observed_y for point in early))
        apex_time = output["continuation"]["apex"]["seconds_after_observation"]
        self.assertGreaterEqual(apex_time, 0.65)

    def test_assisted_continuation_reaches_user_endpoint(self):
        output = continue_document(legacy_document(), "assisted_landing", 0.5, (150.0, 180.0))
        last = output["trajectory"][-1]
        self.assertAlmostEqual(last["x"], 150.0)
        self.assertAlmostEqual(last["y"], 180.0)
        self.assertEqual(output["continuation"]["endpoint"]["source"], "user_tap")

    def test_physics_template_writes_reproducible_inferred_metadata(self):
        template = template_bank("iron")[1]
        sample_times = [index * 0.1 for index in range(20)]
        positions = []
        for timestamp in sample_times:
            positions.append([
                float(__import__("numpy").interp(timestamp, template.simulation.times_seconds, template.simulation.positions_m[:, axis]))
                for axis in range(3)
            ])
        denominator = [15.0 + position[0] for position in positions]
        horizontal = [(position[1] / depth, position[0] / depth) for position, depth in zip(positions, denominator)]
        vertical = [(position[2] / depth, position[0] / depth) for position, depth in zip(positions, denominator)]
        anchor_h, anchor_v = horizontal[-1], vertical[-1]
        trajectory = []
        for index, (h, v) in enumerate(zip(horizontal, vertical)):
            trajectory.append({
                "frame_index": index,
                "timestamp_seconds": sample_times[index],
                "x": 220.0 + 350.0 * (h[0] - anchor_h[0]) + 80.0 * (h[1] - anchor_h[1]),
                "y": 110.0 - 650.0 * (v[0] - anchor_v[0]) + 500.0 * (v[1] - anchor_v[1]),
                "estimated": False,
            })
        address = (trajectory[0]["x"], trajectory[0]["y"])
        timestamps = [index * 0.1 for index in range(100)]
        source = {
            "schema_version": 1,
            "video": {"width": 400, "height": 400, "fps": 10.0, "frame_count": 100, "timestamps_seconds": timestamps},
            "initialization": {"address_x": address[0], "address_y": address[1]},
            "trajectory": trajectory,
        }
        output = continue_document(source, "physics_template", 6.0, shot_profile="iron")
        inferred = [point for point in output["trajectory"] if point["provenance"] == "inferred"]
        self.assertGreater(len(inferred), 1)
        self.assertAlmostEqual(inferred[0]["timestamp_seconds"], 2.0)
        self.assertTrue(all(a["uncertainty_px"] < b["uncertainty_px"] for a, b in zip(inferred, inferred[1:])))
        continuation = output["continuation"]
        self.assertEqual(continuation["mode"], "physics_template")
        self.assertEqual(continuation["model"]["template_bank_version"], TEMPLATE_BANK_VERSION)
        self.assertEqual(continuation["shot_profile"]["value"], "iron")
        self.assertFalse(continuation["endpoint"]["is_measured_landing"])
        self.assertEqual(output["observation_end_frame"], 19)
        round_tripped = upgrade_trajectory_document(output)
        self.assertEqual(round_tripped["continuation"], continuation)
        self.assertEqual(
            [(point["timestamp_seconds"], point["x"], point["y"], point["provenance"]) for point in round_tripped["trajectory"]],
            [(point["timestamp_seconds"], point["x"], point["y"], point["provenance"]) for point in output["trajectory"]],
        )

    def test_withheld_evaluator_runs_only_fixed_models(self):
        points = [
            {"frame_index": i, "timestamp_seconds": i * 0.05, "x": 20 + i, "y": 200 - 3 * i + 0.1 * i * i}
            for i in range(30)
        ]
        cases = evaluate_points(points, "synthetic")
        self.assertEqual({case["model"] for case in cases}, set(MODEL_NAMES))
        self.assertEqual({case["withheld_fraction"] for case in cases}, {0.1, 0.2, 0.3})


if __name__ == "__main__":
    unittest.main()
