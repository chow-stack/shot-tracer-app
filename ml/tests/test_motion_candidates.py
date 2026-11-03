import importlib.util
import math
from pathlib import Path
import random
import unittest

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detect_motion_candidates.py"
SPEC = importlib.util.spec_from_file_location("detect_motion_candidates", SCRIPT_PATH)
motion = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(motion)


class MotionCandidateTests(unittest.TestCase):
    def test_fast_pixel_primitives_preserve_inclusive_uint8_semantics(self):
        first = np.asarray(
            [[[0, 20, 255], [100, 90, 80]], [[5, 5, 5], [250, 0, 125]]],
            dtype=np.uint8,
        )
        second = np.asarray(
            [[[10, 10, 250], [95, 120, 70]], [[5, 7, 1], [0, 255, 125]]],
            dtype=np.uint8,
        )
        expected = np.max(cv2.absdiff(first, second), axis=2).astype(np.uint8)
        np.testing.assert_array_equal(motion.frame_change(first, second), expected)
        response = motion.frame_change(first, second)
        np.testing.assert_array_equal(
            cv2.compare(response, 10, cv2.CMP_GE),
            np.where(response >= 10, 255, 0).astype(np.uint8),
        )

    def test_fast_percentile_matches_numpy_linear_interpolation_exactly(self):
        generator = np.random.default_rng(17)
        for size in (*range(1, 40), 100, 255, 800):
            for _ in range(20):
                values = generator.integers(0, 256, size=size, dtype=np.uint8)
                self.assertEqual(
                    motion.percentile90_uint8(values),
                    float(np.percentile(values, 90)),
                )
        with self.assertRaises(ValueError):
            motion.percentile90_uint8(np.asarray([], dtype=np.uint8))

    def test_three_frame_response_isolates_middle_position(self):
        previous = np.zeros((40, 60, 3), dtype=np.uint8)
        current = previous.copy()
        following = previous.copy()
        previous[18:21, 8:11] = 240
        current[18:21, 28:31] = 240
        following[18:21, 48:51] = 240

        response = motion.three_frame_response(previous, current, following)

        self.assertEqual(int(response[19, 29]), 240)
        self.assertEqual(int(response[19, 9]), 0)
        self.assertEqual(int(response[19, 49]), 0)

    def test_component_center_and_multithreshold_merge(self):
        response = np.zeros((50, 70), dtype=np.uint8)
        response[20:24, 30:35] = 100
        candidates = motion.component_candidates(
            response,
            (10, 30),
            frame_index=7,
            source="test",
            source_bonus=0.0,
            min_area=2,
            max_area=100,
            max_dimension=20,
            max_aspect_ratio=10.0,
            maximum_y=None,
        )
        merged = motion.merge_nearby_candidates(candidates, radius=3.0)

        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["x"], 32.0)
        self.assertAlmostEqual(merged[0]["y"], 21.5)
        self.assertEqual(merged[0]["frame_index"], 7)

    def test_background_sampling_uses_only_configured_early_window(self):
        indexes = motion.choose_background_indexes(background_end=120, sample_count=15)
        self.assertEqual(len(indexes), 15)
        self.assertLess(max(indexes), 120)

    def test_cross_source_support_is_resorted(self):
        candidates = [
            {"x": 0.0, "y": 0.0, "confidence": 0.60, "source": "temporal"},
            {"x": 20.0, "y": 0.0, "confidence": 0.58, "source": "temporal"},
            {"x": 20.5, "y": 0.0, "confidence": 0.57, "source": "background"},
        ]
        merged = motion.merge_nearby_candidates(candidates, radius=2.0)
        self.assertEqual(merged[0]["x"], 20.0)
        self.assertAlmostEqual(merged[0]["confidence"], 0.63)

    def test_spatial_merge_matches_original_linear_scan(self):
        rng = random.Random(12)
        candidates = [
            {
                "x": rng.uniform(0, 300),
                "y": rng.uniform(0, 500),
                "confidence": rng.random(),
                "source": f"source-{index % 3}",
            }
            for index in range(300)
        ]

        expected = []
        for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            duplicate = next(
                (
                    existing
                    for existing in expected
                    if math.hypot(candidate["x"] - existing["x"], candidate["y"] - existing["y"])
                    <= 8.0
                ),
                None,
            )
            if duplicate is None:
                expected.append(candidate.copy())
            else:
                sources = set(duplicate.get("supporting_sources", [duplicate["source"]]))
                sources.add(candidate["source"])
                duplicate["supporting_sources"] = sorted(sources)
                duplicate["confidence"] = min(
                    1.0,
                    duplicate["confidence"]
                    + (0.05 if candidate["source"] != duplicate["source"] else 0.0),
                )
        expected.sort(key=lambda item: item["confidence"], reverse=True)

        actual = motion.merge_nearby_candidates([candidate.copy() for candidate in candidates], 8.0)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
