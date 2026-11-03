from __future__ import annotations

import sys
import unittest
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from temporal.manifest import (  # noqa: E402
    context_indexes,
    deterministic_tile_indexes,
    point_in_rectangle,
    tile_rectangles,
    validate_no_video_leakage,
)


class TemporalGeometryTests(unittest.TestCase):
    def test_context_clamps_only_at_video_edges(self):
        self.assertEqual(context_indexes(0, 10, 1), [0, 0, 1])
        self.assertEqual(context_indexes(5, 10, 1), [4, 5, 6])
        self.assertEqual(context_indexes(9, 10, 1), [8, 9, 9])

    def test_training_tiles_match_full_frame_inference_grid(self):
        rectangles = tile_rectangles(1080, 1920, 640, 0.25)
        self.assertEqual(len(rectangles), 8)
        containing = [rect for rect in rectangles if point_in_rectangle(600, 1500, rect)]
        self.assertGreaterEqual(len(containing), 1)

    def test_negative_tile_selection_is_repeatable_and_unique(self):
        first = deterministic_tile_indexes("shot", 42, 5, 8)
        second = deterministic_tile_indexes("shot", 42, 5, 8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 5)


class SplitLeakageTests(unittest.TestCase):
    def test_accepts_whole_video_split(self):
        manifest = {
            "fold": {
                "train_video_ids": ["a", "b"],
                "validation_video_ids": ["c"],
            },
            "samples": [
                {"sample_id": "a:1", "video_id": "a", "split": "train"},
                {"sample_id": "c:1", "video_id": "c", "split": "validation"},
            ],
        }
        validate_no_video_leakage(manifest)

    def test_rejects_video_in_both_splits(self):
        manifest = {
            "fold": {
                "train_video_ids": ["a", "c"],
                "validation_video_ids": ["c"],
            },
            "samples": [],
        }
        with self.assertRaisesRegex(ValueError, "both splits"):
            validate_no_video_leakage(manifest)

    def test_rejects_mislabeled_sample(self):
        manifest = {
            "fold": {
                "train_video_ids": ["a"],
                "validation_video_ids": ["c"],
            },
            "samples": [
                {"sample_id": "c:1", "video_id": "c", "split": "train"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "violates"):
            validate_no_video_leakage(manifest)


if __name__ == "__main__":
    unittest.main()

