from __future__ import annotations

import sys
import unittest
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "scripts"))

from detect_tiled_candidates import (  # noqa: E402
    axis_starts,
    box_iou,
    merge_cross_tile_duplicates,
    tile_rectangles,
)


class TileCoverageTests(unittest.TestCase):
    def test_portrait_frame_is_fully_covered(self):
        rectangles = tile_rectangles(1080, 1920, 640, 0.25)
        self.assertEqual(len(rectangles), 8)
        self.assertEqual(rectangles[0], (0, 0, 640, 640))
        self.assertEqual(rectangles[-1], (440, 1280, 1080, 1920))
        self.assertEqual(axis_starts(1080, 640, 0.25), [0, 440])
        self.assertEqual(axis_starts(1920, 640, 0.25), [0, 480, 960, 1280])

    def test_small_dimension_uses_one_tile(self):
        self.assertEqual(tile_rectangles(320, 200, 640, 0.25), [(0, 0, 320, 200)])


class CrossTileNmsTests(unittest.TestCase):
    def test_duplicate_boxes_keep_higher_confidence(self):
        low = {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 20.0, "confidence": 0.4}
        high = {"x1": 11.0, "y1": 11.0, "x2": 21.0, "y2": 21.0, "confidence": 0.9}
        separate = {"x1": 40.0, "y1": 40.0, "x2": 50.0, "y2": 50.0, "confidence": 0.5}
        merged = merge_cross_tile_duplicates([low, high, separate], 0.3)
        self.assertEqual(merged, [high, separate])
        self.assertGreater(box_iou(low, high), 0.3)


if __name__ == "__main__":
    unittest.main()
