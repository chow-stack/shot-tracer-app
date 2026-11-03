from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from temporal.dataset import TemporalTileDataset, gaussian_heatmap  # noqa: E402
from temporal.model import TemporalHeatmapNet, temporal_loss  # noqa: E402
from scripts.detect_temporal_candidates import suppress_nearby_points  # noqa: E402


class HeatmapTargetTests(unittest.TestCase):
    def test_visible_target_peaks_at_center(self):
        heatmap = gaussian_heatmap(20, 10, 7, 4, 1.5)
        flat_index = int(torch.argmax(heatmap))
        y, x = divmod(flat_index, 20)
        self.assertEqual((x, y), (7, 4))
        self.assertAlmostEqual(float(heatmap[0, 4, 7]), 1.0)

    def test_invisible_target_is_zero(self):
        heatmap = gaussian_heatmap(20, 10, None, None, 1.5)
        self.assertEqual(float(heatmap.sum()), 0.0)

    def test_memory_mapped_cache_loads_three_frame_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            array = np.memmap(
                cache / "tiles.uint8", dtype=np.uint8, mode="w+", shape=(3, 8, 8, 3)
            )
            array[0] = 10
            array[1] = 20
            array[2] = 30
            array.flush()
            del array
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": "uint8_memmap",
                        "data_file": "tiles.uint8",
                        "input_size": 8,
                        "shape": [3, 8, 8, 3],
                        "index": {
                            "shot/000000/t00": 0,
                            "shot/000001/t00": 1,
                            "shot/000002/t00": 2,
                        },
                    }
                )
            )
            manifest = {
                "videos": [{"video_id": "shot", "video_path": "unused.mov"}],
                "samples": [
                    {
                        "split": "train",
                        "sample_id": "shot:1:t0",
                        "video_id": "shot",
                        "context_frame_indexes": [0, 1, 2],
                        "target_frame_index": 1,
                        "tile_index": 0,
                        "tile": [0, 0, 8, 8],
                        "target": {"visible": True, "x": 4.0, "y": 4.0},
                        "uncertain": False,
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = TemporalTileDataset(
                str(manifest_path), "train", input_size=8, output_stride=4, cache_dir=str(cache)
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (9, 8, 8))
            self.assertAlmostEqual(float(sample["image"][0, 0, 0]), 10 / 255)
            self.assertAlmostEqual(float(sample["image"][3, 0, 0]), 20 / 255)


class TemporalModelTests(unittest.TestCase):
    def test_output_shapes_and_finite_loss(self):
        model = TemporalHeatmapNet()
        images = torch.rand(2, 9, 64, 64)
        heatmap_logits, visibility_logits = model(images)
        self.assertEqual(tuple(heatmap_logits.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(visibility_logits.shape), (2,))
        targets = torch.zeros_like(heatmap_logits)
        target_visibility = torch.tensor([0.0, 1.0])
        loss, metrics = temporal_loss(
            heatmap_logits,
            visibility_logits,
            targets,
            target_visibility,
            torch.ones(2),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["total_loss"], 0)

    def test_nearby_candidate_suppression_keeps_best_point(self):
        best = {"x": 100.0, "y": 100.0, "confidence": 0.9}
        duplicate = {"x": 104.0, "y": 103.0, "confidence": 0.7}
        separate = {"x": 140.0, "y": 100.0, "confidence": 0.5}
        self.assertEqual(
            suppress_nearby_points([duplicate, separate, best], 12.0),
            [best, separate],
        )


if __name__ == "__main__":
    unittest.main()
