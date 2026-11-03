from pathlib import Path
import unittest


class ProductionInferenceBoundaryTests(unittest.TestCase):
    def test_production_call_graph_has_no_ground_truth_loader(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        tracker = Path(__file__).resolve().parents[1] / "tracker"
        production_files = [
            scripts / "run_shot_tracer.py",
            scripts / "detect_motion_candidates.py",
            scripts / "detect_tiled_candidates.py",
            scripts / "associate_candidates.py",
            scripts / "render_associated_track.py",
            tracker / "video_io.py",
        ]
        forbidden = ("load_annotation_document", "POINT_VISIBILITIES", "annotations.json")
        for path in production_files:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path.name} contains forbidden token {token}")


if __name__ == "__main__":
    unittest.main()
