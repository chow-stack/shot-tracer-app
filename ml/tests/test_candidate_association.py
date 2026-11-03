import importlib.util
from pathlib import Path
import random
import sys
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "associate_candidates.py"
SPEC = importlib.util.spec_from_file_location("associate_candidates", SCRIPT_PATH)
association = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = association
SPEC.loader.exec_module(association)


def candidate(frame, x, y, confidence=0.7):
    return association.Candidate(
        frame_index=frame,
        x=float(x),
        y=float(y),
        confidence=confidence,
        sources=("three_frame_motion",),
        source_ranks={"three_frame_motion": 20},
        raw={},
    )


class CandidateAssociationTests(unittest.TestCase):
    def test_bounded_gap_is_interpolated_and_marked_estimated(self):
        first = association.BeamState(1.0, "active", 1, candidate(1, 10, 20), None, ((1, 10, 20),), 0, 1)
        gap = association.BeamState(0.5, "gap", 2, None, first, first.observations, 1, 1)
        last_candidate = candidate(3, 14, 28)
        last = association.BeamState(1.5, "active", 3, last_candidate, gap, ((1, 10, 20), (3, 14, 28)), 0, 2)

        points = association.trajectory_from_states(
            [first, gap, last], {1: 0.0, 2: 0.1, 3: 0.2}
        )

        self.assertEqual([point["frame_index"] for point in points], [1, 2, 3])
        self.assertEqual(points[1]["x"], 12.0)
        self.assertEqual(points[1]["y"], 24.0)
        self.assertTrue(points[1]["estimated"])

    def test_beam_selects_launch_and_terminates_after_track(self):
        rng = random.Random(4)
        frames = list(range(10, 30))
        by_frame = {}
        true_points = {}
        x, y, dx, dy = 100.0, 500.0, 8.0, -100.0
        for frame in frames[:10]:
            true_points[frame] = (x, y)
            clutter = [candidate(frame, rng.uniform(0, 800), rng.uniform(0, 600), 0.85) for _ in range(30)]
            by_frame[frame] = clutter + [candidate(frame, x, y, 0.55)]
            x += dx
            y += dy
            dx *= 0.8
            dy *= 0.72
        for frame in frames[10:]:
            by_frame[frame] = [candidate(frame, rng.uniform(0, 800), rng.uniform(0, 600), 0.9) for _ in range(30)]

        final = association.associate(
            frames,
            by_frame,
            address=(100.0, 515.0),
            beam_width=200,
            max_gap=1,
            max_launch_radius=100.0,
        )
        selected = [state for state in association.unwind(final) if state.candidate is not None]

        self.assertGreaterEqual(len(selected), 8)
        self.assertLessEqual(selected[-1].frame_index, 21)
        for state in selected[:8]:
            expected = true_points[state.frame_index]
            self.assertAlmostEqual(state.candidate.x, expected[0])
            self.assertAlmostEqual(state.candidate.y, expected[1])

    def test_full_clip_search_waits_for_plausible_launch(self):
        rng = random.Random(8)
        frames = list(range(1, 36))
        by_frame = {
            frame: [candidate(frame, rng.uniform(300, 800), rng.uniform(50, 450), 0.9) for _ in range(12)]
            for frame in frames
        }
        x, y, dx, dy = 100.0, 500.0, 7.0, -90.0
        true_points = {}
        for frame in range(14, 24):
            true_points[frame] = (x, y)
            by_frame[frame].append(candidate(frame, x, y, 0.55))
            x += dx
            y += dy
            dx *= 0.82
            dy *= 0.72

        for offset in ((0.0, 0.0), (-8.0, 6.0), (9.0, -7.0)):
            with self.subTest(tap_offset=offset):
                final = association.associate_full_clip(
                    frames,
                    by_frame,
                    address=(100.0 + offset[0], 515.0 + offset[1]),
                    beam_width=200,
                    max_gap=1,
                    max_launch_radius=100.0,
                )
                selected = [
                    state for state in association.unwind(final) if state.candidate is not None
                ]

                self.assertEqual(selected[0].frame_index, 14)
                self.assertGreaterEqual(len(selected), 8)
                for state in selected[:8]:
                    self.assertEqual(
                        (state.candidate.x, state.candidate.y), true_points[state.frame_index]
                    )


if __name__ == "__main__":
    unittest.main()
