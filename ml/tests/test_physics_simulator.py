from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from physics.aerodynamics import AerodynamicConfig, GRAVITY_MPS2, accelerations  # noqa: E402
from physics.simulator import SimulationConfig, simulate_flight  # noqa: E402
from physics.state import FlightState  # noqa: E402


class PhysicsSimulatorTests(unittest.TestCase):
    def test_vacuum_agrees_with_analytic_parabola(self):
        state = FlightState((0, 0, 1), (25, 0, 18), (0, 0, 0))
        result = simulate_flight(state, SimulationConfig(
            timestep_seconds=0.01,
            maximum_seconds=1.0,
            aerodynamics=AerodynamicConfig(enable_drag=False, enable_lift=False, spin_decay_seconds=None),
        ))
        time = result.times_seconds[-1]
        expected = np.asarray((25 * time, 0, 1 + 18 * time - 0.5 * GRAVITY_MPS2 * time**2))
        np.testing.assert_allclose(result.positions_m[-1], expected, atol=1e-9)

    def test_drag_reduces_speed_and_mechanical_energy(self):
        initial = FlightState((0, 0, 1), (60, 0, 12), (0, 0, 0))
        drag = simulate_flight(initial, SimulationConfig(maximum_seconds=0.5))
        vacuum = simulate_flight(initial, SimulationConfig(
            maximum_seconds=0.5,
            aerodynamics=AerodynamicConfig(enable_drag=False, enable_lift=False),
        ))
        self.assertLess(np.linalg.norm(drag.velocities_mps[-1]), np.linalg.norm(vacuum.velocities_mps[-1]))
        drag_energy = 0.5 * np.dot(drag.velocities_mps[-1], drag.velocities_mps[-1]) + GRAVITY_MPS2 * drag.positions_m[-1, 2]
        initial_energy = 0.5 * np.dot(initial.velocity_mps, initial.velocity_mps) + GRAVITY_MPS2
        self.assertLess(drag_energy, initial_energy)

    def test_magnus_acceleration_is_perpendicular_to_relative_velocity(self):
        state = FlightState((0, 0, 1), (50, 2, 4), (0, 300, 40))
        full, _ = accelerations(state, AerodynamicConfig(enable_drag=False, enable_lift=True))
        gravity = np.asarray((0.0, 0.0, -GRAVITY_MPS2))
        lift = full - gravity
        self.assertGreater(np.linalg.norm(lift), 0)
        self.assertAlmostEqual(float(np.dot(lift, state.velocity_mps)), 0.0, places=9)

    def test_timestep_halving_converges(self):
        state = FlightState((0, 0, 0.01), (52, 0, 17), (0, 420, 25))
        coarse = simulate_flight(state, SimulationConfig(timestep_seconds=0.04, maximum_seconds=1.0))
        medium = simulate_flight(state, SimulationConfig(timestep_seconds=0.02, maximum_seconds=1.0))
        fine = simulate_flight(state, SimulationConfig(timestep_seconds=0.01, maximum_seconds=1.0))
        coarse_error = np.linalg.norm(coarse.positions_m[-1] - fine.positions_m[-1])
        medium_error = np.linalg.norm(medium.positions_m[-1] - fine.positions_m[-1])
        self.assertLess(medium_error, coarse_error)

    def test_simulation_is_finite_and_terminates_at_ground(self):
        state = FlightState((0, 0, 0.01), (55, 0, 16), (0, 400, 0))
        result = simulate_flight(state)
        self.assertTrue(result.terminated_at_ground)
        self.assertTrue(np.all(np.isfinite(result.positions_m)))
        self.assertAlmostEqual(result.positions_m[-1, 2], 0.0)
        self.assertGreater(result.flight_seconds, 1.0)

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "three-vector"):
            FlightState((0, 0), (1, 2, 3), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "timestep"):
            SimulationConfig(timestep_seconds=0)
        with self.assertRaisesRegex(ValueError, "below ground"):
            simulate_flight(FlightState((0, 0, -1), (1, 0, 1), (0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
