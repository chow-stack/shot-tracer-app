"""Deterministic forward simulator that stops at first ground intersection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aerodynamics import AerodynamicConfig, state_derivative
from .integrator import rk4_step
from .state import FlightState


@dataclass(frozen=True)
class SimulationConfig:
    timestep_seconds: float = 0.01
    maximum_seconds: float = 15.0
    aerodynamics: AerodynamicConfig = AerodynamicConfig()

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestep_seconds) or self.timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be finite and positive")
        if self.timestep_seconds > 0.1:
            raise ValueError("timestep_seconds must be at most 0.1 seconds")
        if not math.isfinite(self.maximum_seconds) or self.maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be finite and positive")


@dataclass(frozen=True)
class SimulationResult:
    times_seconds: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    spins_radps: np.ndarray
    terminated_at_ground: bool

    @property
    def flight_seconds(self) -> float:
        return float(self.times_seconds[-1])


def _interpolate_ground(previous: FlightState, current: FlightState, fraction: float) -> FlightState:
    vector = previous.as_vector() + fraction * (current.as_vector() - previous.as_vector())
    vector[2] = 0.0
    return FlightState.from_vector(vector)


def simulate_flight(initial_state: FlightState, config: SimulationConfig = SimulationConfig()) -> SimulationResult:
    if initial_state.position_m[2] < 0:
        raise ValueError("initial position cannot be below ground")
    states = [initial_state]
    times = [0.0]
    state = initial_state
    terminated = False
    maximum_steps = math.ceil(config.maximum_seconds / config.timestep_seconds)
    derivative = lambda value: state_derivative(value, config.aerodynamics)
    for _ in range(maximum_steps):
        advanced = rk4_step(state, config.timestep_seconds, derivative)
        next_time = times[-1] + config.timestep_seconds
        crossed_ground = state.position_m[2] > 0 and advanced.position_m[2] <= 0
        if crossed_ground:
            denominator = state.position_m[2] - advanced.position_m[2]
            fraction = float(state.position_m[2] / denominator) if denominator > 0 else 1.0
            states.append(_interpolate_ground(state, advanced, fraction))
            times.append(times[-1] + fraction * config.timestep_seconds)
            terminated = True
            break
        states.append(advanced)
        times.append(next_time)
        state = advanced
        if not np.all(np.isfinite(advanced.as_vector())):
            raise FloatingPointError("simulation produced non-finite state")
    matrix = np.asarray([value.as_vector() for value in states], dtype=np.float64)
    return SimulationResult(
        times_seconds=np.asarray(times, dtype=np.float64),
        positions_m=matrix[:, :3],
        velocities_mps=matrix[:, 3:6],
        spins_radps=matrix[:, 6:9],
        terminated_at_ground=terminated,
    )
