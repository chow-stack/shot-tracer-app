"""Numerical integration for the complete coupled flight state."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .state import FlightState


Derivative = Callable[[FlightState], np.ndarray]


def rk4_step(state: FlightState, timestep_seconds: float, derivative: Derivative) -> FlightState:
    if not math.isfinite(timestep_seconds) or timestep_seconds <= 0:
        raise ValueError("timestep_seconds must be finite and positive")
    vector = state.as_vector()

    def evaluate(candidate: np.ndarray) -> np.ndarray:
        result = np.asarray(derivative(FlightState.from_vector(candidate)), dtype=np.float64)
        if result.shape != vector.shape or not np.all(np.isfinite(result)):
            raise ValueError("state derivative must contain nine finite values")
        return result

    k1 = evaluate(vector)
    k2 = evaluate(vector + 0.5 * timestep_seconds * k1)
    k3 = evaluate(vector + 0.5 * timestep_seconds * k2)
    k4 = evaluate(vector + timestep_seconds * k3)
    advanced = vector + timestep_seconds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return FlightState.from_vector(advanced)
