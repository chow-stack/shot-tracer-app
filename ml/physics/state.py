"""Validated SI-unit state for a golf-ball flight.

Coordinates are ``(downrange, lateral_right, up)`` in metres. Velocity is in
metres/second and spin is an angular-velocity vector in radians/second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_VECTOR_SIZE = 9


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector.copy()


@dataclass(frozen=True)
class FlightState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    spin_radps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vector3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", _vector3(self.velocity_mps, "velocity_mps"))
        object.__setattr__(self, "spin_radps", _vector3(self.spin_radps, "spin_radps"))

    def as_vector(self) -> np.ndarray:
        return np.concatenate((self.position_m, self.velocity_mps, self.spin_radps))

    @classmethod
    def from_vector(cls, value) -> "FlightState":
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (STATE_VECTOR_SIZE,) or not np.all(np.isfinite(vector)):
            raise ValueError("state vector must contain nine finite values")
        return cls(vector[:3], vector[3:6], vector[6:9])
