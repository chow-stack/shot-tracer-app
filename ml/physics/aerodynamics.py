"""Small, bounded golf-ball aerodynamic model.

The coefficient envelopes are intentionally conservative shape priors based on
the measured ranges reported by Bearman & Harvey (1976) and Smith et al. (2018).
They are not fitted from video and must not be reported as measured launch data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .state import FlightState


AERODYNAMIC_MODEL_VERSION = "golf-flight-envelope-v1"
AERODYNAMIC_SOURCES = (
    "https://doi.org/10.1017/S0001925900007617",
    "https://doi.org/10.3390/proceedings2060238",
)
GRAVITY_MPS2 = 9.80665
STANDARD_AIR_DENSITY_KG_M3 = 1.225
GOLF_BALL_MASS_KG = 0.04593
GOLF_BALL_DIAMETER_M = 0.04267
GOLF_BALL_RADIUS_M = 0.5 * GOLF_BALL_DIAMETER_M
GOLF_BALL_AREA_M2 = math.pi * GOLF_BALL_RADIUS_M**2
CD_BOUNDS = (0.18, 0.34)
CL_BOUNDS = (0.0, 0.34)


def aerodynamic_model_metadata() -> dict:
    return {
        "version": AERODYNAMIC_MODEL_VERSION,
        "sources": list(AERODYNAMIC_SOURCES),
        "drag_coefficient_bounds": list(CD_BOUNDS),
        "lift_coefficient_bounds": list(CL_BOUNDS),
        "ball_mass_kg": GOLF_BALL_MASS_KG,
        "ball_diameter_m": GOLF_BALL_DIAMETER_M,
        "standard_air_density_kg_m3": STANDARD_AIR_DENSITY_KG_M3,
    }


def drag_coefficient(speed_mps: float) -> float:
    """Return a smooth coefficient constrained to the documented envelope."""
    if not math.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError("speed_mps must be finite and non-negative")
    value = 0.22 + 0.07 * math.exp(-speed_mps / 28.0)
    return float(np.clip(value, *CD_BOUNDS))


def lift_coefficient(speed_mps: float, spin_radps: np.ndarray) -> float:
    """Return bounded lift from the nondimensional spin ratio ``r*omega/v``."""
    if not math.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError("speed_mps must be finite and non-negative")
    spin = np.asarray(spin_radps, dtype=np.float64)
    if spin.shape != (3,) or not np.all(np.isfinite(spin)):
        raise ValueError("spin_radps must be a finite three-vector")
    if speed_mps <= 1e-9:
        return 0.0
    spin_ratio = GOLF_BALL_RADIUS_M * float(np.linalg.norm(spin)) / speed_mps
    return float(np.clip(1.45 * spin_ratio, *CL_BOUNDS))


@dataclass(frozen=True)
class AerodynamicConfig:
    air_density_kg_m3: float = STANDARD_AIR_DENSITY_KG_M3
    wind_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_drag: bool = True
    enable_lift: bool = True
    spin_decay_seconds: float | None = 18.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.air_density_kg_m3) or self.air_density_kg_m3 <= 0:
            raise ValueError("air_density_kg_m3 must be finite and positive")
        wind = np.asarray(self.wind_mps, dtype=np.float64)
        if wind.shape != (3,) or not np.all(np.isfinite(wind)):
            raise ValueError("wind_mps must be a finite three-vector")
        if self.spin_decay_seconds is not None and (
            not math.isfinite(self.spin_decay_seconds) or self.spin_decay_seconds <= 0
        ):
            raise ValueError("spin_decay_seconds must be positive or None")


def accelerations(state: FlightState, config: AerodynamicConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return translational acceleration and spin derivative for ``state``."""
    relative_velocity = state.velocity_mps - np.asarray(config.wind_mps, dtype=np.float64)
    speed = float(np.linalg.norm(relative_velocity))
    acceleration = np.asarray((0.0, 0.0, -GRAVITY_MPS2), dtype=np.float64)
    if speed > 1e-9:
        pressure_area_over_mass = (
            0.5 * config.air_density_kg_m3 * GOLF_BALL_AREA_M2 / GOLF_BALL_MASS_KG
        )
        if config.enable_drag:
            acceleration += (
                -pressure_area_over_mass
                * drag_coefficient(speed)
                * speed
                * relative_velocity
            )
        spin_magnitude = float(np.linalg.norm(state.spin_radps))
        if config.enable_lift and spin_magnitude > 1e-9:
            lift_direction = np.cross(relative_velocity, state.spin_radps)
            direction_norm = float(np.linalg.norm(lift_direction))
            if direction_norm > 1e-12:
                acceleration += (
                    pressure_area_over_mass
                    * lift_coefficient(speed, state.spin_radps)
                    * speed**2
                    * lift_direction
                    / direction_norm
                )
    spin_derivative = (
        np.zeros(3, dtype=np.float64)
        if config.spin_decay_seconds is None
        else -state.spin_radps / config.spin_decay_seconds
    )
    return acceleration, spin_derivative


def state_derivative(state: FlightState, config: AerodynamicConfig) -> np.ndarray:
    acceleration, spin_derivative = accelerations(state, config)
    return np.concatenate((state.velocity_mps, acceleration, spin_derivative))
