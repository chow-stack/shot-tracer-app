"""Frozen, deliberately small bank of club-conditioned flight templates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .aerodynamics import AERODYNAMIC_MODEL_VERSION
from .simulator import SimulationConfig, SimulationResult, simulate_flight
from .state import FlightState


TEMPLATE_BANK_VERSION = "club-profile-bank-v1"


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    profile: str
    shape: str
    launch_speed_mps: float
    launch_angle_degrees: float
    backspin_rpm: float
    sidespin_rpm: float


@dataclass(frozen=True)
class TemplateFlight:
    spec: TemplateSpec
    simulation: SimulationResult


# These are fixed cinematic priors, not inferred launch values. Two bounded
# variants per club profile prevent one nominal trajectory from dominating the
# transfer gate while keeping the complete bank small (18 flights).
_PROFILE_VARIANTS = {
    "wood": ((58.0, 13.0, 3000.0), (66.0, 11.0, 2500.0)),
    "iron": ((42.0, 22.0, 5600.0), (49.0, 18.0, 4700.0)),
    "wedge": ((27.0, 34.0, 8500.0), (33.0, 29.0, 7500.0)),
}
_SHAPE_SIDESPIN_RPM = {"draw": -350.0, "straight": 0.0, "fade": 350.0}


def template_specs(profile: str | None = None) -> tuple[TemplateSpec, ...]:
    if profile is not None and profile not in _PROFILE_VARIANTS:
        raise ValueError(f"Unknown shot profile: {profile}")
    profiles = (profile,) if profile is not None else tuple(_PROFILE_VARIANTS)
    values: list[TemplateSpec] = []
    for profile_name in profiles:
        for variant, (speed, angle, backspin) in enumerate(_PROFILE_VARIANTS[profile_name], start=1):
            for shape, sidespin in _SHAPE_SIDESPIN_RPM.items():
                values.append(TemplateSpec(
                    template_id=f"{TEMPLATE_BANK_VERSION}:{profile_name}:{variant}:{shape}",
                    profile=profile_name,
                    shape=shape,
                    launch_speed_mps=speed,
                    launch_angle_degrees=angle,
                    backspin_rpm=backspin,
                    sidespin_rpm=sidespin,
                ))
    return tuple(values)


def _simulate_template(spec: TemplateSpec) -> TemplateFlight:
    angle = math.radians(spec.launch_angle_degrees)
    rpm_to_radps = 2.0 * math.pi / 60.0
    initial = FlightState(
        (0.0, 0.0, 0.01),
        (spec.launch_speed_mps * math.cos(angle), 0.0, spec.launch_speed_mps * math.sin(angle)),
        (0.0, spec.backspin_rpm * rpm_to_radps, spec.sidespin_rpm * rpm_to_radps),
    )
    simulation = simulate_flight(initial, SimulationConfig(timestep_seconds=0.02, maximum_seconds=15.0))
    if not simulation.terminated_at_ground:
        raise RuntimeError(f"Template did not reach ground: {spec.template_id}")
    return TemplateFlight(spec, simulation)


@lru_cache(maxsize=4)
def template_bank(profile: str | None = None) -> tuple[TemplateFlight, ...]:
    """Return the deterministic bank; runtime never modifies its physics."""
    return tuple(_simulate_template(spec) for spec in template_specs(profile))


def bank_metadata() -> dict:
    return {
        "template_bank_version": TEMPLATE_BANK_VERSION,
        "aerodynamic_model_version": AERODYNAMIC_MODEL_VERSION,
        "template_count": len(template_specs()),
        "profiles": sorted(_PROFILE_VARIANTS),
        "shapes": sorted(_SHAPE_SIDESPIN_RPM),
    }
