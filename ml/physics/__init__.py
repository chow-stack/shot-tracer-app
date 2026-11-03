"""Bounded forward-flight priors for post-tracking cinematic continuation."""

from .aerodynamics import AERODYNAMIC_MODEL_VERSION
from .simulator import SimulationConfig, SimulationResult, simulate_flight
from .templates import TEMPLATE_BANK_VERSION, TemplateFlight, template_bank

__all__ = [
    "AERODYNAMIC_MODEL_VERSION",
    "SimulationConfig",
    "SimulationResult",
    "TEMPLATE_BANK_VERSION",
    "TemplateFlight",
    "simulate_flight",
    "template_bank",
]
