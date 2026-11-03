"""Fit frozen 3D templates to a factual 2D track without inverse physics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aerodynamics import AERODYNAMIC_MODEL_VERSION, aerodynamic_model_metadata
from .templates import TEMPLATE_BANK_VERSION, TemplateFlight, template_bank


PROJECTION_MODEL_VERSION = "anchored-perspective-linear-v1"
UNCERTAINTY_MODEL_VERSION = "fit-growth-v1"
TIME_SCALES = (0.65, 0.85, 1.05, 1.3, 1.6, 2.0)
TIME_OFFSETS_SECONDS = (0.0, 0.05, 0.10, 0.15)
CAMERA_DISTANCES_M = (4.0, 8.0, 15.0, 30.0, 60.0)
VERTICAL_HEIGHT_SCALES = (-600.0, -1000.0, -1600.0, -2500.0)
MIN_OBSERVATIONS = 8
MIN_OBSERVED_SPAN_SECONDS = 0.10
MAX_FIT_RMS_PX = 60.0
MAX_STABILITY_PX = 120.0


@dataclass(frozen=True)
class ProjectionParameters:
    time_scale: float
    time_offset_seconds: float
    camera_distance_m: float
    horizontal_lateral_scale: float
    horizontal_depth_scale: float
    vertical_height_scale: float
    vertical_depth_scale: float
    anchor_template_time_seconds: float
    anchor_x: float
    anchor_y: float


@dataclass(frozen=True)
class TemplateFit:
    template: TemplateFlight
    projection: ProjectionParameters
    visible_rms_px: float
    join_velocity_error_px_per_second: float
    address_error_px: float
    stability_endpoint_shift_px: float
    score: float


def _observed_arrays(observed: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if len(observed) < MIN_OBSERVATIONS:
        raise ValueError(f"physics_template requires at least {MIN_OBSERVATIONS} observed points")
    ordered = sorted(observed, key=lambda point: point["timestamp_seconds"])
    times = np.asarray([point["timestamp_seconds"] for point in ordered], dtype=np.float64)
    pixels = np.asarray([(point["x"], point["y"]) for point in ordered], dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(pixels)):
        raise ValueError("Observed points must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Observed timestamps must be strictly increasing")
    if times[-1] - times[0] < MIN_OBSERVED_SPAN_SECONDS:
        raise ValueError("Observed trajectory span is too short for physics_template")
    return times, pixels


def _sample_positions(template: TemplateFlight, template_times: np.ndarray) -> np.ndarray | None:
    source_times = template.simulation.times_seconds
    if template_times[0] < 0 or template_times[-1] > source_times[-1]:
        return None
    return np.column_stack([
        np.interp(template_times, source_times, template.simulation.positions_m[:, axis])
        for axis in range(3)
    ])


def _features(positions: np.ndarray, camera_distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    denominator = camera_distance_m + positions[:, 0]
    if np.any(denominator <= 1e-6):
        raise ValueError("Projection depth passed the camera plane")
    horizontal = np.column_stack((positions[:, 1] / denominator, positions[:, 0] / denominator))
    vertical = np.column_stack((positions[:, 2] / denominator, positions[:, 0] / denominator))
    return horizontal, vertical


def _ridge(design: np.ndarray, values: np.ndarray, penalty: float = 1e-8) -> np.ndarray:
    regularizer = penalty * np.eye(design.shape[1], dtype=np.float64)
    return np.linalg.solve(design.T @ design + regularizer, design.T @ values)


def _terminal_velocity(times: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    count = min(7, len(times))
    velocities = np.diff(pixels[-count:], axis=0) / np.diff(times[-count:])[:, None]
    return np.median(velocities, axis=0)


def _parameters_with_anchor_features(
    camera_distance: float,
    time_scale: float,
    time_offset: float,
    template_time: float,
    anchor_pixel: np.ndarray,
    horizontal_coefficients: np.ndarray,
    vertical_coefficients: np.ndarray,
) -> ProjectionParameters:
    return ProjectionParameters(
        time_scale=time_scale,
        time_offset_seconds=time_offset,
        camera_distance_m=camera_distance,
        horizontal_lateral_scale=float(horizontal_coefficients[0]),
        horizontal_depth_scale=float(horizontal_coefficients[1]),
        vertical_height_scale=float(vertical_coefficients[0]),
        vertical_depth_scale=float(vertical_coefficients[1]),
        anchor_template_time_seconds=template_time,
        anchor_x=float(anchor_pixel[0]),
        anchor_y=float(anchor_pixel[1]),
    )


def _project_with_anchor(
    positions: np.ndarray, anchor_position: np.ndarray, parameters: ProjectionParameters
) -> np.ndarray:
    horizontal, vertical = _features(positions, parameters.camera_distance_m)
    anchor_horizontal, anchor_vertical = _features(anchor_position[None, :], parameters.camera_distance_m)
    horizontal -= anchor_horizontal[0]
    vertical -= anchor_vertical[0]
    x = parameters.anchor_x + horizontal @ np.asarray((
        parameters.horizontal_lateral_scale, parameters.horizontal_depth_scale
    ))
    y = parameters.anchor_y + vertical @ np.asarray((
        parameters.vertical_height_scale, parameters.vertical_depth_scale
    ))
    return np.column_stack((x, y))


def _fit_fixed_mapping(
    template: TemplateFlight,
    times: np.ndarray,
    pixels: np.ndarray,
    time_scale: float,
    time_offset: float,
    camera_distance: float,
    address: tuple[float, float],
    horizon_y: float,
    vertical_height_scale: float,
) -> tuple[ProjectionParameters, float, float, float] | None:
    relative_times = times - times[0]
    template_times = time_offset + time_scale * relative_times
    positions = _sample_positions(template, template_times)
    if positions is None:
        return None
    anchor_position = positions[-1]
    horizontal, vertical = _features(positions, camera_distance)
    anchor_horizontal, anchor_vertical = horizontal[-1], vertical[-1]
    horizontal_design = horizontal - anchor_horizontal
    vertical_design = vertical - anchor_vertical
    pixel_delta = pixels - pixels[-1]

    # The address tap is a low-weight near-ground reference. It constrains the
    # projection without pretending that it identifies a metric landing.
    launch_position = template.simulation.positions_m[0]
    launch_h, launch_v = _features(launch_position[None, :], camera_distance)
    address_weight = 0.35
    horizontal_augmented = np.vstack((horizontal_design, address_weight * (launch_h[0] - anchor_horizontal)))
    # Ground perspective is not an unconstrained regression coefficient. A
    # ground point recedes from the address line toward the supplied/shared
    # horizon as downrange distance increases. Only height scale is fitted.
    fixed_vertical_depth_scale = -(address[1] - horizon_y)
    x_values = np.append(pixel_delta[:, 0], address_weight * (address[0] - pixels[-1, 0]))
    try:
        horizontal_coefficients = _ridge(horizontal_augmented, x_values)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(horizontal_coefficients)):
        return None
    parameters = _parameters_with_anchor_features(
        camera_distance, time_scale, time_offset, float(template_times[-1]), pixels[-1],
        horizontal_coefficients,
        np.asarray((vertical_height_scale, fixed_vertical_depth_scale)),
    )
    projected = _project_with_anchor(positions, anchor_position, parameters)
    residuals = np.linalg.norm(projected - pixels, axis=1)
    rms = float(math.sqrt(np.mean(residuals**2)))
    launch_projected = _project_with_anchor(launch_position[None, :], anchor_position, parameters)[0]
    address_error = float(np.linalg.norm(launch_projected - np.asarray(address)))
    epsilon = min(0.01, 0.25 * (template.simulation.flight_seconds - template_times[-1]))
    if epsilon <= 1e-6:
        return None
    derivative_positions = _sample_positions(template, np.asarray((template_times[-1], template_times[-1] + epsilon)))
    if derivative_positions is None:
        return None
    derivative_pixels = _project_with_anchor(derivative_positions, anchor_position, parameters)
    predicted_velocity = (derivative_pixels[1] - derivative_pixels[0]) / (epsilon / time_scale)
    velocity_error = float(np.linalg.norm(predicted_velocity - _terminal_velocity(times, pixels)))
    return parameters, rms, address_error, velocity_error


def _endpoint(template: TemplateFlight, parameters: ProjectionParameters) -> np.ndarray:
    anchor_position = _sample_positions(template, np.asarray((parameters.anchor_template_time_seconds,)))[0]
    return _project_with_anchor(template.simulation.positions_m[-1:], anchor_position, parameters)[0]


def fit_template(
    observed: list[dict],
    frame_size: tuple[int, int],
    address: tuple[float, float],
    shot_profile: str,
    horizon_y: float | None = None,
) -> tuple[TemplateFit, int]:
    times, pixels = _observed_arrays(observed)
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not all(math.isfinite(value) for value in address):
        raise ValueError("address point must be finite")
    if horizon_y is not None and (not math.isfinite(horizon_y) or not 0 <= horizon_y <= height):
        raise ValueError("horizon_y must be within the frame")
    candidates: list[TemplateFit] = []
    terminal_speed = max(1.0, float(np.linalg.norm(_terminal_velocity(times, pixels))))
    for template in template_bank(shot_profile):
        for time_scale in TIME_SCALES:
            for time_offset in TIME_OFFSETS_SECONDS:
                for camera_distance in CAMERA_DISTANCES_M:
                    for vertical_height_scale in VERTICAL_HEIGHT_SCALES:
                        fitted = _fit_fixed_mapping(
                            template, times, pixels, time_scale, time_offset, camera_distance, address,
                            horizon_y if horizon_y is not None else 0.32 * height,
                            vertical_height_scale,
                        )
                        if fitted is None:
                            continue
                        parameters, rms, address_error, velocity_error = fitted
                        endpoint = _endpoint(template, parameters)
                        if horizon_y is not None:
                            horizon_penalty = 0.12 * abs(endpoint[1] - horizon_y)
                        else:
                            horizon_penalty = 0.0
                        score = rms + 0.08 * address_error + 0.025 * velocity_error + horizon_penalty
                        candidates.append(TemplateFit(
                            template, parameters, rms, velocity_error, address_error, math.inf, score
                        ))
    if not candidates:
        raise ValueError("No physics template can cover the observed timestamps")
    candidates.sort(key=lambda candidate: candidate.score)

    # Refit a bounded shortlist after dropping the earliest 15 percent. Endpoint
    # movement is the stability signal; no clip-specific threshold is introduced.
    drop = max(1, round(0.15 * len(times)))
    stable_candidates: list[TemplateFit] = []
    for candidate in candidates[:12]:
        parameters = candidate.projection
        refit = _fit_fixed_mapping(
            candidate.template,
            times[drop:],
            pixels[drop:],
            parameters.time_scale,
            parameters.time_offset_seconds + parameters.time_scale * (times[drop] - times[0]),
            parameters.camera_distance_m,
            address,
            horizon_y if horizon_y is not None else 0.32 * height,
            parameters.vertical_height_scale,
        )
        if refit is None:
            continue
        refit_parameters = refit[0]
        shift = float(np.linalg.norm(_endpoint(candidate.template, parameters) - _endpoint(candidate.template, refit_parameters)))
        stable_candidates.append(TemplateFit(
            candidate.template,
            parameters,
            candidate.visible_rms_px,
            candidate.join_velocity_error_px_per_second,
            candidate.address_error_px,
            shift,
            candidate.score + 0.08 * shift,
        ))
    if not stable_candidates:
        raise ValueError("Physics-template fit is unstable under fit-window changes")
    stable_candidates.sort(key=lambda candidate: candidate.score)
    best = stable_candidates[0]
    endpoint = _endpoint(best.template, best.projection)
    if best.visible_rms_px > MAX_FIT_RMS_PX:
        raise ValueError(f"Physics-template visible fit is too large: {best.visible_rms_px:.1f}px")
    if best.stability_endpoint_shift_px > MAX_STABILITY_PX:
        raise ValueError("Physics-template endpoint is unstable")
    if not (-0.1 * width <= endpoint[0] <= 1.1 * width and -0.1 * height <= endpoint[1] <= 1.05 * height):
        raise ValueError("Physics-template projected ground is outside defensible frame bounds")
    close_threshold = best.score * 1.10 + 2.0
    close_count = int(sum(candidate.score <= close_threshold for candidate in stable_candidates))
    return best, close_count


def physics_template_continuation(
    observed: list[dict],
    frame_indexes: list[int],
    timestamps: list[float],
    frame_size: tuple[int, int],
    address: tuple[float, float],
    shot_profile: str,
    shot_profile_source: str,
    horizon_y: float | None = None,
    horizon_source: str = "not_provided",
) -> tuple[list[dict], dict]:
    if len(frame_indexes) != len(timestamps) or not frame_indexes:
        raise ValueError("physics_template requires matching non-empty future frames")
    effective_horizon_y = horizon_y if horizon_y is not None else 0.32 * frame_size[1]
    effective_horizon_source = (
        horizon_source if horizon_y is not None else "shared_default_frame_fraction"
    )
    fit, close_count = fit_template(observed, frame_size, address, shot_profile, effective_horizon_y)
    ordered = sorted(observed, key=lambda point: point["timestamp_seconds"])
    observed_times, observed_pixels = _observed_arrays(ordered)
    start_time = float(observed_times[-1])
    elapsed = np.asarray(timestamps, dtype=np.float64) - start_time
    if np.any(elapsed <= 0) or np.any(np.diff(elapsed) <= 0):
        raise ValueError("Future timestamps must be strictly increasing after observations")
    parameters = fit.projection
    template_times = parameters.anchor_template_time_seconds + parameters.time_scale * elapsed
    valid = template_times <= fit.template.simulation.flight_seconds + 1e-9
    template_times = template_times[valid]
    if len(template_times) == 0:
        raise ValueError("Selected physics template has already reached ground")
    positions = _sample_positions(fit.template, template_times)
    anchor_position = _sample_positions(fit.template, np.asarray((parameters.anchor_template_time_seconds,)))[0]
    projected = _project_with_anchor(positions, anchor_position, parameters)

    # Exact C1 join: match the robust terminal screen velocity, then decay the
    # correction so the frozen template controls the later flight shape.
    terminal_velocity = _terminal_velocity(observed_times, observed_pixels)
    epsilon = min(0.01, 0.25 * (fit.template.simulation.flight_seconds - parameters.anchor_template_time_seconds))
    derivative_positions = _sample_positions(fit.template, np.asarray((
        parameters.anchor_template_time_seconds,
        parameters.anchor_template_time_seconds + epsilon,
    )))
    derivative_pixels = _project_with_anchor(derivative_positions, anchor_position, parameters)
    template_velocity = (derivative_pixels[1] - derivative_pixels[0]) / (epsilon / parameters.time_scale)
    velocity_delta = terminal_velocity - template_velocity
    valid_elapsed = elapsed[valid]
    blend_seconds = 0.35
    projected += valid_elapsed[:, None] * np.exp(-valid_elapsed[:, None] / blend_seconds) * velocity_delta

    width, height = frame_size
    diagonal = math.hypot(width, height)
    base_uncertainty = max(5.0, fit.visible_rms_px)
    inferred: list[dict] = []
    for frame_index, timestamp, seconds, pixel in zip(
        np.asarray(frame_indexes)[valid], np.asarray(timestamps)[valid], valid_elapsed, projected
    ):
        uncertainty = base_uncertainty + 10.0 * seconds**1.35 + 0.10 * fit.stability_endpoint_shift_px
        if uncertainty > 0.14 * diagonal:
            break
        if not (-0.12 * width <= pixel[0] <= 1.12 * width and -0.12 * height <= pixel[1] <= 1.08 * height):
            break
        opacity = float(np.clip(0.82 * (1.0 - uncertainty / (0.16 * diagonal)), 0.16, 0.78))
        inferred.append({
            "frame_index": int(frame_index),
            "timestamp_seconds": float(timestamp),
            "x": float(pixel[0]),
            "y": float(pixel[1]),
            "uncertainty_px": float(uncertainty),
            "provenance": "inferred",
            "estimated": True,
            "source": "physics_template_projection",
            "opacity": opacity,
        })
    if not inferred:
        raise ValueError("Physics-template continuation was rejected before its first point")
    endpoint = _endpoint(fit.template, parameters)
    metadata = {
        "mode": "physics_template",
        "quality_status": "accepted",
        "shot_profile": {"value": shot_profile, "source": shot_profile_source},
        "model": {
            "family": "fixed_forward_physics_template",
            "template_bank_version": TEMPLATE_BANK_VERSION,
            "aerodynamic_model_version": AERODYNAMIC_MODEL_VERSION,
            "aerodynamic_model": aerodynamic_model_metadata(),
            "projection_model_version": PROJECTION_MODEL_VERSION,
            "fit_end_frame": int(ordered[-1]["frame_index"]),
        },
        "selected_template_id": fit.template.spec.template_id,
        "close_alternative_count": close_count,
        "projection": {
            "time_scale": parameters.time_scale,
            "time_offset_seconds": parameters.time_offset_seconds,
            "camera_distance_m": parameters.camera_distance_m,
            "horizontal_lateral_scale": parameters.horizontal_lateral_scale,
            "horizontal_depth_scale": parameters.horizontal_depth_scale,
            "vertical_height_scale": parameters.vertical_height_scale,
            "vertical_depth_scale": parameters.vertical_depth_scale,
            "anchor_template_time_seconds": parameters.anchor_template_time_seconds,
            "horizon_y": effective_horizon_y,
            "horizon_source": effective_horizon_source,
        },
        "visible_fit": {
            "rms_error_px": fit.visible_rms_px,
            "join_velocity_error_px_per_second": fit.join_velocity_error_px_per_second,
            "address_error_px": fit.address_error_px,
        },
        "stability": {
            "status": "accepted",
            "endpoint_shift_px": fit.stability_endpoint_shift_px,
            "fit_window_perturbation": "drop_earliest_15_percent",
        },
        "endpoint": {
            "source": "projected_template_ground_intersection",
            "x": float(endpoint[0]),
            "y": float(endpoint[1]),
            "is_measured_landing": False,
        },
        "uncertainty": {
            "model_version": UNCERTAINTY_MODEL_VERSION,
            "base_fit_rms_px": base_uncertainty,
            "increases_after_observation": True,
        },
        "disclaimer": "Club-conditioned cinematic prior; not measured carry, launch, spin, or landing.",
    }
    return inferred, metadata
