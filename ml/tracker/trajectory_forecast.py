"""Small, post-tracking screen-space trajectory forecasting baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MODEL_NAMES = ("regularized_quadratic", "kinematic_state")
MAX_AUTOMATIC_HORIZON_SECONDS = 0.25


@dataclass(frozen=True)
class ForecastModel:
    name: str
    origin_time: float
    scale_seconds: float
    parameters_x: tuple[float, ...]
    parameters_y: tuple[float, ...]
    residual_rms_px: float
    observed_span_seconds: float


def _arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 8:
        raise ValueError("At least eight observed points are required for continuation")
    ordered = sorted(points, key=lambda point: point["timestamp_seconds"])
    times = np.asarray([point["timestamp_seconds"] for point in ordered], dtype=np.float64)
    x = np.asarray([point["x"] for point in ordered], dtype=np.float64)
    y = np.asarray([point["y"] for point in ordered], dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observed trajectory contains non-finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Observed timestamps must be strictly increasing")
    return times, x, y


def _robust_ridge(design: np.ndarray, values: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    weights = np.ones(len(values), dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(4):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(weighted.T @ design + penalty, weighted.T @ values)
        residuals = values - design @ coefficients
        scale = max(1.0, 1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))))
        weights = np.minimum(1.0, (1.5 * scale) / np.maximum(np.abs(residuals), 1e-9))
    return coefficients


def fit_model(points: list[dict], name: str) -> ForecastModel:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown forecast model {name!r}")
    times, x, y = _arrays(points)
    origin = float(times[-1])
    span = float(times[-1] - times[0])
    scale = max(span, float(np.median(np.diff(times))))

    if name == "regularized_quadratic":
        u = (times - origin) / scale
        design = np.column_stack((np.ones_like(u), u, u * u))
        penalty = np.diag((1e-8, 1e-6, 1e-3))
        px = _robust_ridge(design, x, penalty)
        py = _robust_ridge(design, y, penalty)
        predicted_x = design @ px
        predicted_y = design @ py
    else:
        window = min(10, len(times))
        recent_t = times[-window:]
        recent_x = x[-window:]
        recent_y = y[-window:]
        dt = np.diff(recent_t)
        vx = np.diff(recent_x) / dt
        vy = np.diff(recent_y) / dt
        ax = np.diff(vx) / ((dt[1:] + dt[:-1]) * 0.5) if len(vx) > 1 else np.asarray([0.0])
        ay = np.diff(vy) / ((dt[1:] + dt[:-1]) * 0.5) if len(vy) > 1 else np.asarray([0.0])
        px = np.asarray((x[-1], float(np.median(vx[-6:])), float(np.median(ax[-5:]))))
        py = np.asarray((y[-1], float(np.median(vy[-6:])), float(np.median(ay[-5:]))))
        delta = times - origin
        predicted_x = px[0] + px[1] * delta + 0.5 * px[2] * delta * delta
        predicted_y = py[0] + py[1] * delta + 0.5 * py[2] * delta * delta

    residuals = np.hypot(predicted_x - x, predicted_y - y)
    return ForecastModel(
        name=name,
        origin_time=origin,
        scale_seconds=scale,
        parameters_x=tuple(float(value) for value in px),
        parameters_y=tuple(float(value) for value in py),
        residual_rms_px=max(1.0, float(np.sqrt(np.mean(residuals * residuals)))),
        observed_span_seconds=span,
    )


def predict(model: ForecastModel, timestamps: list[float]) -> list[dict]:
    output: list[dict] = []
    for timestamp in timestamps:
        horizon = float(timestamp) - model.origin_time
        if horizon <= 0:
            raise ValueError("Forecast timestamps must follow the final observation")
        if model.name == "regularized_quadratic":
            u = horizon / model.scale_seconds
            basis = (1.0, u, u * u)
            x = sum(value * weight for value, weight in zip(model.parameters_x, basis))
            y = sum(value * weight for value, weight in zip(model.parameters_y, basis))
        else:
            x0, vx, ax = model.parameters_x
            y0, vy, ay = model.parameters_y
            x = x0 + vx * horizon + 0.5 * ax * horizon * horizon
            y = y0 + vy * horizon + 0.5 * ay * horizon * horizon
        growth = 1.0 + 2.0 * horizon / max(model.observed_span_seconds, 1e-6)
        uncertainty = model.residual_rms_px * growth
        output.append({
            "timestamp_seconds": float(timestamp),
            "x": float(x),
            "y": float(y),
            "uncertainty_px": float(uncertainty),
        })
    return output


def automatic_continuation(
    observed: list[dict],
    frame_indexes: list[int],
    timestamps: list[float],
    model_name: str,
    frame_size: tuple[int, int],
) -> tuple[list[dict], ForecastModel, str]:
    if len(frame_indexes) != len(timestamps) or len(frame_indexes) > 60:
        raise ValueError("Automatic continuation requires matching inputs and at most 60 frames")
    model = fit_model(observed, model_name)
    predicted = predict(model, timestamps)
    width, height = frame_size
    diagonal = math.hypot(width, height)
    accepted: list[dict] = []
    for frame_index, point in zip(frame_indexes, predicted):
        if point["timestamp_seconds"] - model.origin_time > MAX_AUTOMATIC_HORIZON_SECONDS:
            break
        margin_x, margin_y = width * 0.25, height * 0.25
        if not (-margin_x <= point["x"] <= width + margin_x):
            break
        if not (-margin_y <= point["y"] <= height + margin_y):
            break
        if point["uncertainty_px"] > 0.25 * diagonal:
            break
        opacity = max(0.15, 0.75 * (1.0 - point["uncertainty_px"] / (0.25 * diagonal)))
        accepted.append({
            **point,
            "frame_index": frame_index,
            "provenance": "inferred",
            "estimated": True,
            "source": "forecast",
            "opacity": float(opacity),
        })
    return accepted, model, "accepted" if accepted else "rejected"


def assisted_continuation(
    start: dict,
    previous: dict,
    endpoint: tuple[float, float],
    frame_indexes: list[int],
    timestamps: list[float],
) -> list[dict]:
    """Create a simple cubic Hermite path with a fixed endpoint."""
    if len(frame_indexes) != len(timestamps) or not frame_indexes:
        raise ValueError("Assisted continuation requires matching non-empty inputs")
    duration = timestamps[-1] - float(start["timestamp_seconds"])
    prior_dt = float(start["timestamp_seconds"]) - float(previous["timestamp_seconds"])
    if duration <= 0 or prior_dt <= 0:
        raise ValueError("Assisted continuation timestamps must be increasing")
    tangent = np.asarray((start["x"] - previous["x"], start["y"] - previous["y"])) / prior_dt
    p0 = np.asarray((start["x"], start["y"]), dtype=np.float64)
    p1 = np.asarray(endpoint, dtype=np.float64)
    m0 = tangent * min(duration, 0.35)
    m1 = np.asarray((0.0, max(0.0, (endpoint[1] - start["y"]) * 0.35)), dtype=np.float64)
    points: list[dict] = []
    for frame_index, timestamp in zip(frame_indexes, timestamps):
        u = (timestamp - float(start["timestamp_seconds"])) / duration
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        position = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        points.append({
            "frame_index": frame_index,
            "timestamp_seconds": float(timestamp),
            "x": float(position[0]),
            "y": float(position[1]),
            "uncertainty_px": None,
            "provenance": "inferred",
            "estimated": True,
            "source": "user_assisted_curve",
            "opacity": float(max(0.25, 0.75 - 0.35 * u)),
        })
    return points


def cinematic_continuation(
    observed: list[dict],
    frame_indexes: list[int],
    timestamps: list[float],
    frame_size: tuple[int, int],
    address_y: float | None = None,
) -> tuple[list[dict], dict]:
    """Complete a plausible 2D tracer arc; this is not a landing estimate."""
    if len(observed) < 8:
        raise ValueError("At least eight observed points are required for cinematic completion")
    if len(frame_indexes) != len(timestamps) or not frame_indexes:
        raise ValueError("Cinematic completion requires matching non-empty future frames")
    ordered = sorted(observed, key=lambda point: point["timestamp_seconds"])
    start = ordered[-1]
    start_time = float(start["timestamp_seconds"])
    duration = float(timestamps[-1]) - start_time
    if duration <= 0:
        raise ValueError("Cinematic completion timestamps must follow the observed track")
    if duration < 1.5:
        raise ValueError("Cinematic completion requires at least 1.5 seconds")

    width, height = frame_size
    recent = ordered[-min(8, len(ordered)):]
    velocities = []
    for previous, current in zip(recent, recent[1:]):
        dt = float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"])
        if dt > 0:
            velocities.append(((current["x"] - previous["x"]) / dt, (current["y"] - previous["y"]) / dt))
    vx_values = np.asarray([velocity[0] for velocity in velocities], dtype=np.float64)
    vy_values = np.asarray([velocity[1] for velocity in velocities], dtype=np.float64)
    vx = float(np.median(vx_values[-6:]))
    vy = float(np.median(vy_values[-6:]))

    acceleration_y_values = []
    recent_times = np.asarray([point["timestamp_seconds"] for point in recent], dtype=np.float64)
    for index in range(1, len(vy_values)):
        midpoint_dt = 0.5 * (
            recent_times[index + 1] - recent_times[index]
            + recent_times[index] - recent_times[index - 1]
        )
        if midpoint_dt > 0:
            acceleration_y_values.append((vy_values[index] - vy_values[index - 1]) / midpoint_dt)
    acceleration_y = (
        float(np.median(acceleration_y_values[-5:]))
        if acceleration_y_values
        else 0.0
    )

    launch_y = float(address_y) if address_y is not None else float(ordered[0]["y"])
    if vy >= 0:
        descent_room = max(0.0, launch_y - float(start["y"]))
        endpoint_y = float(start["y"]) + min(0.12 * height, 0.4 * descent_room)
    else:
        endpoint_y = max(0.55 * height, float(start["y"]) + 0.18 * height)
    endpoint_y = float(np.clip(endpoint_y, 0.35 * height, 0.82 * height))

    total_dx = float(start["x"]) - float(ordered[0]["x"])
    endpoint_dx = 0.35 * total_dx + 0.20 * vx * duration
    endpoint_dx = float(np.clip(endpoint_dx, -0.18 * width, 0.18 * width))
    endpoint_x = float(np.clip(float(start["x"]) + endpoint_dx, 0.05 * width, 0.95 * width))
    center = 0.5 * width
    center_margin = 0.06 * width
    first_x = float(ordered[0]["x"])
    if first_x > center and float(start["x"]) > center:
        endpoint_x = max(endpoint_x, center + center_margin)
    elif first_x < center and float(start["x"]) < center:
        endpoint_x = min(endpoint_x, center - center_margin)
    endpoint_dx = endpoint_x - float(start["x"])

    start_position = np.asarray((start["x"], start["y"]), dtype=np.float64)
    endpoint_position = np.asarray((endpoint_x, endpoint_y), dtype=np.float64)

    ascending = vy < -10.0
    if ascending:
        raw_time_to_apex = -vy / acceleration_y if acceleration_y > 1e-6 else 0.8
        time_to_apex = float(np.clip(raw_time_to_apex, 0.65, min(1.2, 0.45 * duration)))
        apex_y = float(np.clip(float(start["y"]) + 0.5 * vy * time_to_apex, 0.08 * height, float(start["y"])))
        apex_vx = 0.55 * vx
        apex_x = float(np.clip(float(start["x"]) + 0.5 * (vx + apex_vx) * time_to_apex, 0.04 * width, 0.96 * width))
        apex_position = np.asarray((apex_x, apex_y), dtype=np.float64)
    else:
        time_to_apex = 0.0
        apex_vx = vx
        apex_position = start_position.copy()

    descent_duration = duration - time_to_apex
    if descent_duration <= 0.5:
        raise ValueError("Cinematic completion needs at least 0.5 seconds after the apex")

    def hermite(position0, position1, velocity0, velocity1, segment_seconds, u):
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        return (
            h00 * position0
            + h10 * velocity0 * segment_seconds
            + h01 * position1
            + h11 * velocity1 * segment_seconds
        )

    terminal_descent_velocity = np.asarray((
        0.25 * apex_vx,
        1.10 * (endpoint_y - float(apex_position[1])) / descent_duration,
    ), dtype=np.float64)

    points: list[dict] = []
    base_uncertainty = max(6.0, float(np.median([
        math.hypot(
            current["x"] - previous["x"],
            current["y"] - previous["y"],
        )
        for previous, current in zip(recent, recent[1:])
    ])) * 0.35)
    for frame_index, timestamp in zip(frame_indexes, timestamps):
        elapsed = float(timestamp) - start_time
        overall_u = elapsed / duration
        if ascending and elapsed <= time_to_apex:
            segment_u = elapsed / time_to_apex
            position = hermite(
                start_position,
                apex_position,
                np.asarray((vx, vy), dtype=np.float64),
                np.asarray((apex_vx, 0.0), dtype=np.float64),
                time_to_apex,
                segment_u,
            )
        else:
            segment_u = (elapsed - time_to_apex) / descent_duration
            position = hermite(
                apex_position,
                endpoint_position,
                np.asarray((apex_vx, 0.0 if ascending else vy), dtype=np.float64),
                terminal_descent_velocity,
                descent_duration,
                segment_u,
            )
        points.append({
            "frame_index": frame_index,
            "timestamp_seconds": float(timestamp),
            "x": float(position[0]),
            "y": float(position[1]),
            "uncertainty_px": float(base_uncertainty * (1.0 + 5.0 * overall_u)),
            "provenance": "inferred",
            "estimated": True,
            "source": "cinematic_curve_completion",
            "opacity": float(max(0.28, 0.78 - 0.40 * overall_u)),
        })
    return points, {
        "endpoint": {
            "source": "automatic_screen_space_ground_band",
            "x": endpoint_x,
            "y": endpoint_y,
            "is_measured_landing": False,
        },
        "apex": {
            "source": "physics_informed_screen_space_constraint",
            "seconds_after_observation": time_to_apex,
            "x": float(apex_position[0]),
            "y": float(apex_position[1]),
            "is_measured": False,
        },
        "terminal_motion": {
            "vx_pixels_per_second": vx,
            "vy_pixels_per_second": vy,
            "ay_pixels_per_second_squared": acceleration_y,
            "ascending_at_observation_end": ascending,
        },
    }
