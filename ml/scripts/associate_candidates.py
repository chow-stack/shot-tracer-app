"""Select one complete golf-ball path from clip-level top-K candidates.

This is a training-free, annotation-free association stage. It combines
motion-component and tiled-detector candidates, searches the complete clip for
a launch from the user-provided address tap, and uses beam search with explicit
gap and terminated states.

Usage:
    ml/.venv/bin/python ml/scripts/associate_candidates.py \
        ml/output/curve-iron.motion-candidates.json \
        --tap-x 576 --tap-y 1659.7333
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.video_io import atomic_write_json  # noqa: E402
from tracker.trajectory_io import build_segments  # noqa: E402


@dataclass(slots=True)
class Candidate:
    frame_index: int
    x: float
    y: float
    confidence: float
    sources: tuple[str, ...]
    source_ranks: dict[str, int]
    raw: dict
    cached_quality: float | None = None


@dataclass(slots=True)
class BeamState:
    score: float
    phase: str
    frame_index: int
    candidate: Candidate | None
    parent: "BeamState | None"
    observations: tuple[tuple[int, float, float], ...]
    gap_count: int
    visible_count: int


def load_candidate_file(path: str) -> tuple[dict, list[dict]]:
    import json

    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    candidates = document.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Candidate file has no candidates array: {path}")
    return document, candidates


def combine_candidates(
    paths: list[str],
    frame_indexes: set[int],
    merge_radius: float,
    allow_background_only: bool = True,
) -> tuple[dict[int, list[Candidate]], list[dict]]:
    """Merge near-identical centers across candidate sources."""
    raw_by_frame: dict[int, list[tuple[dict, str, int]]] = defaultdict(list)
    source_documents: list[dict] = []
    for path in paths:
        document, values = load_candidate_file(path)
        source_documents.append(document)
        default_source = str(document.get("detector", Path(path).stem))
        per_frame_rank: dict[int, int] = defaultdict(int)
        for value in values:
            frame_index = value.get("frame_index")
            if frame_index not in frame_indexes:
                continue
            per_frame_rank[frame_index] += 1
            source = str(value.get("source", default_source))
            if source == "median_background_motion" and not allow_background_only:
                continue
            rank = int(value.get("rank", per_frame_rank[frame_index]))
            raw_by_frame[frame_index].append((value, source, rank))

    combined: dict[int, list[Candidate]] = {}
    for frame_index in sorted(frame_indexes):
        kept: list[Candidate] = []
        values = sorted(
            raw_by_frame.get(frame_index, []),
            key=lambda item: float(item[0].get("confidence", 0.0)),
            reverse=True,
        )
        for raw, source, rank in values:
            x, y = float(raw["x"]), float(raw["y"])
            existing = next(
                (
                    candidate
                    for candidate in kept
                    if math.hypot(candidate.x - x, candidate.y - y) <= merge_radius
                ),
                None,
            )
            if existing is None:
                kept.append(
                    Candidate(
                        frame_index=frame_index,
                        x=x,
                        y=y,
                        confidence=float(raw.get("confidence", 0.0)),
                        sources=(source,),
                        source_ranks={source: rank},
                        raw=raw,
                    )
                )
            else:
                existing.confidence = max(existing.confidence, float(raw.get("confidence", 0.0)))
                existing.sources = tuple(sorted(set(existing.sources) | {source}))
                existing.source_ranks[source] = min(existing.source_ranks.get(source, rank), rank)
        combined[frame_index] = kept
    return combined, source_documents


def candidate_quality(candidate: Candidate) -> float:
    """Small appearance/motion tie-breaker; continuity remains dominant."""
    if candidate.cached_quality is not None:
        return candidate.cached_quality
    quality = 0.18 * max(0.0, min(1.0, candidate.confidence))
    if "overlapping_tiled_yolo" in candidate.sources:
        quality += 0.22 * max(0.0, min(1.0, candidate.confidence))
    if "three_frame_motion" in candidate.sources:
        quality += 0.08
    if candidate.sources == ("median_background_motion",):
        # Background-only evidence is weak at launch, but can legitimately be
        # the only signal once an established track slows to sub-pixel motion.
        # Continuity and the launch geometry remain responsible for identity.
        quality -= 0.25
    if len(candidate.sources) > 1:
        quality += 0.08
    best_rank = min(candidate.source_ranks.values(), default=100)
    quality += 0.08 * max(0.0, 1.0 - (best_rank - 1) / 100.0)
    candidate.cached_quality = quality
    return quality


def initial_states(
    first_index: int,
    second_index: int,
    candidates_by_frame: dict[int, list[Candidate]],
    address: tuple[float, float],
    beam_width: int,
    max_launch_radius: float,
) -> list[BeamState]:
    first = [
        candidate
        for candidate in candidates_by_frame.get(first_index, [])
        if math.hypot(candidate.x - address[0], candidate.y - address[1]) <= max_launch_radius
        and candidate.y <= address[1] + 40.0
    ]
    second = candidates_by_frame.get(second_index, [])
    states: list[BeamState] = []
    for start in first:
        launch_distance = math.hypot(start.x - address[0], start.y - address[1])
        for following in second:
            dx, dy = following.x - start.x, following.y - start.y
            displacement = math.hypot(dx, dy)
            if not 40.0 <= displacement <= 320.0:
                continue
            if dy >= -35.0 or abs(dx) > 170.0:
                continue
            score = (
                candidate_quality(start)
                + candidate_quality(following)
                + 1.8
                - 0.003 * launch_distance
                - 0.002 * abs(dx)
            )
            root = BeamState(
                score=score,
                phase="active",
                frame_index=first_index,
                candidate=start,
                parent=None,
                observations=((first_index, start.x, start.y),),
                gap_count=0,
                visible_count=1,
            )
            states.append(
                BeamState(
                    score=score,
                    phase="active",
                    frame_index=second_index,
                    candidate=following,
                    parent=root,
                    observations=(
                        (first_index, start.x, start.y),
                        (second_index, following.x, following.y),
                    ),
                    gap_count=0,
                    visible_count=2,
                )
            )
    if not states:
        raise RuntimeError("No plausible two-frame launch hypotheses were found")
    return sorted(states, key=lambda state: state.score, reverse=True)[:beam_width]


def predict_next(
    observations: tuple[tuple[int, float, float], ...], frame_index: int
) -> tuple[float, float, float, float]:
    previous_index, previous_x, previous_y = observations[-2]
    last_index, last_x, last_y = observations[-1]
    frame_delta = max(1, last_index - previous_index)
    velocity_x = (last_x - previous_x) / frame_delta
    velocity_y = (last_y - previous_y) / frame_delta
    speed = math.hypot(velocity_x, velocity_y)
    if len(observations) >= 3:
        older_index, older_x, older_y = observations[-3]
        older_delta = max(1, previous_index - older_index)
        older_speed = math.hypot(
            (previous_x - older_x) / older_delta,
            (previous_y - older_y) / older_delta,
        )
        decay = max(0.55, min(1.08, speed / older_speed)) if older_speed > 1e-6 else 0.9
    else:
        decay = 0.72 if speed > 30.0 else 0.9
    ahead = max(1, frame_index - last_index)
    if abs(1.0 - decay) < 1e-6:
        multiplier = float(ahead)
    else:
        multiplier = decay * (1.0 - decay**ahead) / (1.0 - decay)
    return last_x + velocity_x * multiplier, last_y + velocity_y * multiplier, speed, decay


def transition_score(
    state: BeamState, candidate: Candidate, frame_index: int
) -> float | None:
    predicted_x, predicted_y, previous_speed, decay = predict_next(state.observations, frame_index)
    error = math.hypot(candidate.x - predicted_x, candidate.y - predicted_y)
    gap = max(0, frame_index - state.observations[-1][0] - 1)
    gate = max(18.0, 0.34 * previous_speed + 9.0) * (1.0 + 0.65 * gap)
    if error > 2.2 * gate:
        return None

    _, last_x, last_y = state.observations[-1]
    dx, dy = candidate.x - last_x, candidate.y - last_y
    observation_delta = max(1, frame_index - state.observations[-1][0])
    # In the supported rear-view setup, the labelable segment is the rising
    # flight. Allow small downward image motion for quantization/jitter, but
    # reject paths that turn back toward the ground while claiming visibility.
    if dy > 12.0 * observation_delta:
        return None
    displacement = math.hypot(dx, dy) / observation_delta
    if displacement > 340.0:
        return None
    if previous_speed < 15.0 and displacement > max(12.0, 2.0 * previous_speed):
        return None
    previous_dx = state.observations[-1][1] - state.observations[-2][1]
    previous_dy = state.observations[-1][2] - state.observations[-2][2]
    previous_norm = math.hypot(previous_dx, previous_dy)
    direction_penalty = 0.0
    if displacement > 1e-6 and previous_norm > 1e-6:
        cosine = (dx * previous_dx + dy * previous_dy) / (displacement * previous_norm)
        if cosine < -0.15:
            return None
        if previous_speed > 15.0 and cosine < 0.35:
            return None
        direction_penalty = 0.7 * (1.0 - max(-1.0, min(1.0, cosine)))

    expected_speed = max(1.0, previous_speed * decay)
    speed_penalty = 0.35 * abs(math.log(max(1.0, displacement) / expected_speed))
    return (
        0.17
        + candidate_quality(candidate)
        - 1.65 * error / gate
        - speed_penalty
        - direction_penalty
        - 0.18 * gap
    )


def prune(states: list[BeamState], beam_width: int, spatial_bin: float) -> list[BeamState]:
    """Keep diverse hypotheses instead of many near-identical paths."""
    ordered = sorted(states, key=lambda state: state.score, reverse=True)
    kept: list[BeamState] = []
    per_bin: dict[tuple, int] = defaultdict(int)
    for state in ordered:
        if state.phase == "terminated":
            key = ("terminated",)
        elif state.candidate is None:
            last = state.observations[-1]
            key = ("gap", round(last[1] / spatial_bin), round(last[2] / spatial_bin), state.gap_count)
        else:
            key = (
                "active",
                round(state.candidate.x / spatial_bin),
                round(state.candidate.y / spatial_bin),
            )
        if per_bin[key] >= 3:
            continue
        per_bin[key] += 1
        kept.append(state)
        if len(kept) >= beam_width:
            break
    return kept


def associate(
    frame_indexes: list[int],
    candidates_by_frame: dict[int, list[Candidate]],
    address: tuple[float, float],
    *,
    beam_width: int = 600,
    max_gap: int = 2,
    max_launch_radius: float = 190.0,
    spatial_bin: float = 10.0,
    min_visible_before_termination: int = 8,
) -> BeamState:
    if len(frame_indexes) < 2:
        raise ValueError("Association requires at least two frames")
    beams = initial_states(
        frame_indexes[0],
        frame_indexes[1],
        candidates_by_frame,
        address,
        beam_width,
        max_launch_radius,
    )
    for frame_index in frame_indexes[2:]:
        expanded: list[BeamState] = []
        for state in beams:
            if state.phase == "terminated":
                expanded.append(
                    BeamState(
                        score=state.score,
                        phase="terminated",
                        frame_index=frame_index,
                        candidate=None,
                        parent=state,
                        observations=state.observations,
                        gap_count=state.gap_count,
                        visible_count=state.visible_count,
                    )
                )
                continue

            for candidate in candidates_by_frame.get(frame_index, []):
                increment = transition_score(state, candidate, frame_index)
                if increment is None:
                    continue
                expanded.append(
                    BeamState(
                        score=state.score + increment,
                        phase="active",
                        frame_index=frame_index,
                        candidate=candidate,
                        parent=state,
                        observations=(state.observations + ((frame_index, candidate.x, candidate.y),))[-3:],
                        gap_count=0,
                        visible_count=state.visible_count + 1,
                    )
                )

            if state.gap_count < max_gap:
                expanded.append(
                    BeamState(
                        score=state.score - 0.55 - 0.25 * state.gap_count,
                        phase="gap",
                        frame_index=frame_index,
                        candidate=None,
                        parent=state,
                        observations=state.observations,
                        gap_count=state.gap_count + 1,
                        visible_count=state.visible_count,
                    )
                )
            if state.visible_count >= min_visible_before_termination:
                expanded.append(
                    BeamState(
                        score=state.score - 0.35,
                        phase="terminated",
                        frame_index=frame_index,
                        candidate=None,
                        parent=state,
                        observations=state.observations,
                        gap_count=state.gap_count,
                        visible_count=state.visible_count,
                    )
                )
        if not expanded:
            raise RuntimeError(f"All association hypotheses died at frame {frame_index}")
        beams = prune(expanded, beam_width, spatial_bin)
    return max(beams, key=lambda state: state.score)


def expand_beams(
    beams: list[BeamState],
    frame_index: int,
    candidates: list[Candidate],
    *,
    max_gap: int,
    min_visible_before_termination: int,
) -> list[BeamState]:
    """Advance existing launched hypotheses by one frame."""
    expanded: list[BeamState] = []
    for state in beams:
        if state.phase == "terminated":
            expanded.append(
                BeamState(
                    state.score,
                    "terminated",
                    frame_index,
                    None,
                    state,
                    state.observations,
                    state.gap_count,
                    state.visible_count,
                )
            )
            continue
        for candidate in candidates:
            increment = transition_score(state, candidate, frame_index)
            if increment is None:
                continue
            expanded.append(
                BeamState(
                    state.score + increment,
                    "active",
                    frame_index,
                    candidate,
                    state,
                    (state.observations + ((frame_index, candidate.x, candidate.y),))[-3:],
                    0,
                    state.visible_count + 1,
                )
            )
        if state.gap_count < max_gap:
            expanded.append(
                BeamState(
                    state.score - 0.55 - 0.25 * state.gap_count,
                    "gap",
                    frame_index,
                    None,
                    state,
                    state.observations,
                    state.gap_count + 1,
                    state.visible_count,
                )
            )
        if state.visible_count >= min_visible_before_termination:
            expanded.append(
                BeamState(
                    state.score - 0.35,
                    "terminated",
                    frame_index,
                    None,
                    state,
                    state.observations,
                    state.gap_count,
                    state.visible_count,
                )
            )
    return expanded


def associate_full_clip(
    frame_indexes: list[int],
    candidates_by_frame: dict[int, list[Candidate]],
    address: tuple[float, float],
    *,
    beam_width: int = 600,
    max_gap: int = 2,
    max_launch_radius: float = 190.0,
    spatial_bin: float = 10.0,
    min_visible_before_termination: int = 8,
) -> BeamState:
    """Search every adjacent frame pair for launch while retaining old paths."""
    if len(frame_indexes) < 2:
        raise ValueError("Association requires at least two frames")
    beams: list[BeamState] = []
    for position in range(1, len(frame_indexes)):
        previous_index = frame_indexes[position - 1]
        frame_index = frame_indexes[position]
        expanded = expand_beams(
            beams,
            frame_index,
            candidates_by_frame.get(frame_index, []),
            max_gap=max_gap,
            min_visible_before_termination=min_visible_before_termination,
        )
        try:
            expanded.extend(
                initial_states(
                    previous_index,
                    frame_index,
                    candidates_by_frame,
                    address,
                    beam_width,
                    max_launch_radius,
                )
            )
        except RuntimeError:
            pass
        if expanded:
            beams = prune(expanded, beam_width, spatial_bin)
    if not beams:
        raise RuntimeError("No plausible launch hypotheses were found in the complete clip")
    qualified = [state for state in beams if state.visible_count >= min_visible_before_termination]
    if not qualified:
        raise RuntimeError("No launch hypothesis produced a sufficiently supported track")
    return max(qualified, key=lambda state: state.score)


def unwind(final_state: BeamState) -> list[BeamState]:
    states: list[BeamState] = []
    current: BeamState | None = final_state
    while current is not None:
        states.append(current)
        current = current.parent
    states.reverse()
    return states


def trajectory_from_states(states: list[BeamState], timestamps: dict[int, float]) -> list[dict]:
    """Export observed candidates and bounded gap estimates separately."""
    points: dict[int, dict] = {}
    for state in states:
        if state.candidate is None or state.phase != "active":
            continue
        candidate = state.candidate
        points[state.frame_index] = {
            "frame_index": state.frame_index,
            "timestamp_seconds": float(timestamps[state.frame_index]),
            "x": candidate.x,
            "y": candidate.y,
            "confidence": candidate.confidence,
            "sources": list(candidate.sources),
            "source_ranks": candidate.source_ranks,
            "estimated": False,
            "provenance": "observed",
            "uncertainty_px": None,
            "source": "association",
        }

    for position, state in enumerate(states):
        if state.phase != "gap":
            continue
        left = next(
            (prior for prior in reversed(states[:position]) if prior.phase == "active" and prior.candidate),
            None,
        )
        right = next(
            (later for later in states[position + 1 :] if later.phase == "active" and later.candidate),
            None,
        )
        if left is None or right is None or right.frame_index - left.frame_index > 3:
            continue
        left_time, right_time = timestamps[left.frame_index], timestamps[right.frame_index]
        current_time = timestamps[state.frame_index]
        ratio = (
            (current_time - left_time) / (right_time - left_time)
            if right_time > left_time
            else (state.frame_index - left.frame_index) / (right.frame_index - left.frame_index)
        )
        points[state.frame_index] = {
            "frame_index": state.frame_index,
            "timestamp_seconds": float(current_time),
            "x": left.candidate.x + ratio * (right.candidate.x - left.candidate.x),
            "y": left.candidate.y + ratio * (right.candidate.y - left.candidate.y),
            "confidence": 0.5 * min(left.candidate.confidence, right.candidate.confidence),
            "sources": ["short_gap_interpolation"],
            "source_ranks": {},
            "estimated": True,
            "provenance": "estimated_gap",
            "uncertainty_px": None,
            "source": "short_gap_interpolation",
        }
    return [points[index] for index in sorted(points)]


def default_output_path(candidate_path: str) -> str:
    stem = Path(candidate_path).name.split(".", 1)[0]
    return str(Path(__file__).resolve().parent.parent / "output" / f"{stem}.associated-track.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_files", nargs="+")
    parser.add_argument("--tap-x", required=True, type=float)
    parser.add_argument("--tap-y", required=True, type=float)
    parser.add_argument("--output")
    parser.add_argument("--beam-width", type=int, default=600)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--merge-radius", type=float, default=8.0)
    parser.add_argument("--max-launch-radius", type=float, default=190.0)
    parser.add_argument("--spatial-bin", type=float, default=10.0)
    parser.add_argument(
        "--exclude-background-only",
        action="store_true",
        help="Drop median-background components without three-frame/YOLO support",
    )
    return parser.parse_args()


def associate_candidate_files(
    candidate_files: list[str],
    tap: tuple[float, float],
    output_path: str,
    *,
    beam_width: int = 600,
    max_gap: int = 2,
    merge_radius: float = 8.0,
    max_launch_radius: float = 190.0,
    spatial_bin: float = 10.0,
    allow_background_only: bool = True,
) -> dict:
    """Associate complete-clip candidate files and export a trajectory."""
    if beam_width <= 0 or max_gap < 0:
        raise ValueError("beam width must be positive and max gap non-negative")
    first_document, _ = load_candidate_file(candidate_files[0])
    metadata = first_document.get("video", {})
    frame_count = int(metadata.get("frame_count", 0))
    timestamps_values = metadata.get("timestamps_seconds")
    if frame_count <= 0 or not isinstance(timestamps_values, list) or len(timestamps_values) != frame_count:
        raise ValueError("Candidate metadata must contain exact timestamps for the complete clip")
    frame_indexes = list(range(1, frame_count - 1))
    candidates_by_frame, source_documents = combine_candidates(
        candidate_files,
        set(frame_indexes),
        merge_radius,
        allow_background_only=allow_background_only,
    )
    print(
        f"Searching frames {frame_indexes[0]}–{frame_indexes[-1]} for launch with "
        f"{sum(len(values) for values in candidates_by_frame.values())} candidates; "
        f"beam={beam_width}, max_gap={max_gap}"
    )
    final_state = associate_full_clip(
        frame_indexes,
        candidates_by_frame,
        tap,
        beam_width=beam_width,
        max_gap=max_gap,
        max_launch_radius=max_launch_radius,
        spatial_bin=spatial_bin,
    )
    states = unwind(final_state)
    timestamps = {index: float(value) for index, value in enumerate(timestamps_values)}
    trajectory = trajectory_from_states(states, timestamps)
    trajectory_by_frame = {point["frame_index"]: point for point in trajectory}
    frame_states: list[dict] = []
    for state in states:
        frame_record = {"frame_index": state.frame_index, "state": state.phase, "score": state.score}
        point = trajectory_by_frame.get(state.frame_index)
        if point is not None:
            frame_record.update(point)
            if point["estimated"]:
                frame_record["state"] = "estimated_gap"
        frame_states.append(frame_record)

    first_observation = next(
        (state.frame_index for state in states if state.phase == "active" and state.candidate), None
    )
    last_observation = next(
        (state.frame_index for state in reversed(states) if state.phase == "active" and state.candidate), None
    )
    output_path = os.path.abspath(output_path)
    output = {
        "schema_version": 1,
        "trajectory_schema_version": 2,
        "tracker": "clip_level_beam_search",
        "launch_frame": first_observation,
        "termination_frame": last_observation,
        "observation_end_frame": last_observation,
        "continuation_mode": "observed_only",
        "video": metadata,
        "initialization": {"address_x": tap[0], "address_y": tap[1]},
        "settings": {
            "beam_width": beam_width,
            "max_gap": max_gap,
            "merge_radius": merge_radius,
            "max_launch_radius": max_launch_radius,
            "spatial_bin": spatial_bin,
            "allow_background_only": allow_background_only,
            "candidate_files": [os.path.abspath(path) for path in candidate_files],
        },
        "result": {
            "final_score": final_state.score,
            "visible_points": len(trajectory),
            "first_visible_frame": trajectory[0]["frame_index"] if trajectory else None,
            "last_visible_frame": trajectory[-1]["frame_index"] if trajectory else None,
        },
        "observations": [point for point in trajectory if not point["estimated"]],
        "trajectory": trajectory,
        "segments": build_segments(trajectory),
        "frames": frame_states,
    }
    atomic_write_json(output_path, output)
    print(
        f"Saved {len(trajectory)} points ({output['result']['first_visible_frame']}–"
        f"{output['result']['last_visible_frame']}) to: {output_path}"
    )
    return output


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.candidate_files[0])
    associate_candidate_files(
        args.candidate_files,
        (args.tap_x, args.tap_y),
        output_path,
        beam_width=args.beam_width,
        max_gap=args.max_gap,
        merge_radius=args.merge_radius,
        max_launch_radius=args.max_launch_radius,
        spatial_bin=args.spatial_bin,
        allow_background_only=not args.exclude_background_only,
    )


if __name__ == "__main__":
    main()
