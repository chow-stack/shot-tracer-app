# Inferred trajectory design

Updated 2026-08-04. This document defines how the app may continue a tracer
after the ball is no longer defensibly observable. It is a product-rendering
contract, not an expansion of what the camera measured.

Current handoff note: the physics-template continuation is implemented and
frozen. It is not the next task. The current iOS priorities are mandatory user
address-ball initialization and physical-device processing-time optimization;
see `HANDOFF.md` and Section 6 of `Master Plan.md`. Optimizations must preserve
this document's provenance, join, confidence, and rejection contract.

## Decision

Add optional continuation after association and confidence-aware smoothing,
before rendering, runtime optimization, and iOS packaging.

The tracker still stops at the last evidence-qualified point. A separate module
may complete a plausible screen-space tail. Because the landing can be invisible
to both camera and user, the default does not request a landing tap. Fully
automatic continuation remains a cinematic estimate; manual endpoint editing
is optional and aesthetic only.

The first long-form implementation—a two-segment screen-space Hermite curve—is
now a rejected visual baseline. It correctly preserves provenance, but fixed
apex, duration, endpoint-band, and terminal-slope constraints dominate the
shape. Do not tune it per clip. The approved next experiment is a fixed bank of
club-conditioned trajectories produced by a small forward physics simulator,
then fitted projectively to the frozen observed 2D track.

This placement is deliberate:

1. candidate generation and association establish ball identity;
2. smoothing freezes the factual 2D path and bounded internal gaps;
3. inferred continuation consumes that frozen path without feedback;
4. rendering styles each provenance separately; and
5. optimization and iOS packaging implement the already-frozen contract.

## Why this boundary is necessary

The archived golf paper predicts only the next Kalman-filter crop and corrects
it with a detector measurement in the following frame. It does not recover the
unseen remainder of a shot or its landing.

A pixel coordinate from one camera defines a viewing ray, not a unique 3D
point. Research that localizes a sports ball from one image adds constraints
such as camera calibration and measurable, known ball diameter. Multi-camera
systems instead triangulate. This project currently has neither reliable
distant-ball diameter nor scene/camera calibration.

Golf flight is also not an ideal gravity-only parabola: lift and drag vary with
spin, speed, Reynolds number, and ball construction. A low-order image-space
curve can look good, but it is not a physical carry or landing measurement.

Temporary-gap repair and an unseen terminal tail are different problems.
TrackNetV3, for example, rectifies occasional missing locations inside a
trajectory using surrounding visual context. Our existing `estimated_gap`
points follow that bounded principle. Once no later observation closes the gap,
the continuation is forecasting and must be labeled `inferred`.

## Output modes

### `observed_only`

Default factual mode. Render observed points and short, evidence-bounded gap
estimates. Stop or fade when the tracker terminates.

### `automatic_cinematic`

Legacy A/B baseline only. It fits a hand-constrained image-space Hermite tail.
It must remain available only until the replacement visual gate is complete and
must not receive further per-clip tuning.

### `physics_template`

The intended automatic mode. Select from a small, versioned bank of normalized
wood, iron, and wedge flights generated offline with a tested forward simulator.
Fit only a low-dimensional time mapping and screen projection to all reliable
observed points, anchor the join exactly at the final observation, and rank
templates by reprojection error, tangent continuity, and stability. Render the
best plausible fit, or the median of a few statistically indistinguishable fits,
with uncertainty increasing after evidence ends.

Use the address point as a near-ground reference. Accept an optional horizon or
distant-ground-line control to constrain perspective; this control does not
identify the landing. Without calibrated camera pose and ground geometry, the
projected endpoint is still cinematic and must not produce carry, launch, spin,
or landing claims.

### `assisted_landing`

An optional editor, not the required workflow. The user may adjust the apparent
endpoint after seeing a proposed tail. Fit a smooth continuation that
matches the final observed position and direction while reaching the selected
endpoint. Record the endpoint source as `user_tap`; do not convert it into a
metric carry claim.

Use a constrained cubic Hermite/Bézier segment for assisted rendering: its
start is the last observed point, its initial tangent comes from the robust
terminal track direction, and its endpoint is the user control. Regularize the
remaining tangent/curvature and reject self-intersections or discontinuities.
This is a geometric editor, not the automatic physical forecast.

An apex control can be added later only if a single landing control proves
insufficient. Do not begin with multiple mandatory keyframes.

## Trajectory schema v2

Keep exact frame timestamps and original pixel coordinates. Preserve backward
compatibility by accepting the current track JSON as all-observed/estimated and
upgrading it in memory.

```json
{
  "trajectory_schema_version": 2,
  "observation_end_frame": 158,
  "continuation_mode": "assisted_landing",
  "segments": [
    {
      "kind": "observed",
      "points": [
        {
          "frame_index": 158,
          "timestamp_seconds": 2.633,
          "x": 721.4,
          "y": 524.8,
          "uncertainty_px": 3.2,
          "source": "association"
        }
      ]
    },
    {
      "kind": "inferred",
      "points": [
        {
          "frame_index": 159,
          "timestamp_seconds": 2.650,
          "x": 725.1,
          "y": 522.0,
          "uncertainty_px": 7.8,
          "opacity": 0.72,
          "source": "forecast"
        }
      ],
      "model": {
        "family": "regularized_quadratic_2d",
        "fit_end_frame": 158,
        "quality_status": "accepted"
      },
      "endpoint": {
        "source": "user_tap",
        "x": 814.0,
        "y": 1162.0
      }
    }
  ]
}
```

Required provenance values are:

- `observed`: selected candidate supported by the image;
- `estimated_gap`: short missing interval bounded by observations on both sides;
- `inferred`: any point after the last evidence-qualified observation.

Never rename inferred points to observed after smoothing. Store model version,
fit window, quality result, endpoint source, and uncertainty parameters so an
export can be reproduced.

For `physics_template`, continuation metadata must additionally store the shot
profile and its source, template-bank/aerodynamic-model versions, selected
template ID, number of close alternatives, fitted time/projection parameters,
horizon source, visible reprojection error, stability result, and uncertainty
model. Its endpoint must explicitly include `is_measured_landing: false`.

## Replacement implementation

Keep the experiment deliberately small and deterministic.

1. Preserve the approved tracker and schema boundary. Read only the frozen
   exported observations, exact timestamps, video dimensions, address point,
   coarse club profile, and optional horizon metadata.
2. Add `ml/physics/` with separate state, aerodynamics, integration, simulation,
   template, and projection modules. Integrate the coupled position/velocity
   state; do not repeat the position/velocity RK4 mix-up in the pasted proposal.
3. Use gravity, wind-relative quadratic drag, Magnus lift, bounded
   source-documented coefficient functions, optional simple spin decay, and
   ground intersection. Keep coefficient/model versions in output metadata.
4. Precompute a deliberately small fixed bank covering plausible straight,
   draw, and fade shapes for wood, iron, and wedge launch profiles. Runtime must
   not optimize aerodynamic coefficients or spin from the video.
5. Fit a low-dimensional time and projective screen mapping to all reliable
   visible points. The final observed point is immutable and the inferred join
   must be position- and tangent-continuous.
6. Rank templates by visible-point reprojection error, join continuity, and
   stability under modest fit-window changes. Reject or fade when no stable fit
   exists.
7. Preserve `automatic_cinematic` only for A/B comparison. Render the two
   3-woods and straight wedge once with shared settings; do not create per-video
   patches.

The simulator supplies shape priors, not recovered truth. Do not estimate depth
from existing motion/YOLO box size, fit a unique 3D EKF, infer spin from noisy
screen curvature, or report a measured landing. Those upgrades require reliable
ball diameter, calibrated intrinsics/extrinsics and ground, or external launch
telemetry.

## Quality and safety gates

Automatic continuation is rejected or faded early when any of these occur:

- too few observations or too little time/span to constrain the fit;
- unstable endpoint under modest changes to the fit window;
- large held-in residuals or disagreement between the two simple baselines;
- reversal, discontinuity, extreme curvature, or departure from frame bounds;
- forecast uncertainty exceeds the renderer's configured visual tolerance;
- no defensible screen-space ground region exists for a proposed endpoint.

All durations and thresholds must be shared settings selected from development
data, not per-clip patches. The automatic tail has a hard maximum horizon and
must fail closed. Assisted mode may extend farther because the user supplies the
endpoint, but the segment remains inferred.

## Rendering contract

- Observed: current solid tracer treatment.
- Estimated gap: current amber/dashed treatment or an equivalent distinct style.
- Inferred: subtly dashed or translucent, with opacity decreasing as uncertainty
  grows. A legend or one-time UI label should say “estimated path.”
- Never move or replace the last observed point to make the join prettier.
- Rendering settings may alter appearance only, never coordinates or provenance.

## Acceptance gates

1. With continuation disabled, machine-readable tracks and rendered output are
   unchanged from the approved annotation-free pipeline.
2. The continuation module has no candidate-generation or association imports
   and cannot feed inferred points back into tracking.
3. The tracking evaluator ignores `inferred`; a separate evaluator reports
   withheld-tail forecast metrics.
4. Schema round-trip tests preserve timestamps, coordinates, provenance,
   endpoint source, model version, and uncertainty.
5. The fixed baseline comparison is reported honestly. Long visual completion
   uses a separately labeled constrained cinematic curve, not a physical claim.
6. Visual QA covers straight, curved, short-visible, and cloud-blended clips.
7. Documentation and UI make no claim of actual landing, carry, launch angle,
   spin, or 3D reconstruction.
8. Physics unit tests cover vacuum motion, drag energy loss, lift direction,
   timestep convergence, finite outputs, and ground intersection.
9. Template/profile choices are fixed before the three-clip visual gate. Wood,
   iron, and wedge use shared profile-level settings, never clip-specific ones.
10. The template bank and fitter are post-tracking only; no physics prediction
    changes candidates, association, launch, termination, or tracking metrics.

The withheld-tail test measures forecasting where later visible truth exists.
It calibrates safe horizon and model choice; it cannot validate the true landing
after the real ball is unobservable. That would require additional ground truth,
calibration, or sensors.

## Later upgrades, only if evidence warrants them

- Automatically estimate the horizon/ground mask after the optional manual
  horizon control has demonstrated value.
- Use camera intrinsics and reliable apparent ball diameter for constrained 3D
  localization while the ball is still resolvable.
- Learn a probabilistic prior from many source-diverse, sensor-ground-truthed
  golf shots, reporting multiple hypotheses rather than false certainty.
- Add launch-monitor, radar, or stereo input for actual flight and landing
  measurement.

These are separate product phases, not prerequisites for the inferred tracer.

## Completed baseline result and superseding decision

Implemented 2026-08-04 without adding further models or tuning loops. The fixed
five-clip, 10%/20%/30% withheld-tail comparison produced:

- regularized quadratic: 114.32px median of case medians and 481.70px p95 of
  final-point errors;
- kinematic state: 8.06px median of case medians and 646.66px p95 of
  final-point errors.

The kinematic baseline is clearly better for most short tails, but its long-tail
failures mean it is not approved as a ground-reaching predictor. The product
therefore uses it only as research evidence for short continuity. Full visual
completion uses a two-segment cubic Hermite curve joined to the robust terminal
direction and ending in an automatic screen-space ground band. For rising
tracks, recent timestamped velocities and robust vertical acceleration estimate
a bounded remaining ascent; the apex is constrained to 0.65–1.2 seconds after
the last observation with zero vertical derivative. The default total
completion is 2.0 seconds, leaving a longer, shallower descent. Both apex and
endpoint are stored as unmeasured; optional manual adjustment is not truth
either.
The machine-readable report is
`ml/output/trajectory-withheld-tail.evaluation.json`.

The first shared-setting review artifacts are
`ml/output/left-3wood.cinematic.mp4` and
`ml/output/curve-3wood.cinematic.mp4`, followed by
`ml/output/straight-wedge.cinematic.mp4`.

The user subsequently rejected the invisible descent as unrealistic. Local
inspection confirms that all three clips end while rising and all three hit the
same 0.65-second apex floor; the wedge endpoint is exactly the fixed 0.55-frame-
height band. Therefore the Hermite completion did not pass the visual gate and
runtime optimization is paused only long enough to implement the bounded
`physics_template` replacement described above.

The replacement was implemented and accepted on 2026-08-04. Its frozen bank
contains 18 versioned wood/iron/wedge flights. Shared-setting review on
`left-3wood`, `curve-3wood`, and `straight-wedge` found the physics-template
tails more realistic, with continuous joins and an apex before descent. The
simulator reaches metric ground internally, but its image endpoint is a
cinematic projected ground intersection; it is not expected to return to the
near-camera address point or imply a measured landing. Stage F2 is now frozen.

Repository review reinforces this boundary:

- Mizuno's Apple Vision example detects/fits observed image trajectories; its
  projected points do not recover an unseen flight.
- `samo2491/GolfSimulator` uses close, side-on capture plus club priors before a
  forward simulator; its fixed aerodynamic constants are not adopted.
- `Simple-Golf-Simulator` is an educational forward model that assumes launch
  state and coefficients are already known.
- `libgolf` is the strongest modular forward-simulation reference and validation
  example, but still requires launch data and is GPL-3.0. Do not copy or link it
  into the product without an explicit licensing decision.

## Primary sources

- Zhang et al., [Efficient Golf Ball Detection and Tracking Based on CNNs and
  Kalman Filter](https://arxiv.org/abs/2012.09393)
- Chen and Wang, [TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations
  and Trajectory Rectification](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)
- Van Zandycke and De Vleeschouwer, [Ball 3D Localization From a Single
  Calibrated Image](https://arxiv.org/abs/2204.00003)
- Bearman and Harvey, [Golf Ball Aerodynamics](https://doi.org/10.1017/S0001925900007617)
- Smith et al., [Aerodynamics of Golf Balls in Still Air](https://doi.org/10.3390/proceedings2060238)
- Grad et al., [Physics-Based 3D Ball Trajectory Reconstruction from Monocular
  Soccer Video](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Grad_Physics-Based_3D_Ball_Trajectory_Reconstruction_from_Monocular_Soccer_Video_A_CVPRW_2026_paper.html)
