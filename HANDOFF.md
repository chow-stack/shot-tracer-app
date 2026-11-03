# Shot Tracer Agent Handoff

Last updated: 2026-08-05

## Progress since 2026-08-05 device-feedback pass

The user ran the app on a physical device and reported three things: the tap
surface's thumbnail sometimes shows a frame well after time zero (ball
already in flight), a 6-second clip took ~2-3 minutes to process, and the
resulting track "focused on another object" instead of the ball. Fixed the
thumbnail frame-accuracy bug (`AVAssetImageGenerator` now requests zero time
tolerance so it decodes frame zero exactly instead of snapping to the nearest
sync sample) and reworded the writer-stall error, which mentioned "ProRes" in
a way that read as a format restriction — the import path has never had a
format allowlist; the message now says so and explains the real cause (a
30-second `AVAssetWriter` backpressure timeout that heavier formats like
ProRes are more likely to hit).

**Not yet diagnosed, and the next agent should not guess:** whether the
mis-tracked run was on a 4K-recorded clip (already addressed by the
2026-08-04 coordinate-mapping fix — verify this first) or a genuine
association failure on a stable/1080p clip, and where the 2-3 minutes of
processing time actually goes. Both require the `performance`/`tracking`
unified-log lines (subsystem `com.caio.ShotTracer`) from an actual device
run; this environment cannot produce them. Ask the user to rebuild, retest
on the same clip, and paste back the "Stage timings" line and the
"Tracking candidates"/"Accepted track" lines before changing any detection,
association, or performance code.

## Progress since 2026-08-04

Both parts of item 2 below (mandatory tap) are done: `manualAddress` is
non-optional through `ShotProcessingRequest`/`VisionShotDetector.detect`,
`AutomaticShotSetup.initialize` is unreachable from production, and the UI
always shows the tap surface with "Trace Shot" disabled until a tap is set.
While doing that, a real coordinate-mapping bug was found and fixed: the
setup-view thumbnail was capped at 1080x1920 while decode/export use the
source's full `AVVideoComposition.renderSize`, so on any clip recorded above
that (e.g. 4K 60fps) the tap landed in the wrong pixel space by roughly 2x.
This is a strong candidate for at least some of the reported "ball not
detected" behavior — see `NOTES.md` for detail and verify explicitly on a
4K-recorded clip.

Item 1 (runtime profiling) is partially done: stage-level `ContinuousClock`
timing instrumentation is implemented and logged via unified logging
(subsystem `com.caio.ShotTracer`, category `performance`), covering decode,
vImage downscale/change, connected components, Vision requests, candidate
merge, association, continuation, and export. Actual physical-device
numbers are still outstanding — this environment cannot run a Release build
on real hardware. See `NOTES.md` "Current status" for exactly what to read
in Console.app/Instruments and report back.

21 active Swift tests and the unsigned generic iOS build both pass after
these changes.

Read this file first. Then read the current-status, architecture, and immediate-
next-steps sections of `Master Plan.md`, plus `ios/ShotTracerCore/README.md`.
Read `INFERRED_TRAJECTORY.md` only when touching continuation. Use `RESEARCH.md`
for method rationale and `NOTES.md` for operational history. `paper.md`, most of
`ml/TRACKING.md`, and `golf_ball_reference/` are historical references, not the
current architecture.

## Immediate directive for the next agent

The user has made two product priorities explicit. Until both are complete, do
not start new model training, handheld support, live capture, physics retuning,
or visual feature work.

1. **Reduce physical-device processing time.** A video shorter than 10 seconds
   currently takes unacceptably long to produce an output. Measure stage-level
   wall time on the iPhone first: sequential decode, vImage downscale/change,
   connected components, Vision requests, candidate merge, association,
   continuation, and AVAssetWriter export. Optimize the measured bottleneck,
   with cancellation and bounded memory intact. Report before/after timings on
   the same clips and require trajectory/output regression equivalence; do not
   trade correctness for a faster progress bar or add retry loops.
2. **Remove automatic ball initialization.** Every selected video must require
   the user to tap the ball at address before processing can start. Remove the
   `Automatic setup` copy and `Use automatic ball position` control, make the
   address step mandatory in the workflow/request boundary, and never call
   `AutomaticShotSetup.initialize` in the production iOS path. Keep the tap in
   original portrait pixel coordinates and use it to anchor association. The
   automatic code and tests may remain as research/reference only if clearly
   unreachable from production. This directive concerns ball/address
   initialization; do not invent a broader club-profile redesign without a
   separate decision.

Recommended order: make the tap requirement explicit first so profiling and
optimization measure the intended production workflow, then use the known
address to safely reduce unnecessary full-frame work. Verify manual tap,
cancel, preview, audio, save, and share on the physical iPhone.

Profile a Release configuration on the physical iPhone; Debug timings are not
an acceptance result. Use unified signposts or `ContinuousClock` stage metrics
plus Instruments Time Profiler/Allocations where needed. Keep the test set
bounded and repeatable: the same representative sub-10-second clips, the same
tap coordinates, and comparable thermal state. Save total time, per-stage time,
frame count/resolution/fps, and output trajectory summary. If a proposed speedup
changes selected frames or coordinates, treat it as an accuracy change and run
the frozen parity/device visual gates before accepting it.

## Goal and scope

Build an offline tracer for stable, rear-view, approximately 60 fps iPhone golf
videos. The user supplies one address-ball tap. The system detects launch,
tracks the same ball while evidence exists, stops honestly, optionally appends
a clearly labeled inferred cinematic tail, renders in original coordinates, and
exports a video.

Current scope excludes handheld compensation, live capture, metric carry,
launch-monitor claims, and unique monocular 3D reconstruction.

## Frozen and approved

The annotation-free observed tracker is accepted. Do not retune or replace it.

Production-shaped flow:

1. `ml/scripts/run_shot_tracer.py` sequentially decodes the complete clip with
   exact timestamps and one explicit address tap.
2. `detect_motion_candidates.py` produces permissive top-K motion candidates.
3. Tiled YOLO supplies optional supporting/fallback candidates.
4. `associate_candidates.py` uses clip-level beam search with launch, gap, and
   termination states.
5. `render_associated_track.py` renders observed points and bounded gap
   estimates. Dense annotations are evaluation-only.

The frozen tracker passed three labeled development clips, four untouched visual
clips, and quantitative review on two held-out 3-woods. The two 3-woods stop
15/8 frames before the last weak cloud-blended human labels, with 100% precision
on emitted points and 1.30/1.57 px median error. This conservative evidence
boundary is intentional; do not loosen it to make the tracer reach the ground.

Trajectory schema v2 is also accepted:

- `observed`: image-supported association output;
- `estimated_gap`: short missing interval bounded by observations;
- `inferred`: every point after the final evidence-qualified observation.

Inferred points must never feed candidate generation, association, launch,
termination, or tracking metrics.

## Rejected continuation baseline

`ml/tracker/trajectory_forecast.py::cinematic_continuation` currently creates a
two-segment screen-space Hermite tail. It preserves provenance correctly but
failed the user visual gate on:

- `ml/output/left-3wood.cinematic.mp4`
- `ml/output/curve-3wood.cinematic.mp4`
- `ml/output/straight-wedge.cinematic.mp4`

The descent looks early/steep and unlike a real shot. All three tracks end while
rising and hit the same 0.65-second apex floor. Duration defaults to two seconds,
the endpoint comes from a fixed image-height band, and the terminal slope is
constructed from that endpoint. The wedge endpoint is exactly `0.55 * height`.

Keep this mode temporarily for A/B comparison. Do not tune its constants per
clip and do not proceed directly to runtime optimization.

## Research conclusion

A single center-only monocular track does not identify depth, launch speed,
spin, camera pose, or landing. An EKF cannot create those missing observations,
and current motion/YOLO boxes are not reliable physical ball diameters.

Repository review:

- Mizuno's Apple Vision example is useful for native iOS observed-trajectory
  detection, confidence, ROI, timestamps, and coordinate handling. Its
  projected points fit recent detections; they do not predict an invisible shot.
- `samo2491/GolfSimulator` combines close side-on capture, a club prior, and a
  forward simulator. Borrow the club-conditioned prior concept, not its fixed
  aerodynamic constants or pixel-derived accuracy claims.
- `Simple-Golf-Simulator` is a transparent educational forward model that
  assumes launch state, spin axis, and coefficients are already known.
- `libgolf` is the best architecture/validation reference: pluggable
  aerodynamics, integrator, ground phases, tests, and regression data. It still
  requires launch parameters and is GPL-3.0. Do not copy or link its code into
  the product without an explicit licensing decision.

Use primary literature for equations and constants. The result remains a
cinematic prior, not recovered physical truth.

## Physics-template implementation status

Implemented 2026-08-04. The code now includes the versioned `ml/physics/`
simulator, an 18-flight fixed wood/iron/wedge template bank, a bounded discrete
time/perspective fitter, exact join anchoring with terminal-tangent correction,
fit-window stability checks, uncertainty fade, complete schema metadata, and
the `physics_template` CLI mode.

The complete suite passes 44 tests. The unchanged five-clip withheld-visible-
tail baseline report still gives 114.32 px / 481.70 px for the quadratic model
and 8.06 px / 646.66 px for the kinematic model; this remains short-horizon
diagnostic evidence, not invisible-landing validation.

Shared-setting review artifacts are:

- `ml/output/left-3wood.physics-template.mp4`
- `ml/output/curve-3wood.physics-template.mp4`
- `ml/output/straight-wedge.physics-template.mp4`

Their visible-fit RMS values are 13.70 px, 18.11 px, and 9.66 px. Endpoint
shifts under the fixed fit-window perturbation are 0.47 px, 5.15 px, and 0.16
px. Automated and contact-sheet review found continuous joins and an apex before
descent on all three. The user accepted the shapes as more realistic on
2026-08-04. Stage F2 is frozen; proceed to measured runtime optimization without
retuning the tracker, template bank, projection grid, or individual clips.

## Runtime optimization status

Two regression-protected passes are complete. On the 294-frame `top-iron` clip,
motion extraction fell from 40.47 seconds (138.6 ms/searchable frame) to 17.30
seconds (59.3 ms/searchable frame), a 57.2% reduction. The changes use equivalent
OpenCV uint8 pixel primitives and an exact specialized linear 90th percentile
for bounded component arrays. All 29,091 candidates, settings, raw-component
counts, and the downstream 95-point association track are identical to the
fresh baseline and approved artifact. Candidate-quality caching leaves
association output exact but only changes this case from 11.10 to 10.98 seconds.

The Python suite now passes 47 tests. `ml/fixtures/ios_parity_v1.json` freezes
five reference cases covering motion candidates, association, and the three
physics-template outputs. `ios/ShotTracerCore` now builds as a Swift package and
contains schema/candidate models, deterministic multi-source candidate merging,
the full-clip launch/association/termination beam search, the frozen SI-unit
physics simulator and 18-flight template bank, the anchored screen-projection
continuation fitter, a provenance-aware Core Graphics renderer, and a
sequential `AVAssetReader`/`AVAssetWriter` exporter. The media path applies the
source presentation transform before drawing, preserves exact presentation
times, optionally passes through audio, reports progress, and cooperatively
cancels. Swift
association matches the frozen Python trajectory exactly for both motion-only
and combined motion + tiled-detector inputs. Swift physics continuation also
matches all three accepted Python artifacts within the frozen manifest
tolerances, including template selection, fit metadata, endpoints, and every
inferred point. At that portability checkpoint, eleven active Swift tests
passed; the current total is 21 below. Two full media tests are explicitly
disabled on this host because the sandbox cannot decode/encode the fixtures and
must run in an iOS/device integration scheme.

The offline iOS app stage is implemented. `ios/ShotTracerApp` is an iOS 17
SwiftUI app backed by the reusable `ShotTracerUI` package. It includes Photos
video selection using file transfer, progress/cooperative cancel, autoplaying
source and result previews, reset cleanup, add-only Photos saving, and system
sharing. The current UI still exposes automatic address setup and an optional
tap correction; that behavior is now explicitly scheduled for removal by the
directive above.

The current native detector performs one sequential timestamped portrait-
coordinate pass. It collects plausible Apple Vision trajectory samples and a
half-resolution three-frame motion response, merges cross-source candidates,
then routes them through the parity-tested Swift `TrajectoryAssociator`.
Prediction is confidence-gated and rendering-only smoothing does not mutate
stored observations. Unified logs in category `tracking` report source counts
and the accepted frame range. The automatic address initializer is still
reachable today; the next agent must make a user tap mandatory before treating
the workflow as production-ready.

Host verification passes 47 Python tests and 21 active Swift tests; two
AVFoundation media tests remain device-only. The Xcode project and local Swift
package resolve, and the unsigned generic iOS app build succeeds with the
installed iOS 26.5 SDK.

Physical-device testing then exposed an AVAssetWriter backpressure stall around
83% on a four-second 60 fps clip. The exporter previously appended the complete
video before appending audio, allowing the writer's interleaver to stop accepting
video while it waited for audio timestamps. It now drains audio alongside each
video frame, finishes any remaining audio after video, handles every writer
terminal state, and reports a localized error after a 30-second readiness stall.
All 13 active Swift tests at that checkpoint passed after this fix; later
tracking and reliability coverage raised the current active total to 21.

The UI now uses a retained `AVPlayer` created from a SwiftUI task and starts
playback automatically for both the selected source and completed output. When
Optional corrections is expanded, the source player is replaced by the still
address-tap view. The output screen has a dedicated add-only PhotoKit save
action with permission, saving, success, and error states, while retaining a
separate system ShareLink. The generated app Info.plist contains
`NSPhotoLibraryAddUsageDescription`. Package tests and the unsigned generic iOS
app build pass.

Device review notes for the next bounded pass (not yet implemented): an iron
was automatically classified as wood, and the observed-to-inferred transition
looked paused/disconnected before an implausible-looking predicted tail. Do not
retune from that single clip. First retain the per-profile fit scores and the
exact observed/inferred frame indexes for representative device runs. The
current club selector does not directly use screen starting height; its hard
short-observed-span prior favors wood and is a likely cause. Add confidence
gating and a neutral/fallback policy only after reviewing labeled wood/iron/
wedge scores. For the transition, verify dense consecutive timestamps, then
enforce first-frame position and terminal-velocity continuity and evaluate a
short timestamp-domain blend before changing the physics templates or visual
shape. Re-run the fixed multi-clip visual gate rather than patching one video.

### Reliability pass implemented 2026-08-04

The native app no longer asks `VisionShotDetector` to choose one trajectory by
a coarse whole-track score. It retains all plausible Apple Vision trajectory
samples, merges duplicates within 8 pixels, caps work at 50 candidates per
frame, and routes them through the frozen Swift `TrajectoryAssociator`. A manual
address anchors that search directly. Automatic setup must first find a
detector-supported lower-frame launch, then the same associator must confirm at
least 12 trajectory samples with a positive score. No retry loop or adaptive
per-video tuning was added; decoding, candidate caps, beam width, and export
stall handling are all bounded.

Prediction now has a second conservative gate. It requires at least 12 observed
points, no gap beyond the associator's two missing-frame allowance, a positive
association score, and a directionally stable six-point terminal window. The
automatic club-family continuation also needs confidence of at least 0.70.
Weak evidence exports only the observed tracer and the UI says `Observed
flight`; the previous silent iron fallback was removed. The reviewed template
selector itself remains unchanged for frozen-fixture parity, but weak selections
are no longer automatically presented as fact.

Jaggedness is reduced in the renderer using a symmetric five-sample display
kernel. This does not modify trajectory JSON, timestamps, confidence, source
ranks, or provenance, and the first/last observed positions remain exact so the
address and observed-to-inferred join do not move.

Verification after the first reliability pass: 19 Swift tests pass, including exact frozen
motion/combined association parity, three physics-template parity cases,
automatic launch/profile fixtures, distractor rejection, smoothing, and quality
gates. Two media integration tests remain explicitly device-only. The unsigned
generic iOS app build succeeds against the installed iOS 26.5 SDK.

Device testing still showed incorrect tracks and frequent tap requests, so the
documented fallback is now implemented. A streaming half-resolution
three-frame motion response runs inside the existing sequential decode pass and
adds compact, weighted connected-component candidates. Work is bounded to two
thresholds, 100 candidates per frame, and the existing 600-state association
beam. Motion and Vision candidates within 8 pixels are merged with both sources
preserved. Automatic launch initialization may use either source, but the same
sustained association and prediction gates remain mandatory. Two synthetic
motion-component tests raise the active Swift total to 21; all pass. The
implementation follows Apple's vImage guidance for image scaling while custom
component traversal retains the statistics required by the frozen algorithm.

The matching Xcode platform is installed. Superseding next step: implement the
mandatory address tap and optimize the measured device runtime as specified at
the top of this file. Then rerun the same short clips and record total and
stage-level time plus `correct` or `observed-only`. Do not replace the
associator, loosen prediction gates, add an unbounded retry, or tune a constant
for an individual clip.

## Implemented physics-template contract

Add a new `physics_template` continuation mode. Keep it small and deterministic.

### 1. Forward simulator

Create:

```text
ml/physics/
    __init__.py
    state.py
    aerodynamics.py
    integrator.py
    simulator.py
    templates.py
    projection.py
```

Requirements:

- SI units internally.
- Integrate the coupled position/velocity state correctly.
- Gravity, wind-relative quadratic drag, Magnus lift, bounded and documented
  `Cd`/`Cl`, optional simple spin decay, and ground intersection.
- Version the aerodynamic model and constants.
- No bounce/roll yet; stop at first ground intersection.
- No runtime inference of aerodynamic coefficients, spin, or unique depth.

Required unit tests:

- vacuum trajectory agrees with the analytic parabola;
- drag reduces speed/energy;
- Magnus acceleration is perpendicular to relative velocity;
- timestep halving converges;
- simulation remains finite and terminates at ground;
- invalid state/timestep inputs fail clearly.

### 2. Fixed template bank

Generate a deliberately small, versioned offline bank for:

- profiles: `wood`, `iron`, `wedge`;
- shapes: straight plus a small draw/fade range;
- a predeclared bounded set of plausible launch speed, angle, and spin values.

Do not start a tuning loop. Freeze the bank before the visual gate.

### 3. Runtime 2D fitting

Consume only frozen observations, exact timestamps, frame dimensions, address
point, coarse club profile, and optional horizon/distant-ground line.

- Fit a low-dimensional time mapping and screen projection to all reliable
  observed points.
- Anchor the last observed position exactly.
- Preserve terminal tangent continuity.
- Rank templates by observed reprojection error, join continuity, and stability
  under modest fit-window changes.
- Output increasing uncertainty and opacity fade after evidence ends.
- Stop at projected ground, video end, or uncertainty rejection.
- The address point is the near-ground reference. A horizon control is allowed
  because the user need not see the landing; it is not a landing tap.

Avoid a full 3D EKF, ball-radius depth, nonlinear aerodynamic parameter fitting,
or physics feedback into tracking.

### 4. Schema and CLI

Extend continuation metadata without breaking schema v2:

- `mode: physics_template`
- `shot_profile` and its source
- template-bank/model version
- selected template ID and close-alternative count
- projection parameters and horizon source
- visible-fit error and stability result
- endpoint explicitly marked `is_measured_landing: false`
- uncertainty model/version

Add `physics_template` to `ml/scripts/continue_trajectory.py`. Preserve
`observed_only`, `assisted_landing`, and the legacy cinematic mode during A/B.

### 5. One bounded acceptance pass

1. Run the complete test suite.
2. Run withheld-visible-tail evaluation; report results honestly but do not use
   it to claim invisible landing accuracy.
3. Render exactly these shared-setting transfer clips:
   - `left-3wood`
   - `curve-3wood`
   - `straight-wedge`
4. Compare once against the existing cinematic videos.
5. If the shapes are plausible and joins are clean, freeze Stage F2 and proceed
   to runtime optimization. Do not patch individual videos.

## Key files

- `Master Plan.md` — architecture and phase gates.
- `INFERRED_TRAJECTORY.md` — Stage F2 contract and acceptance rules.
- `RESEARCH.md` — literature and repository review.
- `NOTES.md` — results, operational warnings, and old commands.
- `ios/ShotTracerCore/Sources/ShotTracerUI/ShotTracerAppView.swift` — current
  automatic/optional-tap UI that must become a mandatory tap workflow.
- `ios/ShotTracerCore/Sources/ShotTracerUI/ShotTracerWorkflow.swift` — workflow
  state and process gate; make `manualAddress` required here.
- `ios/ShotTracerCore/Sources/ShotTracerUI/OfflineShotProcessor.swift` — native
  detect/continue/export orchestration and timing boundary.
- `ios/ShotTracerCore/Sources/ShotTracerCore/VisionShotDetector.swift` — one-pass
  Vision + motion candidate collection and automatic/manual association branch.
- `ios/ShotTracerCore/Sources/ShotTracerCore/ThreeFrameMotionCandidateExtractor.swift`
  — likely runtime hotspot; half-resolution vImage scaling, frame changes,
  connected components, and top-100 cap.
- `ios/ShotTracerCore/Sources/ShotTracerCore/TrajectoryAssociator.swift` — frozen
  parity-tested beam association; optimize only with exact regression evidence.
- `ios/ShotTracerCore/Sources/ShotTracerCore/TrajectoryVideoExporter.swift` —
  sequential audio/video export and bounded writer-stall handling.
- `ml/tracker/trajectory_forecast.py` — short forecasts and rejected Hermite
  continuation baseline.
- `ml/tracker/trajectory_io.py` — schema v2/provenance handling.
- `ml/scripts/continue_trajectory.py` — continuation CLI/document assembly.
- `ml/scripts/evaluate_trajectory_forecast.py` — withheld-tail evaluator.
- `ml/scripts/render_associated_track.py` — provenance-aware renderer.
- `ml/tests/test_trajectory_continuation.py` — current continuation tests.

## Verification and safeguards

Run:

```bash
MPLCONFIGDIR=/tmp/shot-tracer-matplotlib \
  ml/.venv/bin/python -m unittest discover -s ml/tests -v
```

Run Swift tests from `ios/ShotTracerCore`:

```bash
CLANG_MODULE_CACHE_PATH="$PWD/.build/module-cache" \
SWIFTPM_CONFIG_PATH="$PWD/.build/swiftpm-cache" \
swift test
```

Verify the app target without signing:

```bash
xcodebuild -project ios/ShotTracerApp/ShotTracerApp.xcodeproj \
  -scheme ShotTracerApp -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
```

Known media rule: decode iPhone MOVs sequentially. OpenCV random seeking was
measured 5–6 frames off despite reporting success.

The workspace currently has no `.git` directory, so inspect changes directly
and do not assume Git history or rollback is available.

Do not overwrite approved `*.annotation-free.mp4`, association JSON, or existing
cinematic A/B artifacts. Use new `*.physics-template.*` output names.

## Stop conditions

For the immediate iOS work, stop and report evidence instead of guessing when:

- physical-device profiling is unavailable after timing instrumentation is in
  place; ask the user to run the bounded profile and return the measurements;
- an optimization changes trajectory identity/coordinates without an explicit
  accuracy review;
- removing automatic setup exposes a coordinate-mapping ambiguity; verify the
  tap overlay against the portrait source frame rather than adding a fallback;
- the production request reaches processing without a tap; fail locally in the
  workflow and do not silently restore automatic initialization.

Stop and document rather than expand scope if the fixed template bank cannot fit
the visible tail stably. Do not respond by adding clip-specific constants, a
large optimizer, learned inverse physics, camera segmentation, or more tracker
tuning. Any such expansion requires a new product decision.
