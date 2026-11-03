# iOS Shot Tracer App: Master Plan

For continuation by another agent, read `HANDOFF.md` first. It records the exact
approved artifacts, inference-leakage boundary, immediate implementation, and
acceptance tests as of 2026-08-04.

Updated 2026-08-04 after review of the original golf-ball paper, the current
Python experiments, modern fast-ball research, external golf tracking and
simulation repositories, and physical-iPhone testing. The observed tracker and
physics-template contract are frozen. The immediate product work is now the
mandatory address-ball tap and measured iPhone runtime optimization described
in Section 6. Earlier continuation tasks in this document are completed history.

## 1. Product goal

Build an offline-first iPhone app that accepts a golf-swing video, identifies
the same golf ball throughout its visible flight, draws a confidence-aware
tracer in the original video coordinates, and exports the rendered video.

The initial product target is a single golf shot recorded from behind the
golfer on a stable phone at 60 fps. Handheld footage, automatic initialization,
and live capture are later milestones, not assumptions of the first MVP.

### Current truth

- The runnable iOS app exists and the unsigned generic iOS build passes. Its
  native candidate pass currently combines Apple Vision trajectories with
  half-resolution three-frame motion components before the frozen Swift beam
  associator. Device testing found both incorrect automatic tracks and long
  output latency for clips under 10 seconds.
- Automatic ball/address initialization is no longer a product goal for the
  current MVP. Every video must require an address-ball tap before processing.
  The current code still exposes automatic setup, so removing that production
  path is immediate work, not a completed state.

- The legacy RANSAC tracer is wrong on every test clip. The replacement now
  runs annotation-free over complete clips from one explicit address-ball tap.
  It meets the measured path-coverage gate and contact-sheet visual QA on all
  three development clips and passed user visual review on four new untouched
  stable clips with frozen settings. Two held-out 3-wood clips were then
  densely scored: emitted-point precision is 100%, while the conservative
  evidence boundary stops 15/8 frames before the last weak cloud-blended
  labels. The tracker is accepted at that honest boundary.
- The existing YOLOv8n model reported strong image-split metrics, but the
  downloaded dataset split is contaminated by shifted/cropped siblings of the
  same source image across train, validation, and test. Treat those metrics as
  training history, not clean held-out evidence. Its measured sequence recall
  is poor on two of the three labeled clips.
- After correcting a 5–6-frame annotation seek offset, the first learned
  3-frame heatmap run scores 36.6% recall@3/5 on `curve-iron`. All its hits are
  already covered by motion candidates, so it adds no candidate recall.
- A smooth parabola is not proof that the fitted points belong to the ball.
- The current YOLO → greedy Kalman crop → motion filter → RANSAC pipeline is a
  legacy baseline. It is useful for comparison and debugging, not the target
  architecture.
- `golf_ball_reference/` and `paper.md` are historical references. They are not
  the source of truth for the new implementation.
- The first `automatic_cinematic` Hermite continuation is also now a rejected
  visual baseline. It preserves provenance correctly, but its 0.65-second apex
  floor, two-second duration, fixed image-height endpoint, and manufactured
  descent dominate the result. Do not tune those constants per clip.

## 2. Non-negotiable development principles

1. **Measure before tuning.** Every architecture change must be evaluated
   against frame-level ground truth, not judged only from a rendered video.
2. **Separate detection from association.** First ask whether the real ball is
   present among the candidates; then ask whether the tracker chose it.
3. **Benchmark the simplest temporal evidence first.** A tiny blurred ball is
   often ambiguous in one frame but identifiable from frame differences and
   adjacent-frame motion. A learned temporal network is justified only if
   simpler motion candidates and a specialized detector leave a measured gap.
4. **Do not commit greedily.** Preserve several candidate paths until later
   frames provide enough evidence to select the most likely complete flight.
5. **Add complexity when the footage requires it.** The current benchmark is
   tripod footage, so first verify that global motion is negligible. Add frame
   registration before differencing for handheld clips or whenever measured
   background motion is material.
6. **Render only from a confidence-qualified track.** Smoothing can improve
   presentation, but must never manufacture evidence or hide a tracking error.
7. **Optimize for iPhone only after correctness.** Desktop offline processing
   is the research environment; Core ML and live latency come after the
   architecture meets accuracy gates.

## 3. Target architecture

### Stage A — Decode and normalize the video

- Read frames with their real presentation timestamps; do not assume a file is
  exactly 60 fps or has no dropped/duplicated frames.
- Preserve original resolution and orientation metadata.
- Create lower-resolution copies only for background-motion estimation.
- Require one real swing per input clip for the first MVP.

### Stage B — User initialization and shot segmentation

- Ask the user to tap the ball at address.
- Use the tap as a spatial prior, not as a permanent fixed-size search crop.
- Detect an impact/launch interval from local motion around the address region,
  club motion, audio, or a combination of those signals.
- Model explicit states: `address`, `launch`, `airborne-visible`,
  `airborne-invisible`, `landed`, and `out-of-frame`.

### Stage C — Motion measurement and optional camera compensation

- Measure frame-to-frame background motion. For the current stable clips, use
  identity transforms if the residual is below a documented threshold.
- When needed, estimate global motion from background features using a robust
  affine transform or homography with RANSAC.
- Mask the golfer, club, address region, and other independently moving areas
  when estimating background motion.
- Accumulate transforms into stabilized coordinates for detection and motion
  analysis.
- Retain every transform so the final trajectory can be mapped back into the
  original, possibly handheld, frames.

Do not make registration a prerequisite for a tripod-only baseline. It adds
interpolation artifacts and failure modes when the correct transform is
effectively identity.

### Stage D — Staged candidate generation

Candidate generation is an evidence ladder, not a commitment to one network.
Every rung exports top-K centers in the same format and is evaluated before the
next rung is built.

1. **Motion-first baseline (completed and frozen):** use robust background/three-frame
   differencing, connected components, and permissive size, contrast, motion
   direction, displacement, and streak-orientation scoring. Preserve multiple
   candidates; do not use YOLO as a veto.
2. **Specialized crop detector:** retrain a small-object detector or center
   predictor using source-grouped splits and real iPhone patches. Search a
   corridor seeded by the address tap, with full-frame/tiled fallback for
   re-acquisition.
3. **Learned temporal fusion (conditional):** only after the first two baselines
   expose a residual recall problem and the training set contains sufficiently
   varied source videos. Fuse appearance with explicit motion rather than
   learning from three raw frames from scratch.

If rung 3 is reached, use a pretrained shared encoder, synchronized geometric
augmentation across the temporal window, realistic photometric/compression and
directional-blur augmentation, explicit visibility supervision, center-distance
validation metrics, and video/session-grouped splits. Five-frame context,
motion-attention, background estimation, and blur heads are ablations—not a
bundle of mandatory features.

Two candidate-generation modes must be benchmarked:

1. **Overlapping full-resolution tiles:** slow but high-recall correctness
   baseline that cannot lose the ball solely because a prior crop was wrong.
2. **Predicted search corridor:** faster path that prioritizes tiles around
   plausible motion while retaining periodic fallback tiles for re-acquisition.

YOLOv8n remains a single-frame baseline. Re-scored against corrected labels,
the implemented temporal heatmap run reaches 36.6% curve recall@5 but adds no
hit outside the motion candidates. Neither is presumed to be the winner.

### Stage E — Clip-level multi-hypothesis association

For each frame, retain the top-K candidate centers plus an explicit no-visible-ball
hypothesis. Build a time-indexed candidate graph and choose the best complete
path using dynamic programming, Viterbi decoding, beam search, or an equivalent
factor-graph formulation.

Score candidate paths using:

- detector/heatmap likelihood;
- agreement with the address tap and launch interval;
- displacement, velocity, and acceleration continuity using real timestamps;
- agreement between movement and predicted blur direction;
- motion-compensated flow or frame-difference evidence;
- penalties for sudden reversals or implausible jumps;
- a calibrated penalty for invisible frames;
- valid transitions between address, flight, landing, and out-of-frame states.

The tracker must keep multiple hypotheses long enough for later observations to
correct earlier ambiguity. A Kalman or extended Kalman filter may be used inside
a hypothesis or as a final smoother, but it must not greedily decide which
detection is the ball.

### Stage F — Confidence-aware trajectory smoothing

- Smooth the selected 2D path in stabilized image coordinates using a robust
  spline or state-space smoother.
- Preserve observed points, estimated points, visibility, and uncertainty as
  separate data; do not collapse them into one unexplained curve.
- Do not require the entire screen-space path to be a single parabola.
- Fill only short gaps supported by strong evidence on both sides.
- Stop or fade the trace when uncertainty becomes excessive or the ball leaves
  the frame.
- Full 3D ball-flight reconstruction is out of scope for the first MVP.

### Stage F2 — Optional inferred continuation

Run this only after Stage E has selected the track and Stage F has frozen the
observed path. It is a post-tracking presentation feature, never candidate
evidence and never input to association.

- Preserve three distinct provenances: `observed`, `estimated_gap`, and
  `inferred`. A long tail after evidence ends is always `inferred`, even when it
  looks physically plausible.
- The completed quadratic/kinematic withheld-tail benchmark remains diagnostic
  evidence for short horizons; neither model is approved for a ground-reaching
  tail.
- Propagate uncertainty after the last observation and reject or fade an
  automatic continuation when fit quality, forecast horizon, bounds, or
  uncertainty fail. Never feed inferred points back into detection, launch,
  termination, or tracking metrics.
- When neither camera nor user can see the landing, do not require a landing
  tap. Replace the rejected long Hermite tail with a fixed bank of physically
  simulated, normalized trajectories for `wood`, `iron`, and `wedge` profiles.
  At runtime, fit only low-dimensional time and screen-projection parameters to
  the frozen observed 2D track, anchor the join at the final observation, rank
  the fixed templates by reprojection/continuity error, and render the best
  plausible continuation with increasing uncertainty.
- Use the address point as the near-ground reference and add an optional
  horizon/distant-ground-line input for perspective. This is not a landing tap.
  A projected ground intersection remains cinematic unless camera pose and the
  ground plane are calibrated.
- Generate templates offline with a small, tested forward simulator containing
  gravity, wind-relative quadratic drag, Magnus lift, bounded source-documented
  coefficients, optional simple spin decay, and ground intersection. Do not
  infer a unique depth, launch state, spin, carry, or landing from the current
  center-only monocular tracks.
- Keep simulation, projection, template selection, and rendering separate. Do
  not feed physics predictions into candidate association or use a narrow hard
  reacquisition gate.
- Keep this module optional and schema-versioned so `observed_only` remains the
  factual default and tracker output is unchanged when inference is disabled.

See `INFERRED_TRAJECTORY.md` for the schema, quality gates, evaluation protocol,
and implementation order.

### Stage G — Rendering and export

- Transform the stabilized path back into each original frame.
- Render a configurable tracer whose head follows the current ball position and
  whose tail length, width, glow, color, and fade are presentation settings.
- Keep tracking data and rendering code independent so visual changes cannot
  alter the underlying track.
- Export using AVFoundation; Metal is optional until profiling shows it is
  required for rendering performance.

## 4. Dataset and evaluation plan

### Ground-truth format

Create one annotation record per frame containing at least:

- frame index and timestamp;
- ball center `(x, y)` when labelable;
- visibility: `visible`, `blurred`, `occluded`, `not_visible`, or
  `out_of_frame`;
- optional blur endpoints or orientation/length;
- shot state: address, flight, landed, or finished;
- an `uncertain` flag for genuinely ambiguous human labels.

Annotations must be stored in original pixel coordinates and versioned with the
video identity and checksum.

### Data sources

- The existing Roboflow dataset may be useful for single-frame pretraining, but
  its supplied split is invalid for model selection: shifted/cropped siblings
  from a common source image occur across train, validation, and test. Rebuild a
  source-grouped split before reporting validation or test metrics.
- The decisive dataset is densely labeled, real iPhone golf-shot video matching
  the intended camera angle, resolution, compression, and backgrounds.
- Split train/validation/test by source video and recording session, never by
  randomly separated frames from the same shot.
- Include stable and handheld footage, multiple clubs and shot shapes, sky and
  ground backgrounds, lighting variation, and hard-negative clips with no
  launched ball.

### Required metrics

Report metrics by clip and in aggregate:

- candidate recall@K on visible/blurred frames;
- visible-frame tracking recall;
- precision and false-track rate;
- median and 95th-percentile center error in pixels;
- longest contiguous correct run and percentage of visible flight covered;
- launch-frame and termination-frame error;
- accuracy by visibility/background category;
- processing time, peak memory, and model size after correctness is established.

Bounding-box mAP is a detector diagnostic, not the product acceptance metric.

### Definition of a correct tracer

A clip is correct only when the selected track follows the annotated ball over
the complete labelable flight, stays within the agreed center-error tolerance,
does not switch to clutter, starts at the correct launch, and stops/fades at the
correct visibility boundary. Visual plausibility alone never qualifies.

## 5. Roadmap and gates

### Phase 0 — Evaluation foundation

**Status:** complete for the three stable clips. The versioned schema, interactive
autosaving labeler, evaluator, and legacy diagnostics export were implemented
on 2026-08-04. All three clips now have QA-valid dense labels. A random-seek bug
that displayed frames 5–6 indexes later than the saved record was fixed by
sequential-only decoding; `curve-iron` was relabeled and targeted review now
contains 43 valid flight points over frames 245–287.
The straight/top fitted baselines score only 1/59 and 1/87 correct visible-flight
frames at 20px, so no tracer success claim changes.

**Deliverables**

- Annotation schema and labeling tool/workflow.
- QA-validated dense labels for the three active stable clips; `practice-iron`
  is an explicit deferred case rather than an unreported success.
- Annotation QA includes enlarged visual crops and a local temporal-change
  sanity check; filled records alone do not qualify a clip as complete.
- Reusable evaluator and per-clip report.
- Archived baseline results from the current pipeline.

**Exit gate:** every tracker run can be scored automatically and failures can be
attributed to candidate generation, association, or smoothing.

### Phase 1 — Candidate-generation benchmark

**Status:** candidate-coverage gate met on the three stable clips. The frozen YOLOv8n model was evaluated on 2026-08-04
using eight overlapping 640px tiles per portrait frame, 25% overlap, a 0.01
confidence threshold, cross-tile NMS, and the top five surviving detections.
This removes the legacy Kalman crop as a failure source. The corrected motion
extractor retains 100 candidates per frame and reaches 100%, 98.3%, and 100%
recall at 20px on curve, straight, and top respectively, with 3.3–4.4px median
localization error. Union with tiled YOLO recovers the one missing straight
frame, yielding 100% oracle candidate coverage on all three clips. The ball can
rank as low as 68 among 1,200–2,000 raw components, so confidence-only top-5 is
not sufficient; clip-level association is now the measured bottleneck.

**Deliverables**

- Run the current YOLO model over overlapping high-resolution tiles without a
  greedy crop dependency.
- Record top-K candidates and heatmaps independently of rendering.
- Benchmark a no-training motion-first candidate generator on the stable clips.
- Rebuild the single-frame dataset split by original source group and report the
  detector honestly; add real iPhone crops before retraining if possible.
- Treat learned temporal fusion as a conditional later experiment, not the next
  mandatory model.
- Compare all candidates on the same held-out video split.

**Exit gate:** the real ball appears in the candidate set for nearly all
labelable flight frames, with failures categorized by visibility and background.
Set the numerical threshold after measuring human-label uncertainty; do not pick
an arbitrary product target before the evaluator exists.

### Phase 2 — Global tracking and camera compensation

**Status:** annotation-free clip-level association and QA rendering are
implemented for stable tripod clips. The three development clips meet their
measured gates; four untouched stable clips passed frozen-setting visual review,
and two subsequently labeled 3-wood clips have 100% emitted-point precision
with conservative early termination documented. Annotation dependence has been
removed from the production call graph. The observed tracker is now frozen
while the separate inferred-continuation contract is prototyped.

**Deliverables**

- Background-motion measurement and, where warranted, a stabilized coordinate
  system.
- Candidate graph with invisible states and clip-level path optimization.
- Ablations for detector score, motion cue, physical continuity, and camera
  compensation.
- Quantitative results on all stable clips.

**Exit gate:** all existing stable clips satisfy the agreed definition of a
correct tracer, with no manual choice of intermediate detections.

### Phase 2.5 — Inferred-continuation contract and prototype

**Status:** complete and frozen. Schema v2, backward-compatible upgrade, fixed
short-horizon baselines, withheld-tail evaluation, optional assisted editing,
and provenance-aware rendering are implemented. The rejected Hermite baseline
remains for A/B only. Its accepted replacement uses a tested forward simulator,
an 18-flight fixed club-profile bank, bounded projective fitting, explicit
uncertainty, and unmeasured projected-ground metadata. All 44 tests pass and the
shared-setting wood/wood/wedge visual transfer gate passed on 2026-08-04.

**Deliverables**

- Versioned observed/estimated/inferred trajectory schema and renderer contract.
- Offline post-tracking prototype operating only on exported track JSON.
- Withheld-tail benchmark on dense labels, comparing a small fixed set of
  simple baselines and reporting error versus prediction horizon.
- `observed_only`, conservative automatic cinematic, and assisted-landing
  modes, with uncertainty-aware fade and safe refusal.

**Exit gate:** disabling inference reproduces the approved tracker output;
inferred points cannot enter tracker metrics; the chosen baseline wins the
predeclared withheld-tail comparison; assisted output is visually reviewed on
the approved stable clips. This gate validates presentation, not true invisible
landing accuracy.

### Phase 3 — Robustness and handheld footage

**Status:** deferred by current product decision. Do not begin this phase before
annotation-free stable-camera inference, untouched stable-clip validation, and
offline performance work are complete.

**Deliverables**

- At least two true handheld test clips plus broader iPhone footage.
- Re-acquisition/fallback search behavior.
- Hard-negative, blur, occlusion, and camera-motion augmentation.
- Confidence calibration and stop/fade behavior.

**Exit gate:** stable and handheld held-out clips meet the same tracking criteria
without clip-specific parameter tuning.

### Phase 4 — Offline iOS MVP

**Status:** interface and media path implemented, but the exit gate is not met.
The checked-in workflow still allows automatic address initialization and must
be changed to require a tap. Physical-device completion time is unacceptable on
videos under 10 seconds and must be profiled and optimized.

**Deliverables**

- AVAssetReader frame/timestamp pipeline.
- Core ML conversion of the winning learned candidate model, if one is needed.
- Swift implementation of camera transforms, candidate association, smoothing,
  rendering, progress reporting, cancellation, and export.
- Tap-to-initialize interface and rendered preview.

**Exit gate:** every video requires a correctly mapped address tap; iPhone
output preserves the accepted association/provenance behavior; and measured
same-clip runtime reaches a user-acceptable target without breaking cancel,
orientation, audio, preview, save, or share. Record the target and results rather
than declaring performance acceptable qualitatively.

### Phase 5 — Capture-quality controls

**Deliverables**

- Device-format selection with fixed 60 fps frame-duration bounds.
- Disable automatic frame-rate reduction where supported.
- Lock focus and white balance after setup.
- Empirically select exposure limits that balance motion blur and sensor noise.
- Compare stabilization modes and preserve actual timestamps.
- On-screen framing, lighting, and tripod guidance.

**Exit gate:** app-recorded footage is at least as trackable as imported benchmark
footage across supported devices and lighting conditions.

### Phase 6 — Live/low-latency mode

Only begin after the offline MVP is correct. Replace full-clip processing with
causal candidate generation, bounded beam search, and a short revisable latency
window. A causal temporal model is optional and depends on the offline winner.
Profile ANE/GPU/CPU placement, thermals, dropped frames, and memory before
choosing final resolution or model size.

## 6. Immediate next steps

### Current iOS priorities (supersede the historical ledger below)

1. Make the address-ball tap mandatory after video selection and before the
   process action becomes available. Remove automatic-address UI and production
   calls to `AutomaticShotSetup.initialize`; retain original-coordinate mapping
   and use the tap as the association anchor.
2. Instrument the physical-device pipeline with stage timings and peak-memory
   observations. Measure sequential decode, vImage motion preprocessing,
   connected components, Vision, candidate merge, association, continuation,
   and export separately on the same sub-10-second clips.
3. Optimize only the measured bottleneck. The tap may safely seed an early
   region/corridor, but re-acquisition behavior must remain bounded and tested.
   Preserve exact timestamps, cancellation, source orientation, audio, schema
   provenance, prediction gates, and the accepted association behavior.
4. Report before/after total and stage timing on a physical iPhone, then verify
   tap accuracy, correct/observed-only output, cancel, preview, audio, save, and
   share. Do not start new model training, automatic initialization work,
   handheld support, live mode, or per-clip tuning.

Performance acceptance must use a Release build on the physical iPhone, not a
Debug/simulator result. Use a small fixed clip set, fixed tap coordinates, and
record resolution/fps/frame count, total wall time, stage times, and trajectory
summary. Instruments and unified signposts are preferred for locating CPU,
allocation, and threading bottlenecks. An optimization that changes selected
trajectory frames or coordinates is not “equivalent” until the relevant parity
and visual gates pass.

The numbered list below is the completed research/implementation ledger. Items
that say “next” or “in progress” describe their historical moment and do not
override the priorities above.

1. **Completed:** define the versioned annotation schema, build the autosaving
   labeling workflow, implement the evaluator, and initialize all four clips
   from exact presentation timestamps.
2. **Completed:** dense annotations for `curve-iron`, `straight-wedge`, and
   `top-iron` pass visual/pixel QA. `practice-iron` is skipped.
3. **Completed:** run and score the frozen legacy trajectory and
   accepted-candidate series for all three benchmark clips.
4. **Completed:** implement top-K candidate-only diagnostics with overlapping
   full-frame tiles and cross-tile duplicate suppression.
5. **Completed on QA-valid labels:** benchmark frozen YOLO. Recall@5 is 29.3%
   on `curve-iron`, 50.8% on `straight-wedge`, and 10.3% on `top-iron`.
6. **Completed:** build and smoke-test the deterministic temporal training-data
   pipeline, three leave-one-video-out manifests, a 3-frame stride-four
   center-heatmap plus visibility model, and evaluator-compatible candidate
   export. The production training path uses a reusable exact uint8 memory map,
   live batch/ETA and accelerator-memory reporting, measured loader timing, and
   resumable atomic checkpoints; repeated random MOV seeking is not used. No
   accuracy claim is attached to the untrained smoke checkpoint.
7. **Completed and corrected:** train the first leave-one-video-out fold with
   `curve-iron` held out. Despite selecting epoch 28 by validation loss, flight
   recall@1/3/5 was 0/0/0% at 20px. Sampled training tiles achieved 45/60
   correct centers with 4px median error, while held-out curve tiles achieved
   0/61 with 161.8px median error under the original bad labels. Re-scoring the
   saved candidates after relabeling gives 36.6% recall@3/5. Those hits are a
   subset of motion hits, so this checkpoint adds no candidate coverage.
8. **Completed:** implement and score motion-only candidates. At top-100, recall
   is 100% (`curve-iron`), 98.3% (`straight-wedge`), and 100% (`top-iron`). The
   tiled detector supplies the one missing straight frame, making union coverage
   100% on every stable clip. Processing currently costs about 440–466ms/frame.
9. **Completed:** fix exact-frame annotation decoding, relabel `curve-iron`, and
   pass structural, temporal-change, and enlarged-crop QA.
10. **Parallel detector-data task:** rebuild/audit the Roboflow split by original source group.
   Do not call its current validation/test metrics held out. Then decide whether
   a specialized crop detector needs retraining with real iPhone patches.
11. **Completed:** implement the beam-search candidate graph, gap/terminated
   states, conservative continuation evidence, and separately marked bounded
   gap interpolation. The same settings reach 100% visible recall on all three.
12. **Completed:** targeted full-resolution annotation review. Curve 283–284 and
   top 189, 192, 197, 206, 211, and 222 are labelable ball points; top 235 stays
   occluded and top 144 remains a genuine club-head occlusion.
13. **Completed:** final metrics, association-backed QA renders, and user visual
   approval. Reconstructed-path recall is 100% on all three clips within the
   annotation-defined flight windows.
14. **Completed:** remove annotation dependence from inference. Complete-clip
   candidates, explicit tap input, automatic launch, and evidence-based
   termination now run without annotation imports.
15. **Completed:** rerun, score, render, and visually inspect all three stable
   clips. Reconstructed recall is 100% on all three; launch error is zero;
   termination error is 0/0/+1.
16. **Completed visual gate:** the frozen configuration passed user review on
   four new untouched tripod/behind-golfer clips without per-clip tuning.
   **Completed quantitative subset:** two 3-wood clips have 100% emitted-point
   precision with 1.30–1.57px median error, while conservative termination is
   15/8 frames early on weak cloud-blended observations. This is accepted as
   the factual observation boundary and must not be tuned away.
17. **Completed:** trajectory schema v2, isolated inferred-tail prototype, two
   fixed baselines, withheld-tail report, constrained piecewise-cubic
   completion, and distinct renderer support. The 36-test suite passes.
18. **Completed but rejected visual baseline:** the shared Hermite cinematic
   completion was rendered on two 3-woods and one wedge. The user found the
   invisible descent unrealistic. All three clips hit the same 0.65-second apex
   floor; the result is constraint-driven rather than shot-driven. Retain it
   only for A/B comparison and do not fine-tune it.
19. **Completed and accepted:** the Stage F2 physics-template replacement has a
   tested forward simulator, fixed wood/iron/wedge bank, bounded 2D fitter,
   optional horizon metadata, schema fields, uncertainty fade, and three-clip
   shared-setting visual approval. The continuation contract is frozen.
20. **Partially completed; superseded by current iPhone profiling:** exact-equivalence runtime work reduced top-iron motion
   extraction from 40.47s to 17.30s (57.2%) with all 29,091 candidates and the
   downstream track unchanged. Five Python-to-iOS parity cases are frozen.
   `ShotTracerCore` now includes schema/candidate models, multi-source merging,
   the full-clip beam association, frozen physics simulator/template bank,
   screen-projection continuation fitter, and a sequential-only `AVAssetReader`
   pipeline. Swift association exactly matches the frozen Python motion-only and
   combined tracks; Swift continuation matches all three accepted physics
   artifacts within manifest tolerances. The provenance renderer and sequential
   AVFoundation exporter now preserve portrait presentation geometry, source
   timing, optional audio, progress, cancellation, and overwrite safety.
   The offline UI and native Vision-plus-motion candidate pass now exist, with
   21 active Swift tests and a successful unsigned generic iOS build. Physical-
   device runtime remains too slow and is the current optimization target.
21. Handheld footage and camera-motion compensation are explicitly deferred.
   If a future stable clip exposes a candidate-recall gap, reconsider a corrected
   specialized detector, source-grouped data expansion, or learned temporal
   fusion. Do not add model complexity while motion plus association is sufficient.

Do not spend the next iteration training the unchanged temporal model, tuning
the legacy RANSAC/Kalman path, tuning the rejected Hermite constants per clip,
exporting Core ML, or building live-camera UI. Implement only the bounded
physics-template work in `INFERRED_TRAJECTORY.md`; it is a cinematic prior, not
a launch monitor or a unique monocular 3D reconstruction.

## 7. Product fallback and observation boundary

- A reliable semi-manual MVP is legitimate: the user can mark impact, apex,
  landing, or a few keyframes and the app can fit and render the tracer. Keep
  this fallback available even if automatic tracking improves.
- The current 60fps clips contain only 37–87 human-labelable flight frames.
  After the ball becomes unobservable, a single camera cannot truthfully
  continue tracking it. Any later segment is an inferred visualization and must
  be represented separately from observed and short-gap-estimated positions.
- Full-flight physical reconstruction, carry, launch, and spin require stronger
  assumptions or additional sensors/cameras and remain outside the first MVP.
- Therefore inferred continuation belongs after tracking/smoothing and before
  rendering, runtime optimization, and iOS packaging. It must never be used to
  make the tracker appear to have observed more frames than it did.

See `RESEARCH.md` for the method survey, primary sources, local evidence, and
decision rationale behind this staged plan.
