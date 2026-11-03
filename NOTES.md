# Project Notes

Agent continuation starts in `HANDOFF.md`; it contains the current evidence,
approved artifacts, exact next implementation, and acceptance gates.

Operational details and decisions that supplement `Master Plan.md`. The Master
Plan is the source of truth for architecture, phase gates, and next steps.

## Current status

- **Completed 2026-08-05 (device-feedback pass):** user reported, from an
  actual device run, that the setup-view thumbnail sometimes showed a frame
  well after time zero (ball already in flight), a ~2-3 minute processing
  time on a 6-second clip, and a track that "focused on another object."
  Fixed the thumbnail bug: `AVAssetImageGenerator` was called without
  `requestedTimeToleranceBefore`/`After`, so it was free to snap to the
  nearest sync sample instead of decoding frame zero exactly; both are now
  `.zero` in `ShotTracerWorkflow.thumbnail(for:)`. Also reworded the
  `TrajectoryVideoExportError.writerStalled` message, which mentioned
  "non-ProRes" — the picker/import path (`PhotosPicker(matching: .videos)`,
  `FileRepresentation(importedContentType: .movie)`) already accepts any
  video format; there is no format allowlist. The message only ever fires
  after a 30-second `AVAssetWriter` backpressure timeout, which ProRes
  sources are more likely to hit because they are far more expensive to
  decode/re-encode; the reworded copy says formats are accepted rather than
  implying a restriction. 21 active Swift tests and the package build still
  pass. **Not yet diagnosed:** whether the mis-tracked run was on a
  4K-recorded clip (which the 2026-08-04 coordinate-mapping fix already
  addresses) or a genuine association failure, and where the 2-3 minute
  processing time actually goes — both need the `performance`/`tracking`
  unified-log lines from an actual device run, which this environment cannot
  produce. See `HANDOFF.md` for exactly what to ask the user for next.
- **Completed 2026-08-04:** the address-ball tap is now mandatory in the
  production iOS workflow. `ShotProcessingRequest.manualAddress` and
  `VisionShotDetector.detect(manualAddress:)` are non-optional `CGPoint`;
  `AutomaticShotSetup.initialize` is no longer called from any production
  path (it remains only as research code exercised by its own unit tests via
  `proposeAddress`/`selectProfile`). `ShotTracerAppView` always shows the tap
  surface, removed the "Automatic setup" copy and "Use automatic ball
  position" control, and disables "Trace Shot" until `manualAddress` is set.
  `ShotProcessingOutput.usedAutomaticAddress` was removed since it was always
  `false`. 21 active Swift tests and the unsigned generic iOS build both pass
  after this change.
- **Coordinate-mapping bug found and fixed during that work:** the setup-view
  thumbnail was generated with `AVAssetImageGenerator.maximumSize = 1080x1920`,
  but `SequentialVideoFrameReader`/the exporter decode at the source's full
  `AVVideoComposition.renderSize`. For any clip recorded above 1080x1920
  (e.g. 4K 60fps rear camera, a common iPhone default), the address tap was
  captured in the downscaled thumbnail's pixel space and then used unscaled
  as a full-resolution coordinate — silently anchoring association roughly at
  2x the intended pixel position. This is a strong candidate for the reported
  "ball not detected" behavior and has been present since automatic setup was
  the default path (it only bites the manual tap once that path is mandatory,
  which is likely why it wasn't caught earlier). Fix: `thumbnail(for:)` in
  `ShotTracerWorkflow.swift` no longer caps `maximumSize`, so the preview
  image's pixel dimensions equal the decode-resolution coordinate space by
  construction. Verify this explicitly on a physical device with a 4K-recorded
  clip before considering ball-detection issues further diagnosed.
- **Completed 2026-08-05:** stage-level timing instrumentation is in place.
  `VisionShotDetector.detect` now measures `decode` (time inside
  `reader.nextFrame()`), `vimage_downscale_change` and `connected_components`
  (split out of `ThreeFrameMotionCandidateExtractor.consume`, which now
  exposes `vImageDownscaleChangeSeconds`/`connectedComponentsSeconds`),
  `vision_requests` (`VNImageRequestHandler.perform`), `candidate_merge`
  (`makeCandidates`), and `association`
  (`TrajectoryAssociator.associateFullClip`), returned as
  `VisionShotDetectionResult.stageTimings`. `OfflineShotProcessor.process`
  adds `continuation` (profile/template selection) and `export`
  (`TrajectoryVideoExporter.export`) timing, then logs one line via
  `Logger(subsystem: "com.caio.ShotTracer", category: "performance")` at
  `.notice` with resolution/fps/frame count, every stage in seconds, and
  `wall_total`. New shared helpers: `StageTimings` (additive/mergeable named
  durations) and `Duration.secondsDouble`, both in `ShotTracerCore`.
  21 active Swift tests and the unsigned generic iOS build still pass.
  **Physical-device profiling itself is still pending** — this environment
  cannot run a Release build on real hardware. The next step is for the user
  to run the app on their iPhone (Release configuration) on a small fixed set
  of sub-10-second clips, open Console.app filtered to subsystem
  `com.caio.ShotTracer` / category `performance` (or attach Instruments Time
  Profiler), and report back the logged stage timings plus which stage
  dominates. Do not guess at the bottleneck or optimize anything before that
  measurement comes back, per the HANDOFF stop condition.
- Current native iOS detection combines Apple Vision candidates with a bounded
  half-resolution three-frame motion source in one sequential pass, then uses
  the frozen Swift associator. There are 21 passing active Swift tests, 47
  passing Python tests, two device-only media tests, and a successful unsigned
  generic iOS build against the installed iOS 26.5 SDK.

- Annotation-free full-clip inference is implemented in
  `ml/scripts/run_shot_tracer.py`. It accepts numeric tap coordinates, probes
  exact timestamps, sequentially extracts full-clip motion candidates, detects
  launch and termination, optionally adds tiled YOLO over an automatic window
  plus periodic fallback, exports JSON artifacts, and renders sequentially.
- The shared configuration recovers 100% of visible/blurred path points within
  20px on curve, straight, and top. Launch error is 0 on all three; termination
  error is 0/0/+1. Median errors remain 3.22/3.10/4.00px. Contact-sheet visual
  QA passes. The expanded suite now has 36 passing tests.
- An unchanged exploratory run on unscored `practice-iron` ignored the early
  practice swing and selected the later real swing at frames 710–745. Visual QA
  passes, but no accuracy claim is made because this clip lacks dense labels.
  A spatial-grid candidate merge replaced the equivalent quadratic scan after
  profiling exposed it as the long-clip runtime bottleneck; equivalence is
  covered by a regression test.
- Four untouched stable clips (`yellow-iron`, `right-iron`, `left-3wood`, and
  `curve-3wood`) were run once with frozen shared settings and explicit taps.
  The user visually approved all four annotation-free videos on 2026-08-04.
  They remain unscored until a representative subset receives dense labels.
- Held-out scoring is complete for `left-3wood` and `curve-3wood`. Every emitted
  point is correct at 20px and median errors are 1.30/1.57px. Recall is
  66.67/88.10% because evidence-based termination occurs 15/8 frames before the
  final human-labelable, cloud-blended ball. The user approved both videos; this
  is accepted as a conservative visibility-boundary limitation, not tuned away.
- The observed tracker and schema v2 are accepted and frozen. The first long
  `automatic_cinematic` Hermite tail was rendered on two 3-woods and one wedge,
  but the user rejected its invisible descent as unrealistic. Inspection shows
  every clip hit the same 0.65-second apex floor and arbitrary ground-band rules
  dominated the output. Do not tune those constants per clip.
- **Historical (completed):** the next implementation at that time was the bounded `physics_template` replacement in
  `INFERRED_TRAJECTORY.md`: a tested forward simulator, a small fixed
  wood/iron/wedge template bank, and low-dimensional 2D time/projection fitting.
  It remains post-tracking and inferred. Runtime optimization waits until this
  one replacement visual gate is complete. That gate has passed and does not
  supersede the current iOS priorities above.
- The old `*_traced.mp4` RANSAC outputs remain incorrect. The newer
  `*.associated-tracer.mp4` files are historical QA renders; the approved
  `*.annotation-free.mp4` outputs use complete-clip inference with no annotated
  flight boundaries in the production call graph.
- `practice-iron.MOV` remains outside the scored set but now has an unscored,
  visually reviewed annotation-free output.
- All four clips used so far are stable/tripod-style. Handheld iPhone evaluation
  and camera-motion compensation were previously planned but are now explicitly
  deferred.
- `practice-iron.MOV` likely contains a practice swing followed by a real swing.
  The first MVP assumes exactly one real swing per input clip.
- Phase 0 tooling now exists: versioned dense annotations, an interactive
  autosaving labeler, a metrics evaluator, and legacy tracker diagnostics JSON.
- Empty annotation documents have been initialized from exact stream timestamps:
  `curve-iron` 519 frames, `straight-wedge` 547, `top-iron` 294, and
  `practice-iron` 1103. OpenCV's container metadata overstated the decodable
  frame count for every clip, so annotations use `ffprobe` timestamps instead.
- `curve-iron.MOV` is correctly relabeled: flight frames 245–439 with 43
  visible/blurred points over frames 245–287. Targeted review added the real
  ball at frames 283–284; frame 283 is a weak blurred observation bridged by the
  association path.
- Root cause found after the first relabel attempt: OpenCV reported successful
  random seeks while displaying frames 5–6 indexes later than requested. The
  annotator saved correct clicks under the wrong frame numbers. The reader now
  always decodes sequentially and was verified at frames 230, 240, 250, 270,
  and 281 with zero pixel difference from an independent sequential decode.
- Corrected `curve-iron` legacy scoring: the fitted trajectory remains 0/41
  correct with 991.5px median error and 100% false-track rate. Accepted legacy
  candidates are accurate when present (12/41 correct, 4.8px median error) but
  sparse; greedy tracking still fails.
- `straight-wedge.MOV` annotation and frozen baseline are complete. All 547
  frames are labeled; flight is 247–495 with 59 visible frames (247–305). At
  20px tolerance the fitted trajectory and accepted candidates each achieved
  only 1/59 correct visible frames. The sole correct candidate is the launch
  frame; subsequent outputs are hundreds of pixels from the ball. The fitted
  trajectory's median error is 912.67px and false-track rate is 99.598%.
- `top-iron.MOV` annotation and frozen baseline are complete. All 294 frames are
  labeled; flight is 141–274 with 93 visible/blurred frames after targeted
  review. Accepted legacy YOLO
  outputs are accurate when present (8/10 correct overlapping outputs, 7.05px
  median error) but cover only 9.195% of visible frames. The fitted trajectory
  is still nearly entirely false: 1/87 correct and 1201.31px median error.
- `practice-iron.MOV` is intentionally skipped for the initial benchmark. The
  QA-valid evaluation clips are `curve-iron`, `straight-wedge`, and `top-iron`.
- The overlapping tiled YOLO benchmark is complete. It used 640px tiles with
  25% overlap, eight tiles per frame, confidence 0.01, cross-tile NMS, and top
  five retention over every annotated flight frame. At 20px tolerance,
  recall@1/3/5 was 19.5/29.3/29.3% for corrected `curve-iron`,
  50.8/50.8/50.8% for `straight-wedge`, and 6.9/10.3/10.3% for `top-iron`.
  Mean batched inference time was 156–170ms per flight frame on MPS.
- Removing the legacy crop therefore helps one clip but does not solve
  candidate coverage. Extra top-K detections add almost no recall.
- The first temporal-model foundation is implemented. Deterministic manifests
  use three adjacent RGB frames, the same eight overlapping 640px tiles used at
  inference, tile-level positive and negative visibility targets, and three
  leave-one-video-out folds. The folds contain 982–1098 training examples and
  461–577 validation examples with no shot shared across splits.
- A compact stride-four heatmap plus visibility network, lazy video dataset,
  training command, checkpoint format, and full-flight top-K candidate exporter
  now pass an end-to-end CPU smoke test. This confirms plumbing only; the smoke
  checkpoint is untrained and has no accuracy claim.
- The original lazy MOV path was rejected for real training after it showed no
  batch-level progress and repeatedly sought compressed video frames, starving
  the accelerator. Training now uses a one-time exact uint8 memory-mapped tile
  cache (3.45GiB), batch progress/ETA, separate data and compute latency,
  throughput, MPS memory reporting, persistent workers when enabled, and atomic
  best/last checkpoints. The cache is generated and reused across all folds.
- The first real leave-one-video-out model (`curve-iron` held out) completed 30
  epochs; epoch 28 was selected at 0.1053 validation loss. Quantitative flight
  evaluation nevertheless scored 0% recall@1/3/5 at 20px and 100% false-track
  rate. Direct tile diagnostics found 0/61 held-out positive tiles within 20px
  (161.8px median error), while 45/60 sampled training positives were within
  20px (4px median error). Re-scoring its saved curve candidates after fixing
  the labels yields 36.6% recall@3/5, all redundant with motion candidates.
- A dataset audit found that the supplied Roboflow train/validation/test split
  leaks shifted/cropped siblings of the same source image across splits. For
  example, `00003_00001`, `00003_00003`, and `00003_00004` appear in train,
  validation, and test respectively and depict the same underlying scene.
  Source-prefix overlap affects 347/348 validation groups and 197/198 test
  groups. The old image metrics are therefore not clean held-out estimates.
- Motion top-100 recall is 100% (`curve`), 98.3% (`straight`), and 100% (`top`),
  with 3.3–4.4px median localization error. Adding tiled YOLO recovers the one
  missing straight frame, so union coverage is 100% on all three. The immediate
  engineering task is clip-level association; the separate detector-data task
  remains a source-grouped split. See `RESEARCH.md`.
- Clip-level beam-search association is implemented with the same configuration
  for all clips. After targeted annotation review, reconstructed-path recall is
  100% on curve, straight, and top, with median errors of 3.22px, 3.10px, and
  4.00px. Observation recall/precision is 97.67/100% on curve, 100/100% on
  straight, and 100/98.94% on top. Curve frame 283 is a bounded estimate. Top
  frame 144 is a genuine club-head occlusion and remains a bounded estimate;
  frame 235 remains human-labeled occluded, so top termination is one frame late.
- `render_associated_track.py` exports persistent cyan observed paths and amber
  dashed bounded estimates. The three reviewed QA videos are in `ml/output/`.

## Architecture decisions

- The current YOLOv8n → Kalman-guided crop → frame-difference filter → RANSAC
  parabola pipeline remains only as the legacy baseline.
- Candidate generation follows a measured ladder: motion-first components,
  then a corrected specialized crop detector, then learned temporal fusion only
  if the simpler signals leave a documented recall gap.
- Candidate generation and candidate association are separate stages with
  separate metrics.
- Tracking retains the top-K candidates and an invisible state per frame, then
  selects a complete path using clip-level dynamic programming or beam search.
- Camera motion is measured first. Registration is used when background motion
  is material; identity is the preferred transform for verified tripod footage.
- Re-acquisition is in scope. A wrong prediction, missed detection, or large
  post-impact displacement must not permanently exclude the real ball.
- A Kalman filter may smooth a chosen path or maintain state inside each
  hypothesis; it must not greedily select the only candidate path.
- A single screen-space parabola is not required. Trajectory smoothing happens
  only after a ball path is selected and confidence-qualified.
- Full 3D reconstruction and shot metrics such as carry, spin, and launch angle
  are out of scope for the first MVP.
- Long continuation after the last defensible observation is presentation
  inference, not tracking. It runs after association/smoothing, never feeds back
  into tracking, carries explicit provenance and uncertainty, and is excluded
  from tracking metrics. Because the landing is also invisible to the user, the
  product does not require a landing tap; optional adjustment is aesthetic only.
- Trajectory schema v2, the two fixed forecast baselines, withheld-tail
  evaluator, assisted Hermite curve, and provenance-aware renderer are now
  implemented. All 36 tests pass. Across five dense clips, the kinematic model
  has an 8.06px median of case medians but unstable long-tail outliers; the
  long-tail completion does not use it as physical prediction. One constrained
  two-segment cubic now preserves measured terminal velocity, delays the apex
  for still-rising tracks, and supplies a shallower explicitly inferred visual
  completion over 2 seconds. See
  `ml/output/trajectory-withheld-tail.evaluation.json`.
- Correctness is established with dense labels and metrics, never only by
  watching the rendered trace.

## Environment

- Python code lives in `ml/`.
- The virtual environment is `ml/.venv` and is gitignored.
- No `.env` file is used.
- `ROBOFLOW_API_KEY` is read from the shell environment by
  `ml/scripts/download_roboflow.py`.
- `train.py` forces `device='mps'` because automatic device selection fell back
  to CPU on this machine's PyTorch/Python configuration.

```bash
# Existing legacy environment
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/requirements.txt

# Existing single-frame dataset/model workflow
export ROBOFLOW_API_KEY=your-key-here
ml/.venv/bin/python ml/scripts/download_roboflow.py
ml/.venv/bin/python ml/scripts/train.py --epochs 100

# Existing legacy tracker baseline
ml/.venv/bin/python ml/scripts/track_video.py path/to/swing.mp4
```

These commands reproduce the older detector/tracker baseline. The annotation,
evaluation, and archived temporal-baseline code exist separately; the commands
above do not run them or implement the planned motion candidate generator and
multi-hypothesis tracker.

## Existing model

The epoch-100 YOLOv8n weights from 2026-08-03 reported precision 96.7%, recall
92.8%, mAP@0.5 98.5%, and mAP@0.5–0.95 67.7% on the supplied image split, at
approximately 2.9 ms/image on the M5.

That split contains source-image sibling leakage, so those numbers are retained
as detector-training history only, not held-out performance. They also do not
measure candidate recall on complete iPhone videos, identity consistency,
flight coverage, false-track rate, or tracer correctness. The Roboflow dataset
is useful for pretraining but is not sufficient validation for the product.

## Current code map

- `ml/scripts/download_roboflow.py` — downloads the existing single-frame
  Roboflow dataset.
- `ml/scripts/train.py` — trains the existing YOLOv8n detector.
- `ml/scripts/track_video.py` — runs and renders the legacy two-pass tracker.
- `ml/scripts/annotate_video.py` — interactive dense labeler; scales the display
  while saving coordinates in original pixels and autosaves every edit. A
  right-click opens a full-resolution crop so tiny balls can be refined before
  left-clicking; `z` returns to the full-frame view. Navigation always decodes
  sequentially because OpenCV's apparently successful random seeks return these
  iPhone MOV frames 5–6 indexes late.
- `ml/scripts/evaluate_tracking.py` — scores either the legacy trajectory or
  candidate series against dense annotations.
- `ml/scripts/detect_tiled_candidates.py` — runs batched overlapping full-frame
  YOLO tiles, suppresses cross-tile duplicates, and exports top-K candidates
  without a tracker crop.
- `ml/scripts/detect_motion_candidates.py` — exports no-training top-K
  candidates from three-frame and median-background differences.
- `ml/scripts/associate_candidates.py` — merges candidate sources and selects a
  clip-level path with launch priors, motion continuity, beam search, explicit
  gaps/termination, and separately marked short-gap estimates.
- `ml/scripts/render_associated_track.py` — sequentially decodes iPhone MOVs and
  renders association paths; cyan is observed and amber dashed is estimated.
- `ml/scripts/build_temporal_manifests.py` — creates deterministic whole-video
  cross-validation folds from dense annotations.
- `ml/scripts/build_temporal_cache.py` — sequentially decodes the source videos
  once and builds the exact indexed uint8 tile memory map used for training.
- `ml/scripts/train_temporal_heatmap.py` — trains the 3-frame heatmap and
  tile-visibility baseline.
- `ml/scripts/detect_temporal_candidates.py` — runs a temporal checkpoint over
  every flight-frame tile and exports evaluator-compatible top-K candidates.
- `ml/temporal/` — temporal manifest, dataset, target-generation, model, and loss
  implementation.
- `ml/tracker/annotation_io.py` — schema creation, exact timestamp extraction,
  validation, hashing, and atomic JSON writes.
- `ml/annotations/annotation.schema.json` — machine-readable annotation schema.
- `ml/annotations/*.annotations.json` — initialized per-video dense label files.
- `ml/tracker/kalman.py` — legacy constant-velocity Kalman filter.
- `ml/TRACKING.md` — detailed legacy debugging history, corrected current
  status, and migration notes.
- `paper.md` — archived 2020-era source paper; useful background, not the target
  system design.
- `golf_ball_reference/` — outdated paper-author repository; do not use it as
  the architecture source of truth.

## Evaluation dataset requirements

The next dataset must contain dense frame-level labels in original pixel
coordinates:

- ball center when labelable;
- visible/blurred/occluded/not-visible/out-of-frame status;
- launch, flight, landing, and finished states;
- optional blur orientation and length;
- explicit uncertainty for ambiguous human labels.

Split data by complete video and recording session. Never put neighboring
frames from the same shot in different train/test splits.

## Immediate work queue

1. **Completed:** build the deterministic 3-frame tiled example generator and
   whole-video folds, with automated leakage tests.
2. **Completed:** implement and smoke-test the heatmap/visibility model,
   training loop, checkpoint, and candidate export path.
3. **Corrected result:** the first temporal fold scores 36.6% recall@3/5 after
   relabeling curve; it contributes no hits outside the motion candidate set.
4. **Completed:** motion top-100 plus tiled YOLO has 100% candidate coverage on
   all three stable clips.
5. **Completed:** implement clip-level association with invisible and terminated
   states; its reconstructed path covers every labeled visible flight frame.
6. Rebuild the Roboflow split by source-image group and rerun detector metrics;
   add real labeled iPhone crop data before relying on detector retraining.
7. **Completed:** targeted annotation review, corrected metrics, and three
   association-backed QA renders. Automated checks and final-frame visual QA
   pass.
8. **Completed:** human visual approval of all three
   `*.associated-tracer.mp4` files.
9. **Completed:** build and verify one annotation-free inference command.
   Candidate extraction covers the complete clip; inference receives one
   user-equivalent tap and does not read dense points or flight boundaries.
10. **Completed:** frozen settings passed four untouched stable visual tests;
   two 3-wood clips also passed quantitative identity/precision review with
   documented conservative early termination.
11. **Completed:** schema v2, bounded inferred-tail prototype, two-baseline
   withheld-visible-tail evaluation, assisted curve, and renderer support.
12. **Completed:** render automatic cinematic outputs for `left-3wood` and
   `curve-3wood` without a landing tap or per-clip settings.
13. **Completed transfer render:** the identical 2-second cinematic settings
   generated `ml/output/straight-wedge.cinematic.mp4`; the final observed wedge
   motion is still ascending, so the shared rule places its inferred apex 0.65
   seconds later. No iron-specific parameters were added.
14. **Completed visual review; baseline rejected:** the tracker portions look
   good, but the invisible descent and trajectory do not. Do not fine-tune the
   Hermite rules.
15. **Completed design/research review:** primary sources plus Mizuno Vision,
   `GolfSimulator`, `Simple-Golf-Simulator`, and `libgolf` confirm that available
   repositories are either image-space detectors or forward simulators that
   require launch state. None recovers unique invisible 3D flight from current
   center-only video. `libgolf` is GPL-3.0 and must not be copied/linked without
   a licensing decision.
16. **Next:** implement the small `physics_template` mode in this order:
   simulator/unit tests, fixed profile bank, projection/fitter, schema/CLI,
   withheld-visible-tail evaluation, then one shared render of `left-3wood`,
   `curve-3wood`, and `straight-wedge`.
17. After that single gate passes, freeze continuation and optimize
   extraction/association runtime. Handheld/camera-motion work remains deferred.

### Archived temporal baseline commands — do not continue unchanged

These commands document how the first temporal baseline was produced. They are not the
current next step. A short MPS preflight prints live loss, ETA, average input wait,
step latency, throughput, and MPS memory without committing to a long run:

```bash
ml/.venv/bin/python ml/scripts/train_temporal_heatmap.py \
  ml/temporal_manifests/validation-curve-iron.json \
  --cache-dir ml/cache/temporal_640 \
  --device mps \
  --batch 2 \
  --workers 0 \
  --smoke-batches 20 \
  --output /tmp/shot-tracer-mps-preflight.pt
```

For reproduction, choose worker count and batch size from this preflight before starting
30 epochs. On macOS, test `--workers 0` first; compare `--workers 2` only if
`data_ms` is a meaningful fraction of `step_ms`. More workers are not assumed
to be faster.

The completed curve-iron validation fold used:

```bash
ml/.venv/bin/python ml/scripts/train_temporal_heatmap.py \
  ml/temporal_manifests/validation-curve-iron.json \
  --cache-dir ml/cache/temporal_640 \
  --device mps \
  --epochs 30 \
  --batch 2 \
  --workers 0
```

Every epoch writes `validation-curve-iron.last.pt` for resumption and updates
`validation-curve-iron.pt` when validation loss improves. Do not evaluate a
preflight checkpoint in `/tmp`; it only validates performance and plumbing.
Do not resume this architecture or run the other folds unless explicitly doing
reproduction; the immediate queue above supersedes this section.

### Labeling command

```bash
ml/.venv/bin/python ml/scripts/annotate_video.py ml/videos/curve-iron.MOV
```

The tool resumes at the first unlabeled frame. Its on-screen help lists all
controls. For `curve-iron`, use `--start-frame 230`. Right-click near the ball
to zoom, then left-click the actual center; the view returns to full frame after
the label. Select the shot state before labeling (`1` address, `2` flight,
`3` landed, `4` finished). Use `o`, `n`, or `x` when the ball is occluded, not
visible, or out of frame; never click an interpolated or expected position.

### Baseline scoring commands

```bash
ml/.venv/bin/python ml/scripts/track_video.py ml/videos/curve-iron.MOV

ml/.venv/bin/python ml/scripts/evaluate_tracking.py \
  ml/annotations/curve-iron.annotations.json \
  ml/output/curve-iron.diagnostics.json \
  --series trajectory

ml/.venv/bin/python ml/scripts/evaluate_tracking.py \
  ml/annotations/curve-iron.annotations.json \
  ml/output/curve-iron.diagnostics.json \
  --series candidates
```
