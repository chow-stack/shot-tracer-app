# Legacy tracking pipeline: architecture, history, and migration status

> Current directive: remove automatic address initialization from the iOS
> production path, require a ball tap for every video, and optimize measured
> physical-device latency. Older automatic-tracking commands below are not the
> current task queue.

For current continuation instructions, begin with `../HANDOFF.md`. This file is
primarily the legacy failure/migration record.

This is the detailed technical record for the existing
`ml/scripts/track_video.py`. It preserves the debugging history so its failures
do not have to be rediscovered. `Master Plan.md` is the source of truth for the
replacement architecture, milestones, and acceptance gates. Read this file
before changing the legacy baseline, but do not extend its design by default.

> **Legacy status correction (2026-08-04): no legacy RANSAC output has a
> correct tracer.** Earlier visual assessments in this document overstated the
> results, especially for `curve-iron.MOV` and `straight-wedge.MOV`. A smooth
> or plausible-looking parabola is not evidence that the fitted points follow
> the golf ball. Until a trace is checked frame-by-frame against annotated
> ball centers (including launch, full visible flight, and termination), it
> must be treated as a failure/unvalidated result rather than a working case.

## What the legacy pipeline does

Two passes, run every time `track_video.py` is invoked on a clip:

**Pass 1 — Collection** (`collect_candidates`): starting from the user's
tap-to-init click, loop frame by frame. A Kalman filter predicts where the
ball should be; a crop is taken around that prediction; YOLO runs on the
crop. Every accepted YOLO output `(frame_idx, x, y)` is recorded as a
candidate. These outputs are only hypotheses: this pass cannot know whether
a detection is the real ball, even though the selected detection is used to
decide where to look next. No detection is ground truth.

**Pass 2 — Trajectory fit**: the legacy implementation assumes that a golf
ball in flight follows a parabola in screen space. `find_launch_frame` finds
where address ends and real flight begins (a single quadratic can't represent
"stationary for hundreds of
frames, then a sudden parabola," so address frames must be excluded before
fitting, not filtered out by the fit itself). `ransac_parabola_fit` then
fits x(t) and y(t) as independent quadratics via RANSAC over the
post-launch candidates, discarding whatever doesn't lie on the
best-supported curve. The final rendered trace is the fitted curve, not raw
per-frame points — smooth by construction, no per-frame jitter or hard
direction-reversals possible.

### Supporting mechanisms inside Pass 1

- **Growing search crop on misses** (`MISS_GROWTH_RATE`,
  `MAX_ROI_MULTIPLIER`): if the ball outruns the fixed base crop
  (`ROI_SIZE`), the search window widens geometrically each consecutive
  miss, capped at 5x base size, and resets on the next hit.
- **`airborne` gate**: growth is only active once a coarse single-frame
  displacement (`COARSE_LAUNCH_DISPLACEMENT`) between two accepted
  detections has been seen. Before that, a miss just means the detector
  didn't fire — the ball hasn't moved — so the crop should NOT widen.
  **This gate also controls the motion filter below.** Both mechanisms
  assume the ball is stationary until struck; neither should activate
  during address.
- **Nearest-to-prediction candidate selection** (`detect_ball_in_roi`):
  among multiple YOLO boxes in a crop, pick the one closest to the Kalman
  prediction, not the highest-confidence one.
- **Motion filter** (`compute_motion_mask`, `box_has_motion`): once
  airborne, a detection is only accepted if its region shows real
  frame-to-frame pixel change (`cv2.absdiff` between consecutive full
  frames, thresholded). A lamp post, flag, cloud, or settled dust can be
  confidently mis-detected as a ball. The filter attempts to reject static
  examples, but camera motion, foliage, clouds, dust, and other moving clutter
  can still pass it. **Gated on `airborne` for the same reason as
  crop growth** — the ball is genuinely motionless during address and would
  fail a motion check for no good reason.

## Why it's built this way — debugging history

Every iteration below fixed the specific failure it targeted and exposed a
new one underneath. Recorded in order so the reasoning chain is clear;
skipping straight to "current status" below is fine if you just need to
know where things stand.

1. **Baseline** (paper-faithful): fixed small crop (200px), Kalman filter
   tuned to smooth heavily (matching a generic default, not the paper's
   actual code). Result: lagged badly and lost the ball at impact — the
   velocity jump from ~0 to 100+mph in one frame outran both the crop and
   the filter's ability to react.
2. **Reference repo investigation**: `golf_ball_reference/test_net.py` (the
   paper authors' actual released code, not just the paper text) uses
   Q=0.1, R=0.0001 — trusts a fresh detection almost completely instead of
   smoothing it against the prediction. Their crop is a fixed 300×300 with
   no adaptive growth, and they never reject a detection (always take
   top-1, no confidence gate). Matched Q/R and crop size in
   `tracker/kalman.py` / `track_video.py`. This fixed the lag.
3. **New failure: locking onto real clutter.** With detections trusted
   almost completely and no gate, the online (single-pass, "trust and
   draw") tracker started locking onto other real, static objects in frame
   — a loose ball on the ground, a parked cart, a lamp post, a flag, dust
   kicked up at impact, even clouds. Root cause: nothing checked a new
   detection against the shape of the trajectory so far, only local
   per-frame plausibility.
4. **Tried and reverted**: confidence threshold + growing crop + plausibility
   gate (reject detections too far from the single-frame Kalman
   prediction). This *increased* problems — the growing crop swept in more
   clutter, and the plausibility gate over-rejected real fast-moving
   detections because of Kalman lag (a genuine post-impact detection looks
   "implausible" relative to a prediction that hasn't caught up yet).
5. **Pivot to trajectory fitting.** Realized no amount of frame-by-frame
   heuristic tuning fixes a problem that requires *aggregate* context: a
   real ball's positions have to lie on one consistent parabola; clutter's
   positions don't. This is what RANSAC is for. Rebuilt as the current
   two-pass architecture (§ above). Validated the fitting logic itself
   against synthetic data with injected outliers first — recovers true
   curve coefficients cleanly (see `ransac_parabola_fit` docstring / can be
   re-run: see "Validating the fit in isolation" below).
6. **New failure: Pass 1 data-starved.** With a *fixed* 300px crop and no
   growth, Pass 1 found only ~22 outputs believed at the time to include ball
   candidates across an entire flight. RANSAC fit whatever little (mostly
   wrong) data existed. Added the growing-crop-on-miss mechanism to collect a
   wider candidate pool, although the selected output still redirects the next
   crop and can therefore corrupt later collection.
7. **New failure: growth active during address.** Growing on *any* miss,
   including during address, swept in far-away false positives before
   flight even started, corrupting `find_launch_frame` (falsely detected
   launch at frame 0). Added the `airborne` gate so growth only activates
   post-launch.
8. **New failure: RANSAC converging on clutter clusters.** Even with a
   richer, mechanically gated candidate pool, RANSAC sometimes found its
   largest self-consistent group among a cluster of repeated false
   detections near a sign board (visually near "EVERYONE CAN PLAY"
   signage) rather than the true, sparser real-ball points — because nothing
   distinguished "consistent because it's the real ball" from "consistent
   because it's several confident false positives on the same static
   object." This is the actual reason `curve-iron.MOV` traced as a tiny
   near-static blob near the tee despite 103 "inliers."
9. **Motion filter added** (current, § above) — directly targets #8: static
   clutter cannot pass a frame-to-frame motion check, no matter how
   confident or self-consistent the detections on it are. This is
   intended to address the missing motion evidence. Later review showed this
   is still insufficient: background/camera motion can pass the filter, and
   threshold sensitivity alone cannot prove that accepted motion belongs to
   the ball.

## Legacy status (corrected 2026-08-04)

**No video produces a correct legacy RANSAC tracer.** The candidate counts and
smooth fitted curves below are debugging observations only. They do not
establish tracking accuracy, because the outputs have not followed the true
ball for the complete visible flight.

The per-clip figures below are frozen legacy reports from before the final
targeted label additions (curve 283–284 and six top frames). They are retained
for failure history rather than recomputed, because the replacement association
pipeline—not legacy RANSAC—is the active system.

### Corrected quantitative baseline: `curve-iron.MOV`

The first 37-point report was invalid because OpenCV random seek displayed
frames 5–6 indexes later than the saved labels. Sequential-only decoding fixed
the reader, and curve was relabeled and visually/pixel-QA validated.

Dense ground truth was completed for all 519 frames and the frozen legacy
pipeline was run with the median annotated address position as the reproducible
equivalent of the user's initialization tap. At a 20-pixel center tolerance:

- Ground-truth flight: frames 245–439, with 41 visible/blurred frames (245–287).
- Legacy launch estimate: frame 239 (six frames early).
- Accepted hypotheses: 150 total; 43 during the labeled flight scope.
- RANSAC: 24 inliers and 21 outliers after the estimated launch.
- **Rendered trajectory:** 0/41 correct visible frames, 0% recall, 0% frame
  precision, 100% false-track rate, 991.52px median center error, 1050.70px
  95th-percentile error, and zero contiguous correct frames.
- **Accepted YOLO series:** 12/41 correct visible frames at 20px (29.27%
  recall), 31.58% frame precision, 68.42% false-track rate, 4.77px median error,
  7.41px 95th-percentile error, and a 12-frame longest correct run.

The fitted trajectory remains entirely false despite some accurate accepted
detections, confirming that sparse greedy collection plus the fit is not enough.

### Quantitative frozen baseline: `straight-wedge.MOV`

Dense ground truth is complete for all 547 frames. The labeled flight begins at
frame 247 and contains 59 visible ball frames (247–305). At 20px tolerance:

- Legacy launch estimate: frame 241 (six frames early).
- Accepted hypotheses: 244 total, but only five overlap visible ground-truth
  flight frames.
- RANSAC: 14 inliers and 6 outliers after the estimated launch.
- **Rendered trajectory:** 1/59 correct visible frames (1.695% recall), 0.402%
  frame precision, 99.598% false-track rate, 912.67px median center error,
  942.04px 95th-percentile error, and a longest correct run of one frame.
- **Accepted YOLO series:** 1/59 correct visible frames (1.695% recall), 7.143%
  frame precision, 92.857% false-track rate, 774.26px median center error, and
  877.36px 95th-percentile error. The sole correct candidate is the launch
  frame itself (frame 247, 1.27px error); subsequent accepted outputs are
  hundreds of pixels from the ball.

This quantitatively rejects the earlier claim that loosening the motion filter
partially recovered the real shot. It recovered candidate count, not ball
identity or flight coverage.

### Quantitative frozen baseline: `top-iron.MOV`

Dense ground truth is complete for all 294 frames. The labeled flight is
frames 141–274 and contains 87 visible/blurred ball frames. At 20px tolerance:

- Legacy launch estimate: frame 135 (six frames early).
- Accepted hypotheses: 82 total, with only 10 overlapping visible ground-truth
  flight frames.
- RANSAC: 8 inliers and 8 outliers after the estimated launch.
- **Rendered trajectory:** 1/87 correct visible frames (1.149% recall), 0.746%
  frame precision, 99.254% false-track rate, 1201.31px median center error,
  1264.92px 95th-percentile error, and a longest correct run of one frame.
- **Accepted YOLO series:** 8/87 correct visible frames (9.195% recall), 80%
  frame precision, 20% false-track rate, and 7.05px median center error on
  frames where an output overlaps ground truth. The longest correct run is only
  three frames and the candidate series ends 69 frames before the last visible
  label.

Unlike the other two clips, this shows that the detector can localize the low
ball accurately on some grass-background frames. The dominant failure is
candidate recall/continuity: the greedy crop produces too few observations for
tracking, and RANSAC converts the sparse set into a nearly entirely false path.

- **`curve-iron.MOV`: incorrect.** The prior claim that its smooth leftward
  parabola matched the shot was based on appearance, not frame-by-frame
  validation. Its later annotations also failed pixel-level QA. It must not be
  used as a validation case until relabeled.
- **`straight-wedge.MOV`: incorrect.** Progression
  while loosening motion thresholds, using the "N candidates in first half
  / second half of video" diagnostic in `main()`:
    - `MOTION_DIFF_THRESHOLD=25, MOTION_MIN_FRACTION=0.1`: 233 before / 1
      after → no launch detected at all.
    - `15 / 0.03`: same result, 233 before / 1 after.
    - `8 / 0.015` (current values): **233 before / 11 after** → launch
      detected at frame 241 (matches this clip's known real launch point
      from earlier sessions), RANSAC fit found with only **14 inliers, 6
      outliers**. Rendered result (checked visually) shows a short,
      mostly-straight segment from the tee to roughly mid-frame. This is not
      a correct trace; direction and smoothness alone cannot establish that
      the detections belong to the ball. The output is sparse and unverified.
  This confirms only that loosening the motion filter changes the candidate
  count (from 1 to 11 post-launch). Without ground-truth comparison it does
  **not** confirm that the added candidates are the ball or validate the
  sky-contrast hypothesis. In any case, 11-14 points for a flight that likely
  spans 200+ frames is sparse. Do not keep loosening
  `MOTION_DIFF_THRESHOLD` / `MOTION_MIN_FRACTION`; follow the measured
  migration path below.
- **`top-iron.MOV`: incorrect and quantitatively validated.** Its sparse legacy
  detections sometimes localize the ball, but coverage is inadequate and the
  fitted tracer is false for 86 of 87 visible/blurred frames.
- **`practice-iron.MOV`: intentionally skipped** for the initial three-clip
  benchmark; no success claim is made for it.

## Migration path, in priority order

The schema, labeler, evaluator, per-video annotation files, and legacy
diagnostics export were implemented on 2026-08-04. `curve-iron.MOV`,
`straight-wedge.MOV`, and `top-iron.MOV` are fully labeled and have frozen
baseline reports. `practice-iron.MOV` is intentionally skipped for this initial
benchmark set.

1. Label every frame of the four existing clips using the implemented shared
   annotation format, recording center, visibility, and shot state.
2. Build an evaluator that reports candidate recall@K, tracking precision and
   recall, center error, longest contiguous correct run, launch/end error, and
   false-track rate.
3. **Completed on all three QA-valid clips:** score the legacy pipeline
   unchanged to create the honest baseline. Do not tune Kalman, motion, or
   RANSAC constants.
4. **Completed:** add a candidate-only diagnostic mode that exports multiple
   YOLO candidates per frame without letting one accepted detection determine
   the only next search area.
5. **Completed on QA-valid labels:** benchmark overlapping high-resolution
   tiled inference. With 640px tiles, 25% overlap, confidence 0.01, cross-tile
   NMS, and a 20px tolerance, recall@1/3/5 was 19.5/29.3/29.3%
   (`curve-iron`), 50.8/50.8/50.8% (`straight-wedge`), and
   6.9/10.3/10.3% (`top-iron`). The crop was not the sole bottleneck and extra
   YOLO hypotheses rarely recovered a missed ball.
6. **Corrected result:** the deterministic 3-frame checkpoint reaches 36.6%
   curve recall@3/5; every hit is already present among motion candidates.
7. **Completed:** motion top-100 recall is 100% (`curve`), 98.3% (`straight`),
   and 100% (`top`). Tiled YOLO recovers the remaining straight frame, giving
   100% union candidate coverage on every stable clip.
8. **Completed:** replace greedy association and the single
   RANSAC fit with a clip-level candidate graph containing top-K detections and
   explicit gap/terminated states. Shared beam-search settings currently achieve
   reconstructed-path recall of 100% on all labeled visible frames with
   3.10–4.00px median error after targeted annotation review.
9. **Completed:** review targeted annotation false negatives, rerun final
   observation/path metrics, and render all three association-backed QA videos.
   Curve/straight terminate exactly; top has one observation on human-labeled
   occluded frame 235 and terminates one frame late.
10. In parallel, rebuild the leaked Roboflow image split by original source group.
11. Measure background motion first; add robust registration before differencing
   handheld footage or whenever tripod motion is not negligible.
12. **Completed:** all three association-backed QA videos passed human visual
   review.
13. **Next:** remove the remaining inference leakage. Motion/YOLO candidate
   generation currently uses the annotated flight range, and association uses
   that same range plus a median address label. Replace this with complete-clip
   candidates, one explicit user-equivalent tap, automatic launch detection,
   and automatic termination. Annotations then belong only to evaluation.
14. Freeze the resulting settings and evaluate new untouched stable
   tripod/behind-golfer clips without clip-specific tuning. Handheld registration
   is deferred.

### Target replacement in one sentence

Decode with real timestamps → measure/compensate camera motion as needed →
generate top-K candidates from motion and a specialized detector (learned
temporal fusion only if justified) → select the best
complete path from top-K and invisible hypotheses → smooth with uncertainty →
map back to original frames → render.

The legacy Kalman filter can remain a baseline smoother or a state model inside
each hypothesis. It must not remain the sole association mechanism. The legacy
RANSAC parabola can remain a diagnostic fit, but it must not define correctness
or synthesize the final tracer from unverified candidates.

## Validating the fit in isolation

If `ransac_parabola_fit` or `find_launch_frame` is ever suspected of being
wrong (as opposed to Pass 1 not collecting good candidates), test them
against synthetic data first, independent of any real video or model
inference — this isolates "is the math right" from "is the input data
good," which is exactly the distinction that mattered in steps 6-9 above:

```python
import sys
sys.path.insert(0, 'ml/scripts')
from track_video import ransac_parabola_fit, find_launch_frame
import numpy as np

rng = np.random.default_rng(0)
candidates = []
for t in range(50):  # address: near-static jitter
    candidates.append((t, 700 + rng.uniform(-3, 3), 1650 + rng.uniform(-3, 3)))
true_px, true_py = [0.01, -3.0, 700], [0.05, -12.0, 1650]
for t in range(50, 121):  # flight: true parabola + noise
    x = np.polyval(true_px, t) + rng.uniform(-5, 5)
    y = np.polyval(true_py, t) + rng.uniform(-5, 5)
    candidates.append((t, x, y))
candidates += [(70, 950, 300), (90, 100, 900), (110, 1800, 200)]  # outliers

launch = find_launch_frame(candidates)
flight = [c for c in candidates if c[0] >= launch]
px, py, n_in, n_out = ransac_parabola_fit(flight)
# Expect: launch ~50, n_in ~70, n_out ~3, px/py close to true_px/true_py
```

This passed cleanly when last checked (see debugging history step 5).

## Key constants reference

| Constant | Current value | Meaning |
|---|---|---|
| `ROI_SIZE` | 300 | Base search crop size, matches reference implementation |
| `CONF_THRESHOLD` | 0.1 | YOLO confidence floor — kept low; RANSAC/motion filter handle outlier rejection, not this |
| `MISS_GROWTH_RATE` / `MAX_ROI_MULTIPLIER` | 1.5 / 5 | Crop widening rate/cap on consecutive misses, post-airborne only |
| `COARSE_LAUNCH_DISPLACEMENT` | 50.0px | Single-frame jump that flips the internal `airborne` flag (gates crop growth + motion filter) |
| `LAUNCH_WINDOW` / `LAUNCH_DISPLACEMENT` / `LAUNCH_CONFIRM` | 8 / 150.0 / 3 | Sustained-motion check (final launch-frame decision for the fit, separate from the coarse gate above) |
| `RANSAC_ITERS` / `RANSAC_INLIER_THRESHOLD` / `RANSAC_MIN_INLIERS` | 500 / 30.0 / 6 | RANSAC fit parameters |
| `MOTION_DIFF_THRESHOLD` / `MOTION_MIN_FRACTION` | 8 / 0.015 | Motion filter sensitivity — under active tuning, see Current Status |

`tracker/kalman.py`: `process_noise=0.1, measurement_noise=0.0001` (Q/R
ratio ~1000, matches reference implementation — trusts fresh measurements
almost completely rather than smoothing). `P` initialized to zero (not a
high-uncertainty default) since frame 0 comes from a known-good point (the
user's tap), matching the reference implementation's treatment of their
hardcoded ground truth.
