# Golf-ball tracking architecture review

> Current implementation note (2026-08-04): this file preserves method
> rationale and experimental evidence; it is not the task queue. The immediate
> production work is to require a user address-ball tap for every iOS video and
> optimize measured physical-device processing time for sub-10-second clips.
> See `HANDOFF.md` and Section 6 of `Master Plan.md`.

Updated 2026-08-04. This document records the evidence behind the architecture
decision in `Master Plan.md`. It deliberately separates methods that are
scientifically plausible from methods justified by this project's current data.

## Decision

The motion-first candidate and clip-level association sequence described below
is complete and the observed tracker is frozen. Further temporal-model training
remains deferred because it added no candidate coverage. The current task is a
post-tracking `physics_template` continuation: a small tested forward simulator,
a fixed wood/iron/wedge template bank, and low-dimensional fitting to the frozen
2D observations. It is completed before runtime optimization and remains
explicitly inferred.

The completed tracking order was motion recall, detector support, candidate
union, and clip-level association. Learned temporal fusion remains conditional
on a future measured candidate-recall gap and more diverse data.

## What the local evidence says

- The legacy fitted trajectory is correct within 20px on only 1/59
  `straight-wedge` and 1/87 `top-iron` visible frames.
- Full-frame tiled YOLO recall@5 is 50.8% and 10.3% respectively. The
  greedy crop is therefore not the only problem.
- A 5–6-frame random-seek offset invalidated the first `curve-iron` labels. The
  reader was fixed and curve was relabeled to 41 QA-valid flight points. There
  are now 187 QA-valid visible/blurred points across three clips.
- Re-scored against correct labels, the first 3-frame model reaches 36.6%
  recall@3/5 on curve, but adds no hit outside the motion candidate set.
- Consecutive ball displacement is large and measurable immediately after
  impact (maximum 129–211px across the clips), while the videos are stable.
  This makes differencing a high-value baseline.
- The ball remains human-labelable for only 37–87 flight frames. Positions
  after that boundary are inferred, not observed by the camera.
- Motion top-100 recall is 100%, 98.3%, and 100% on curve, straight, and top.
  Tiled YOLO recovers the only remaining straight frame, producing 100% oracle
  candidate coverage. The ball may rank 68th among over a thousand components,
  which is why global association—not raw extraction—was implemented next.

### Detector split audit

The downloaded Roboflow dataset contains shifted/cropped variants whose names
share an original-image prefix. Siblings occur across supplied splits; e.g.
`00003_00001`, `00003_00003`, and `00003_00004` occur in train, validation,
and test and show the same source scene. Measured source-prefix overlap is:

- 347 of 348 validation groups also occur in train;
- 197 of 198 test groups also occur in train;
- 165 groups occur in both validation and test.

Therefore the reported YOLO image metrics are contaminated and must not be
described as held out. This is especially important because the source paper
states that originals were split before shifted-patch augmentation.

## Method survey

| Method | Strength | Main failure/risk here | Role in this project |
|---|---|---|---|
| Manual keyframes + fitted/rendered curve | Reliable, shippable with little data | Not fully automatic; inferred segment is not measurement | MVP fallback and truth-preserving UX |
| Frame/background differencing + components | No training; exploits fast motion; ideal for tripod tests | Moving clutter, compression, clouds; needs registration when handheld | Immediate candidate baseline |
| Specialized high-resolution crop detector | Strong appearance/localization after initialization | Domain-specific data; crop can lose a fast launch | Second baseline with fallback search |
| Generic tiled detector / SAHI | Prevents full-frame downsampling loss | Cannot repair detector domain mismatch | Diagnostic/fallback, already measured |
| TrackNet-style temporal heatmap | Joint appearance/time evidence; natural center output | Requires varied dense video data; can memorize backgrounds | Conditional later model |
| Motion-aware learned video detector | Stronger fusion of explicit motion and appearance | Greater data, compute, and implementation cost | Later ablation after simple fusion |
| Multi-hypothesis DP/Viterbi/beam association | Can recover through clutter/misses and avoid greedy lock-on | Cannot recover a ball absent from candidates | Build after candidate recall gate |
| Kalman/EKF | Efficient state estimate and smoothing | Greedy use becomes irreversible; launch violates constant velocity | Smoother or per-hypothesis state only |
| Optical flow / generic point trackers | Can propagate a known point without ball labels | Tiny appearance-changing ball and featureless sky violate assumptions | Optional zero-shot benchmark, not primary |
| Video segmentation foundation model | Promptable and reusable | Object is below normal mask scale; large/slow; error propagation | Low-priority benchmark |
| Blur/streak modeling | Uses signal that box detectors discard | Requires streak labels and sufficiently visible blur | Later auxiliary cue |
| Stereo, radar, event/high-speed cameras | Physical trajectory and high-speed measurement | Outside one-60fps-iPhone product constraint | Out of scope |

## What the literature actually supports

### Golf-specific detector plus Kalman

Zhang et al. use a known/initialized ball position, a Kalman-predicted local
crop, and a specialized detector trained on 416px patches. Their detector data
contains 1,699 originals expanded through shifted patches; their tracking test
contains only 405 selected frames. They report a notable drop when moving from
detector test images to tracking videos and explicitly attribute it to dataset
representativeness. Useful lessons are high-resolution local inference,
specialized anchors/data, and initialization—not that greedy Kalman association
will generalize to our footage.

Source: [Efficient Golf Ball Detection and Tracking Based on CNNs and Kalman Filter](https://arxiv.org/abs/2012.09393)

### Golf-specific motion pipeline

Baidya et al. combine YOLO localization, Hough-based impact localization, frame
differencing during flight, and size/direction filtering. The accessible record
does not provide enough evidence to adopt its reported accuracy uncritically,
but it confirms that motion-first golf-ball candidate generation is a published
alternative and should be tested before a larger learned model.

Source: [Tracking the Trajectory of a Golf Ball Based on Hough Transform and YOLO](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002717147)

### TrackNet family

TrackNet shows that consecutive-frame heatmaps work for tiny fast sports
objects, but its experiments use tens of thousands of labeled frames and show a
large generalization drop on added videos/courts. TrackNetV3 adds a U-Net,
weighted loss, background estimation, mixup/augmentation, and trajectory
rectification. TrackNetV4 adds learnable motion attention based on frame
differences. These works support temporal/motion modeling, but not training a
small raw-RGB network from scratch on two golf videos.

TrackNetV4's published dataset is restricted to research/noncommercial use, so
it can inform architecture experiments but should not silently become commercial
product training data.

Sources: [TrackNet](https://arxiv.org/abs/1907.03698),
[TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf),
[TrackNetV4](https://time.griffith.edu.au/paper-sites/tracknet-v4/)

### Other learned temporal approaches

A soccer-ball study first trains a single-frame encoder-decoder and then adds a
temporal convolutional/recurrent stage, with synthetic sequence pretraining.
Video small-object detectors such as LSTFE-Net and flow-guided feature
aggregation show the benefits of temporal feature fusion, but add substantial
complexity. They are options after the project establishes strong baselines and
acquires more source diversity.

Sources: [Deep Ball Detection in Sports](https://arxiv.org/abs/1909.02406),
[LSTFE-Net](https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.html),
[Flow-Guided Feature Aggregation](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.html)

### Tiling and generic trackers

Sliced inference improves small-object detection when resolution is the
bottleneck, but the local tiled-YOLO benchmark shows that resolution alone does
not solve our domain mismatch. Generic tracking benchmarks also show that small
fast objects remain difficult. TAPIR and CoTracker3 are worth one zero-shot
test, but a golf ball often becomes a changing, blurred point against a
featureless sky; CoTracker3 itself identifies featureless regions as a failure
case. SAM 2 is a large promptable video segmenter, not evidence that sub-mask-
scale golf balls can be followed reliably.

Sources: [SAHI](https://arxiv.org/abs/2202.06934),
[Small and Fast Moving Object Tracking benchmark](https://arxiv.org/abs/2209.04284),
[TAPIR](https://arxiv.org/abs/2306.08637),
[CoTracker3](https://cotracker3.github.io/),
[SAM 2](https://arxiv.org/abs/2408.00714)

### Blur-aware detection

BlurBall models blur-center and orientation information with multiple frames.
This is promising because a fast golf ball can be a streak rather than a box,
but it requires additional labels and should follow the motion baseline.

Source: [BlurBall](https://arxiv.org/abs/2509.18387)

### Commercial workflow evidence

Commercial products are useful product evidence but not reproducible method
evidence. Shot Tracer markets automatic tracing and recommends a steady camera,
but its detector is proprietary. Other apps explicitly use manual impact/apex/
landing points or manual tracking. This supports keeping a polished semi-manual
fallback; it does not establish the accuracy of any automatic architecture.

Sources: [Shot Tracer for iOS](https://www.shottracerapp.com/ios),
[Golf Ball Tracer](https://golfballtracer.com/),
[FairwayPro](https://fairwaypro.ie/)

### Methods outside the product constraint

Radar, stereo/multi-camera reconstruction, high-frame-rate machine vision, and
event cameras can recover information unavailable to one ordinary 60fps phone.
They are valid golf-ball measurement methods but answer a different hardware
question. They should not be used to imply that the monocular iPhone can measure
the ball after it is no longer visible.

## Research addendum: continuing an unobservable tail

The archived Zhang et al. golf tracker does not solve this problem. Its Kalman
filter predicts the next frame's crop and then receives a detector measurement
update. That supports short-horizon search and smoothing, not a full invisible
flight or landing estimate.

Published trajectory rectification also needs a careful boundary. TrackNetV3
repairs occasional missing shuttlecock coordinates by constructing masks around
gaps and inpainting with surrounding sequence evidence. This supports our
bounded `estimated_gap` category; it does not turn an open-ended terminal gap
into an observation.

Monocular 3D ball work adds constraints unavailable to the current app. Van
Zandycke and De Vleeschouwer use a calibrated camera plus the known real ball
diameter and its measurable pixel diameter. Other reconstruction systems use
multiple cameras. Once this iPhone's distant golf ball blends into clouds, its
center and diameter are not measurable enough to supply those constraints.

Finally, an ideal parabola is not a correct golf-flight model. Wind-tunnel work
by Bearman and Harvey measures lift and drag across spin and Reynolds-number
conditions and uses those aerodynamic data to calculate trajectories. Without
launch speed/direction, spin, wind, aerodynamic coefficients, camera geometry,
and ground geometry, a detailed physical simulator would add unjustified
precision. The appropriate MVP result is therefore an explicitly inferred
image-space completion with uncertainty. Because the landing may be invisible
to the user too, endpoint adjustment is optional aesthetic editing rather than
a required or truth-producing input.

A second golf-ball experiment measured 13 production models across 18–91 m/s
and 1500–4500 rpm, finding that lift and drag depend on speed, spin, and ball
model. A 2026 monocular ball-reconstruction benchmark likewise found that
single-view ambiguity and observation noise can dominate the choice of physical
model. Together these results justify preserving measured image-plane velocity
and broad flight-shape constraints while refusing to claim a unique 3D solution.

The corrected Hermite implementation used recent timestamped points to estimate
terminal velocity and vertical deceleration. It preserved the upward derivative
but failed the user visual gate: all three reviewed clips hit the same bounded
apex floor, and arbitrary image-height endpoints still dictated the descent.
It remains a provenance-safe baseline, not the accepted continuation.

## External repository review and revised continuation decision

Four public repositories were reviewed against the current center-only,
behind-the-golfer pipeline:

1. [Mizuno IdentifyingBallTrajectoriesinVideo](https://github.com/MIZUNO-CORPORATION/IdentifyingBallTrajectoriesinVideo)
   wraps Apple's `VNDetectTrajectoriesRequest`. It provides useful native iOS
   observations, UUIDs, confidence, timestamps, ROI/size filters, and coordinate
   conversion. Its `projectedPoints` are a quadratic representation of recent
   detected points, not an invisible full-flight prediction. It assumes a fixed
   background and is a possible later candidate baseline, not a continuation
   solution.
2. [samo2491/GolfSimulator](https://github.com/samo2491/GolfSimulator) combines
   close side-on bright-blob tracking, selected-club priors, and a forward
   simulator. The transferable product idea is the club-conditioned prior and
   simulated fallback. Its fixed `Cd`, `Cl`, spin-decay, and pixel-derived launch
   claims are not sufficiently justified for this footage.
3. [Simple-Golf-Simulator](https://github.com/markccchiang/Simple-Golf-Simulator)
   exposes gravity, wind-relative drag, lift, and spin equations under ISC, but
   assumes metric launch velocity, spin axis, and aerodynamic coefficients. Its
   position/velocity stepping is educational rather than the coupled-state
   integrator to adopt for production.
4. [libgolf](https://github.com/gdifiore/libgolf) has the strongest engineering
   reference: pluggable aerodynamics/integration/ground phases, Reynolds- and
   spin-factor-dependent coefficients, convergence safeguards, tests, and a
   regression harness against shot-reference summaries. It is still a forward
   simulator requiring launch speed/angle/spin, and the repository is GPL-3.0.
   Use its separation and validation philosophy as design references; do not
   copy or link its implementation into a proprietary product without an
   explicit licensing decision.

Together with the primary literature, these repositories support a small
forward simulator but do not make a unique monocular inverse solution possible.
The next implementation is therefore a versioned, offline-generated bank of
wood/iron/wedge physical flight priors. Runtime fits only a low-dimensional time
and screen projection to the frozen visible 2D track, optionally constrained by
the address point and a horizon/distant-ground control. It outputs an inferred
cinematic path with growing uncertainty, not measured depth, spin, carry, or
landing.

Implementation and acceptance details are in `INFERRED_TRAJECTORY.md`.

Primary sources: [Zhang et al.](https://arxiv.org/abs/2012.09393),
[TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf),
[single-calibrated-image ball localization](https://arxiv.org/abs/2204.00003),
and [golf-ball aerodynamics](https://doi.org/10.1017/S0001925900007617). The
official ball limits are documented in the
[USGA Equipment Rules](https://www.usga.org/content/dam/usga/pdf/Equipment/Equipment%20Rules%20Final.pdf);
those constants alone do not reveal flight state.
Additional primary evidence: [still-air golf-ball measurements](https://doi.org/10.3390/proceedings2060238)
and the [monocular physics-model benchmark](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Grad_Physics-Based_3D_Ball_Trajectory_Reconstruction_from_Monocular_Soccer_Video_A_CVPRW_2026_paper.html).

## Experimental decision gates

1. Implement clip-level association over motion top-100, tiled candidates, and
   explicit invisible states; score it on all three clips.
2. Ablate motion strength, address/launch prior, velocity/acceleration
   continuity, candidate source, and invisible-state penalties.
3. Reduce K and optimize the 440–466ms/frame extractor only after correctness.
4. If motion misses specific backgrounds, evaluate the union of motion and a
   source-grouped specialized detector.
5. If that union still misses labelable balls, collect materially more source
   videos before temporal training. Use synchronized augmentation and pretrained
   features only at that point.
6. Keep manual keyframes as the product fallback and separately label rendered
   trajectory segments as observed, short-gap estimated, or inferred.
