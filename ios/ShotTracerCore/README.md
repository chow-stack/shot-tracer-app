# ShotTracerCore

Portable Swift core for the offline iOS shot-tracer MVP. The package contains:

- schema-v2 trajectory and candidate models;
- deterministic multi-source candidate merging;
- the full-clip launch/association/termination beam search;
- the frozen SI-unit physics simulator, 18-flight club template bank, and
  anchored screen-projection continuation fitter;
- a provenance-aware Core Graphics renderer for observed, estimated-gap, and
  inferred points;
- sequential `AVAssetReader`/`AVAssetWriter` export with source orientation,
  exact presentation times, optional audio passthrough, progress, and
  cooperative cancellation;
- an Apple Vision candidate generator routed through the frozen full-clip
  launch/association/termination search;
- conservative continuation quality gates and presentation-only smoothing; and
- a `ShotTracerUI` SwiftUI workflow with Photos selection, address/profile
  controls, progress/cancel, preview, Photos saving, and system sharing.

`ios/ShotTracerApp/ShotTracerApp.xcodeproj` is the runnable iOS 17 app shell.
The checked-in UI still defaults to automatic address setup, but that is now a
known temporary state rather than the intended MVP. The next agent must require
an address-ball tap for every selected video, remove automatic-address controls,
and prevent production processing without a tap. Vision remains a native
candidate source for stable tripod clips; it does not select a trajectory by a
single coarse score. The parity-tested associator must confirm a sustained
launch anchored to the user tap.

The renderer smooths only display coordinates with a symmetric five-sample
kernel. Stored observations, timestamps, confidence, and provenance stay
unchanged. Physics continuation is withheld when the associated track has too
few observations, a gap over two frames, an unstable terminal direction, a
non-positive association score, or a weak automatic club-family margin. A
failed gate exports the observed tracer and labels it `Observed flight`; it does
not silently fall back to an iron prediction.

The tests decode the frozen Python association and physics-template artifacts.
They also execute the Swift association over both motion-only and combined
motion + tiled-detector candidates and require exact agreement with the frozen
Python trajectories. The Swift physics continuation is checked against all
three accepted Python review clips, including the chosen template, projection,
endpoint, uncertainty, opacity, and every inferred coordinate.
Run them from this directory with workspace-local caches:

```bash
CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache" \
SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/swift-module-cache" \
swift test --disable-sandbox --scratch-path "$PWD/.build"
```

This execution sandbox does not expose the AVFoundation media decoder/encoder
needed by the fixtures, so two complete media integration tests remain disabled
in the host package and must run in the future iOS/device test scheme. Bitmap
rendering, schema, numerical parity, overwrite safety, and all non-media tests
remain active here.

The native detector now also generates bounded three-frame motion candidates in
the same sequential decode pass. It uses half-resolution vImage scaling, the
minimum of adjacent frame changes, two inclusive thresholds, 8-connected
components, compact-component rejection, cross-source merging, and a 100-point
per-frame cap. This is intentionally a candidate generator rather than another
tracker; the frozen associator still makes the trajectory decision. Tracking
summaries are emitted through unified logging under the `tracking` category.

## Immediate work for the next agent

1. **Done (2026-08-05):** the address tap is mandatory and
   `AutomaticShotSetup.initialize` is unreachable from production;
   `AutomaticShotSetup` remains only as research code exercised by its own
   tests. This also surfaced and fixed a tap/decode coordinate-mapping bug —
   see `NOTES.md`.
2. **Instrumented, not yet measured:** stage-level `ContinuousClock` timing
   for decode, vImage downscale/change, connected components, Vision,
   candidate merge, association, continuation, and export is implemented and
   logged via `Logger(subsystem: "com.caio.ShotTracer", category:
   "performance")`. Get real physical-device numbers (Release build) before
   doing anything else — this sandbox cannot run on real hardware.
3. Optimize only the measured bottleneck with before/after timings on
   identical clips. Preserve cancellation, bounded memory/candidate counts,
   exact timing, audio/orientation, trajectory correctness, and save/share
   behavior.

Do not add retries, loosen confidence gates, tune individual clips, resume model
training, or work on automatic initialization until these tasks are complete.

Reference tolerances and artifact digests live in
`ml/fixtures/ios_parity_v1.json`; verify them with
`ml/scripts/verify_ios_parity_fixtures.py` before and after porting a stage.
