# Shot Tracer

**An offline-first iPhone app that draws a broadcast-style tracer on your golf shot videos, entirely on device.**

Pick a swing video, tap the ball at address, and Shot Tracer finds the ball through its visible flight, draws a confidence-aware tracer in the original video coordinates, and exports a new video with the original audio. Nothing is uploaded.

The repository contains both halves of the project:

- **`ios/`**: the Swift package (`ShotTracerCore` + `ShotTracerUI`) and the runnable iOS app.
- **`ml/`**: the Python research pipeline used to design, train, evaluate, and freeze the tracking algorithm that the iOS port must match exactly.

![Swift](https://img.shields.io/badge/Swift-6-f05138) ![iOS](https://img.shields.io/badge/iOS-17%2B-black) ![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab) ![Status](https://img.shields.io/badge/status-MVP%20in%20progress-orange) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

---

## Contents

- [How it works](#how-it-works)
- [Project status](#project-status)
- [Repository layout](#repository-layout)
- [iOS app](#ios-app)
- [ML research pipeline](#ml-research-pipeline)
- [Design documents](#design-documents)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## How it works

A golf ball in flight is a few pixels wide, motion-blurred, and often lost against the sky. A single detector with greedy frame-to-frame tracking fails on most real clips, so Shot Tracer uses a staged, whole-clip pipeline:

| Stage | What happens |
|---|---|
| **Initialize** | The user taps the ball at address. This anchors the launch and is required for every video. |
| **Candidates** | Several independent sources propose possible ball positions in each frame: three-frame motion differencing at half resolution (vImage), Apple Vision trajectory detection, and, in the research pipeline, a tiled YOLOv8 detector. Candidates are merged and capped per frame. |
| **Association** | A multi-hypothesis **beam search over the full clip** chooses the launch, links candidates into one physically plausible track, and decides where the observed flight ends. No single noisy detection can hijack the result. |
| **Continuation** *(optional)* | When the observed track is strong enough, a physics simulator (drag, lift, and an 18-flight club template bank) fits a projected continuation past the point where the ball becomes invisible. Strict quality gates withhold it otherwise. |
| **Render and export** | A Core Graphics renderer draws observed, gap-estimated, and inferred segments with distinct provenance. Display-only smoothing never alters stored observations. `AVAssetReader` and `AVAssetWriter` re-encode with exact timestamps, the original orientation, and audio passthrough. |

The tracker and the physics templates are **frozen** in Python, and parity fixtures require the Swift implementation to reproduce the Python trajectories exactly.

## Project status

This is an active research project, not a finished product.

- ✅ The Python tracker meets its path-coverage and visual-QA gates on the development clips and on held-out clips with frozen settings.
- ✅ The Swift port passes numerical parity tests against the frozen Python association and physics artifacts.
- ✅ The iOS app builds and runs end to end: Photos picker → address tap → progress and cancel → preview → save or share.
- 🚧 Real-device performance profiling and optimization are in progress. Stage-level timing is already logged.
- 🔭 Handheld footage, automatic initialization, and live capture are later milestones.

The intended capture is a single shot, filmed from behind the golfer, on a stable phone at 60 fps.

## Repository layout

```
ios/
├── ShotTracerCore/           Swift package
│   ├── Sources/ShotTracerCore    models, candidate generation, association, physics, rendering, export
│   ├── Sources/ShotTracerUI      SwiftUI workflow (picker, tap, progress, preview, save/share)
│   └── Tests/                    parity, physics, rendering, and schema tests
└── ShotTracerApp/            Xcode project for the runnable iOS app
ml/
├── scripts/                  CLI tools: annotate, detect, associate, continue, render, train, evaluate
├── tracker/                  Kalman filter, trajectory I/O, video I/O, forecasting
├── physics/                  Ball-flight aerodynamics, integrator, projection, club templates
├── temporal/                 Experimental temporal heat-map detector (dataset, model, manifests)
├── annotations/              Hand-labeled ground truth for the development clips (+ JSON schema)
├── fixtures/                 iOS parity fixture manifest with tolerances and artifact digests
└── tests/                    Unit tests for every stage
golf_ball_reference/          Third-party baseline (Faster R-CNN + Kalman), kept for comparison
```

## iOS app

### Requirements

- Xcode 16 or newer (Swift 6 toolchain)
- An iPhone running iOS 17 or later (the simulator works for UI, but real performance numbers need a device)

### Run the app

1. Open `ios/ShotTracerApp/ShotTracerApp.xcodeproj`.
2. Under **Signing & Capabilities**, choose your own team and a unique bundle identifier.
3. Build and run on a device.

### Test the core package

```bash
cd ios/ShotTracerCore
swift test
```

Two full media integration tests need the AVFoundation encoder and decoder, so they run only in the iOS or device test scheme. Every other test runs on the host. See [`ios/ShotTracerCore/README.md`](ios/ShotTracerCore/README.md) for details.

## ML research pipeline

### Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets, trained weights, videos, caches, and outputs are git-ignored. To train the tiled detector, download the golf-ball dataset from [Roboflow](https://roboflow.com) (you need your own free API key):

```bash
export ROBOFLOW_API_KEY=your-key-here
python scripts/download_roboflow.py
python scripts/train.py
```

This writes weights to `ml/models/golf_ball_yolov8n/weights/best.pt`.

### Trace a video

```bash
python scripts/run_shot_tracer.py path/to/shot.mov --tap-x 812 --tap-y 1460
```

`--tap-x` and `--tap-y` are the ball's pixel position at address. Use `--device cpu` on machines without Apple-silicon GPU support, or `--skip-yolo` to run with motion candidates only.

### Tests and parity

```bash
python -m unittest discover -s tests                 # run from ml/
python scripts/verify_ios_parity_fixtures.py         # check frozen artifacts against their digests
```

## Design documents

| Document | Contents |
|---|---|
| [`Master Plan.md`](Master%20Plan.md) | Product goal, target architecture (stages A–G), dataset and metrics, roadmap and gates |
| [`TRACKING.md`](TRACKING.md) | Tracker design and the evaluation history behind it |
| [`INFERRED_TRAJECTORY.md`](INFERRED_TRAJECTORY.md) | Contract for the physics-based continuation and its observation boundary |
| [`RESEARCH.md`](RESEARCH.md) | Survey of golf and fast-ball tracking research and open-source projects |
| [`HANDOFF.md`](HANDOFF.md) and [`NOTES.md`](NOTES.md) | Development log, frozen artifacts, and next steps |

## Acknowledgements

- `golf_ball_reference/` is from the paper *Efficient Golf Ball Detection and Tracking Based on Convolutional Neural Networks and Kalman Filter* (Zhang et al., [arXiv:2012.09393](https://arxiv.org/abs/2012.09393)). It builds on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and is distributed under its own [MIT License](golf_ball_reference/LICENSE).
- The detector training data is the *Golf Ball Detection* dataset (v2) from [Roboflow Universe](https://universe.roboflow.com), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), OpenCV, PyTorch, and Apple's Vision, AVFoundation, and Accelerate frameworks.

## License

Released under the [MIT License](LICENSE), except `golf_ball_reference/`, which keeps its original license.
