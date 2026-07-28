# ml/ — offline model training

Python side of the project (ADR 0007): trains the golf-ball detector and exports
it to Core ML. Nothing here ships in the app; the only artifact that crosses
into `app/` is the final `.mlpackage`.

## Setup

```bash
/usr/local/bin/python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

Everything under `data/` is gitignored. See `data/README.md` for the layout.

**Roboflow GolfBall** (~17k images, licence cleared 2026-07-28):

```bash
ROBOFLOW_API_KEY=<your-key> .venv/bin/python scripts/download_roboflow.py
```

**CaddieSet** (MIT — test/validation footage + per-shot ball-flight stats):

```bash
git clone https://github.com/damilab/CaddieSet data/caddieset
```

**GolfDB** (CC BY-NC — research reference ONLY, never in a shipping model):

```bash
git clone https://github.com/wmcnally/golfdb data/golfdb
```

Preprocessed GolfDB clips come from the Google Drive link in their README.

## Train

```bash
.venv/bin/python train.py --data data/roboflow-golfball/data.yaml
```

Defaults: YOLOv8n, 100 epochs, 640px, Apple Silicon GPU (`--device mps`).
Runs land in `models/<run-name>/`.

## Export to Core ML

```bash
.venv/bin/python export_coreml.py models/<run-name>/weights/best.pt
```

Copy the resulting `.mlpackage` into `app/` (TracingCore resource) when a model
is good enough to use — that copy is the single point of contact between `ml/`
and `app/`.
