# Shot Tracer

iPhone app that adds an automatic shot tracer to golf swing videos.

All project knowledge — architecture decisions, research, roadmap — lives in the
Obsidian vault (`shot-tracer-vault`), not here. Read `CLAUDE.md` there first, then
`wiki/index.md`. This repo is the code layer only.

## Layout

```
app/    Swift package: TracingCore library + tracer-harness CLI (ADR 0006)
ml/     Python: offline detector training, Core ML export (ADR 0007)
```

The two never mix. `ml/` trains models; the only artifact that crosses into `app/`
is the exported `.mlpackage`.

## app/ — quickstart

Requires Xcode 26 / Swift 6 toolchain.

```bash
cd app && swift build
```

```bash
cd app && swift run tracer-harness <video-file>
```

The harness is the Phase 0 offline test bed: it will load a clip, run a tracker,
and write an annotated video plus a JSON sidecar of per-frame points. It is the
seed of the production tracking engine, not throwaway code.

## ml/ — quickstart

Requires Python 3.13 (torch has no 3.14 wheels yet — do not use the Homebrew 3.14).

```bash
cd ml && /usr/local/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

See [ml/README.md](ml/README.md) for dataset downloads and training.
