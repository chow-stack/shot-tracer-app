"""Train the first three-frame center-heatmap plus visibility baseline."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from temporal.dataset import TemporalTileDataset  # noqa: E402
from temporal.model import TemporalHeatmapNet, temporal_loss  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run_epoch(
    model: TemporalHeatmapNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    maximum_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"heatmap_loss": 0.0, "visibility_loss": 0.0, "total_loss": 0.0}
    batches = 0
    samples = 0
    data_seconds = 0.0
    step_seconds = 0.0
    total_batches = len(loader)
    if maximum_batches is not None:
        total_batches = min(total_batches, maximum_batches)
    phase = "train" if training else "validation"
    iterator = iter(loader)
    if maximum_batches is not None:
        iterator = itertools.islice(iterator, maximum_batches)
    progress = tqdm(
        iterator,
        total=total_batches,
        desc=f"Epoch {epoch:03d} {phase}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    waiting_started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in progress:
            data_seconds += time.perf_counter() - waiting_started
            step_started = time.perf_counter()
            non_blocking = device.type == "cuda"
            image = batch["image"].to(device, non_blocking=non_blocking)
            target_heatmap = batch["heatmap"].to(device, non_blocking=non_blocking)
            target_visibility = batch["visibility"].to(device, non_blocking=non_blocking)
            sample_weight = batch["weight"].to(device, non_blocking=non_blocking)
            heatmap_logits, visibility_logits = model(image)
            loss, metrics = temporal_loss(
                heatmap_logits,
                visibility_logits,
                target_heatmap,
                target_visibility,
                sample_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
            samples += image.shape[0]
            step_seconds += time.perf_counter() - step_started
            postfix = {
                "loss": f"{totals['total_loss'] / batches:.4f}",
                "data_ms": f"{1000 * data_seconds / batches:.0f}",
                "step_ms": f"{1000 * step_seconds / batches:.0f}",
            }
            if device.type == "mps":
                postfix["mps_MB"] = f"{torch.mps.current_allocated_memory() / 2**20:.0f}"
            progress.set_postfix(postfix)
            waiting_started = time.perf_counter()
    progress.close()
    if not batches:
        raise RuntimeError("Data loader produced no batches")
    result = {key: value / batches for key, value in totals.items()}
    result.update(
        {
            "data_seconds": data_seconds,
            "step_seconds": step_seconds,
            "samples_per_second": samples / max(data_seconds + step_seconds, 1e-9),
        }
    )
    return result


def atomic_torch_save(document: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(document, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", default="ml/cache/temporal_640")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument(
        "--smoke-batches",
        type=int,
        help="Run only this many train and validation batches.",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0:
        raise SystemExit("--epochs and --batch must be positive")

    torch.manual_seed(17)
    device = select_device(args.device)
    train_dataset = TemporalTileDataset(
        args.manifest,
        "train",
        input_size=args.input_size,
        cache_dir=args.cache_dir,
    )
    validation_dataset = TemporalTileDataset(
        args.manifest,
        "validation",
        input_size=args.input_size,
        cache_dir=args.cache_dir,
    )
    generator = torch.Generator().manual_seed(17)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch,
        shuffle=False,
        **loader_options,
    )
    model = TemporalHeatmapNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    best_validation = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["training"]["epoch"]) + 1
        best_validation = float(resume["training"]["best_validation_loss"])
    print(
        f"PyTorch {torch.__version__}; device={device}; "
        f"MPS built={torch.backends.mps.is_built()}, "
        f"available={torch.backends.mps.is_available()}",
        flush=True,
    )
    print(
        f"Dataset: {len(train_dataset)} train / {len(validation_dataset)} validation; "
        f"batch={args.batch}, workers={args.workers}, input={args.input_size}px; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )

    output_path = Path(
        args.output
        or Path("ml/models/temporal_heatmap") / f"{Path(args.manifest).stem}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = output_path.with_name(f"{output_path.stem}.last{output_path.suffix}")
    epochs = 1 if args.smoke_batches is not None else args.epochs
    final_epoch = start_epoch + epochs - 1 if args.smoke_batches is not None else epochs
    for epoch in range(start_epoch, final_epoch + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, epoch, args.smoke_batches
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, None, epoch, args.smoke_batches
        )
        print(
            f"Epoch {epoch:03d}: train={train_metrics['total_loss']:.5f}, "
            f"validation={validation_metrics['total_loss']:.5f}, "
            f"train throughput={train_metrics['samples_per_second']:.2f} samples/s",
            flush=True,
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model": {"context_frames": 3, "output_stride": model.output_stride},
            "training": {
                "manifest": str(Path(args.manifest).resolve()),
                "cache_dir": str(Path(args.cache_dir).resolve()),
                "input_size": args.input_size,
                "epoch": epoch,
                "best_validation_loss": min(
                    best_validation, validation_metrics["total_loss"]
                ),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            },
        }
        atomic_torch_save(checkpoint, last_path)
        if validation_metrics["total_loss"] < best_validation:
            best_validation = validation_metrics["total_loss"]
            checkpoint["training"]["best_validation_loss"] = best_validation
            atomic_torch_save(checkpoint, output_path)
        print(f"Checkpoints: last={last_path}, best={output_path}", flush=True)
    print(f"Training complete. Best checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
