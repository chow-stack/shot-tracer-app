#!/usr/bin/env python3
"""Train a YOLO golf-ball detector (ADR 0007).

Usage:
    python train.py --data data/roboflow-golfball/data.yaml
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/roboflow-golfball/data.yaml",
        help="dataset yaml (YOLO format)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="base checkpoint to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="mps",
        help="mps (Apple Silicon), cpu, or a CUDA index",
    )
    parser.add_argument("--name", default="golfball-yolov8n", help="run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project="models",
        name=args.name,
    )


if __name__ == "__main__":
    main()
