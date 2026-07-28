#!/usr/bin/env python3
"""Export trained YOLO weights to Core ML (ADR 0007).

Usage:
    python export_coreml.py models/golfball-yolov8n/weights/best.pt
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = model.export(format="coreml", imgsz=args.imgsz, nms=True)
    print(f"exported: {exported}")
    print("Copy the .mlpackage into app/ when it's the one we ship.")


if __name__ == "__main__":
    main()
