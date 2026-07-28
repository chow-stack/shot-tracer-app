#!/usr/bin/env python3
"""Download the Roboflow GolfBall dataset in YOLOv8 format.

Usage:
    ROBOFLOW_API_KEY=<key> python scripts/download_roboflow.py

Defaults to the public anna-gaming/golfball dataset; pass --workspace/--project
if downloading from a fork instead. --version must match a version number shown
on the dataset's Roboflow Universe page.
"""

import argparse
import os
import sys

from roboflow import Roboflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="anna-gaming")
    parser.add_argument("--project", default="golfball")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--location", default="data/roboflow-golfball")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY (from your Roboflow account settings).")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=args.location)
    print(f"downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
