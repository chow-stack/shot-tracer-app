"""Download the golf ball detection dataset from Roboflow Universe.

Requires ROBOFLOW_API_KEY to be set in the shell environment (see NOTES.md).
"""

import os
import sys

from roboflow import Roboflow

DATASET_WORKSPACE = "golf-ball-detection"
DATASET_PROJECT = "golf-ball-detection-hii2e"
DATASET_VERSION = 2
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set. Export it first:")
        print("  export ROBOFLOW_API_KEY=your-key-here")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(DATASET_WORKSPACE).project(DATASET_PROJECT)
    version = project.version(DATASET_VERSION)
    # overwrite=True: the roboflow SDK skips downloading if `location` already
    # exists (e.g. our .gitkeep placeholder), so without this it silently no-ops.
    dataset = version.download("yolov8", location=OUTPUT_DIR, overwrite=True)

    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
