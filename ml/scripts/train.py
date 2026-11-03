"""Fine-tune YOLOv8 Nano on the golf ball dataset.

Run download_roboflow.py first so ml/data/data.yaml exists.
"""

import argparse
import os

from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.dirname(__file__), "..", "data", "data.yaml")
BASE_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    # Ultralytics' auto device-selection misdetects CPU on some torch/Python
    # combos even when MPS is available (confirmed working via a direct
    # torch.mps conv2d test) — force it explicitly rather than relying on
    # `device=None` auto-detection.
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found — run download_roboflow.py first."
        )

    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache="ram",
        project=os.path.join(os.path.dirname(__file__), "..", "models"),
        name="golf_ball_yolov8n",
    )


if __name__ == "__main__":
    main()
