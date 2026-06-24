"""Train a YOLOv8 detection model on PlantDoc bounding-box data."""

import os
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "plantdoc_yolo" / "data.yaml"
BEST_OUTPUT = ROOT / "backend" / "app" / "best_det.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "backend" / "app" / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "backend" / "app" / ".matplotlib"))

from ultralytics import YOLO

CONFIG = {
    "model": os.getenv("DETECTION_BASE_MODEL", str(ROOT / "yolov8n.pt")),
    "epochs": 5,
    "imgsz": 320,
    "batch": 4,
    "device": "cpu",
    "workers": 0,
    "project": str(ROOT / "runs" / "detect"),
    "name": "plantdoc_yolov8n_fast30",
    "patience": 3,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "cos_lr": True,
    "fraction": 0.25,
}


def train(config: dict) -> Path:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Run: python training/prepare_plantdoc_detection.py"
        )
    if not Path(config["model"]).exists() and config["model"].endswith(".pt"):
        raise FileNotFoundError(
            f"{config['model']} not found. Download YOLOv8 detection weights as yolov8n.pt "
            "into the project root, or set DETECTION_BASE_MODEL to a local detection .pt/.yaml file."
        )

    model = YOLO(config["model"])
    results = model.train(
        data=str(DATA_YAML),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        workers=config["workers"],
        project=config["project"],
        name=config["name"],
        patience=config["patience"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        cos_lr=config["cos_lr"],
        fraction=config["fraction"],
        save=True,
        plots=True,
        verbose=True,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    BEST_OUTPUT.write_bytes(best_model.read_bytes())
    print(f"Best detection model copied to {BEST_OUTPUT}")
    return BEST_OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantDoc YOLOv8 detection model.")
    parser.add_argument("--quick", action="store_true", help="Very fast smoke run: 3 epochs, 320px, small data subset.")
    parser.add_argument("--better", action="store_true", help="Slower but better CPU run: 10 epochs, 416px, full data.")
    parser.add_argument("--full", action="store_true", help="Best CPU run: 20 epochs, 512px, full data.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--fraction", type=float, help="Override dataset fraction, for example 0.25 or 1.0.")
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.quick:
        config.update({"epochs": 3, "imgsz": 320, "batch": 2, "fraction": 0.10, "patience": 2, "name": "plantdoc_yolov8n_quick"})
    if args.better:
        config.update({"epochs": 10, "imgsz": 416, "batch": 4, "fraction": 1.0, "patience": 5, "name": "plantdoc_yolov8n_better"})
    if args.full:
        config.update({"epochs": 20, "imgsz": 512, "batch": 4, "fraction": 1.0, "patience": 8, "name": "plantdoc_yolov8n_full"})
    if args.epochs:
        config["epochs"] = args.epochs
    if args.imgsz:
        config["imgsz"] = args.imgsz
    if args.batch:
        config["batch"] = args.batch
    if args.fraction:
        config["fraction"] = args.fraction

    print("Training settings:")
    for key in ["model", "epochs", "imgsz", "batch", "fraction", "device", "workers", "name"]:
        print(f"  {key}: {config[key]}")

    train(config)
