"""
YOLOv8 Training Script — Crop Disease Detection
=========================================================
Dataset: PlantVillage (via Roboflow or direct Kaggle download)
Model:   YOLOv8n-cls (classification) or YOLOv8n (detection)

Run on Google Colab with T4 GPU for fastest training.
Expected training time: ~1–2 hours for 50 epochs on T4.

SETUP STEPS:
1. Download dataset (see bottom of file)
2. Organize into train/val/test folders
3. Run: python train.py
"""

from ultralytics import YOLO
import yaml
import os
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Model: yolov8n-cls (fastest), yolov8s-cls, yolov8m-cls (best accuracy)
    "model":       "yolov8n-cls.pt",    # classification model
    "data_dir":  "datasets/dataset_split", # path to your train/val dataset
    "epochs":      30,
    "imgsz":  224,               # image size (224 or 256 for cls)
    "batch":       32,
    "lr0":         0.01,
    "fraction":    1.0,
    "patience":    10,                   # early stopping patience
    "project":     "runs/train",
    "name":        "crop_disease_v1",
    "device":      "cpu",                  # "0" for GPU, "cpu" for CPU
    "workers":     4,
    "optimizer":   "AdamW",
    "cos_lr":      True,                 # cosine LR schedule
    "augment":     True,
    "degrees":     15.0,                 # rotation augmentation
    "flipud":      0.1,
    "fliplr":      0.5,
    "hsv_h":       0.015,
    "hsv_s":       0.7,
    "hsv_v":       0.4,
}

# ── Dataset preparation ────────────────────────────────────────────────────────

def prepare_plantvillage_dataset(source_dir: str, output_dir: str, val_split: float = 0.15):
    """
    Organize PlantVillage dataset into YOLOv8 classification format:
    dataset/
      train/
        Tomato___Early_blight/  [*.jpg ...]
        Tomato___healthy/       [*.jpg ...]
        ...
      val/
        Tomato___Early_blight/  [*.jpg ...]
        ...
    """
    import random
    from pathlib import Path

    source = Path(source_dir)
    output = Path(output_dir)

    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} disease classes")

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
                 list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))

        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_split))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = output / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {class_dir.name}: {len(train_imgs)} train / {len(val_imgs)} val")

    print(f"\n✅ Dataset prepared at: {output_dir}")


# ── Training ────────────────────────────────────────────────────────────────────

def train():
    print("🌿 Starting YOLOv8 Crop Disease Detection Training")
    print("=" * 55)

    # Initialize model
    model = YOLO(CONFIG["model"])

    # Train
    results = model.train(
        data=CONFIG["data_dir"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        lr0=CONFIG["lr0"],
        patience=CONFIG["patience"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        device=CONFIG["device"],
        fraction=CONFIG["fraction"],
        workers=CONFIG["workers"],
        optimizer=CONFIG["optimizer"],
        cos_lr=CONFIG["cos_lr"],
        augment=CONFIG["augment"],
        degrees=CONFIG["degrees"],
        flipud=CONFIG["flipud"],
        fliplr=CONFIG["fliplr"],
        hsv_h=CONFIG["hsv_h"],
        hsv_s=CONFIG["hsv_s"],
        hsv_v=CONFIG["hsv_v"],
        verbose=True,
        save=True,
        plots=True,
    )

    print("\n✅ Training complete!")
    best_model_path = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"
    print(f"📦 Best model saved at: {best_model_path}")
    return results, best_model_path


def evaluate(model_path: str):
    """Evaluate model on validation set."""
    model = YOLO(model_path)
    metrics = model.val(data=CONFIG["data_dir"])
    print(f"\n📊 Validation Results:")
    print(f"   Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"   Top-5 Accuracy: {metrics.top5:.4f}")
    return metrics


def export_model(model_path: str):
    """Export model to ONNX for deployment."""
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=CONFIG["imgsz"], simplify=True)
    print(f"✅ Model exported to ONNX format")


# ── Dataset download helpers ───────────────────────────────────────────────────

def download_from_kaggle():
    """
    Download PlantVillage dataset from Kaggle.
    Requires kaggle API token at ~/.kaggle/kaggle.json

    HOW TO SET UP:
    1. Go to kaggle.com → Account → Create API Token
    2. Place kaggle.json at ~/.kaggle/kaggle.json
    3. Run this function
    """
    import subprocess
    print("📥 Downloading PlantVillage dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "emmarex/plantdisease",
        "--unzip", "-p", "raw_dataset/"
    ], check=True)
    print("✅ Dataset downloaded to raw_dataset/")
    print("   Now run: prepare_plantvillage_dataset('raw_dataset/PlantVillage', 'dataset/')")


def download_from_roboflow(api_key: str, workspace: str, project: str, version: int):
    """
    Download a custom labeled dataset from Roboflow.
    Go to roboflow.com → your project → Export → YOLOv8 → Show download code
    """
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")
    print(f"✅ Dataset downloaded to: {dataset.location}")
    return dataset.location


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Crop Disease Trainer")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train",   action="store_true", help="Train model")
    parser.add_argument("--eval",    type=str, help="Evaluate model (provide path to best.pt)")
    parser.add_argument("--export",  type=str, help="Export to ONNX (provide path to best.pt)")
    parser.add_argument("--source",  type=str, default="raw_dataset/PlantVillage")
    parser.add_argument("--output",  type=str, default="dataset/")
    args = parser.parse_args()

    if args.prepare:
        prepare_plantvillage_dataset(args.source, args.output)

    if args.train:
        results, best_path = train()
        evaluate(best_path)

    if args.eval:
        evaluate(args.eval)

    if args.export:
        export_model(args.export)

    if not any([args.prepare, args.train, args.eval, args.export]):
        print("Usage:")
        print("  python train.py --prepare --source raw_dataset/PlantVillage --output dataset/")
        print("  python train.py --train")
        print("  python train.py --eval runs/train/crop_disease_v1/weights/best.pt")
        print("  python train.py --export runs/train/crop_disease_v1/weights/best.pt")
