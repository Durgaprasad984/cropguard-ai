# YOLOv8 Bounding-Box Detection Upgrade

This project now supports a real YOLOv8 detection model when `backend/app/best_det.pt` exists.

## Dataset

The detection dataset is PlantDoc Object Detection:

- Source: `pratikkayal/PlantDoc-Object-Detection-Dataset`
- Classes: 29 plant leaf/disease classes
- Format converted to YOLOv8 under `datasets/plantdoc_yolo`

Prepare or resume the dataset:

```powershell
python training\prepare_plantdoc_detection.py
```

The downloader is resumable. Some upstream filenames are invalid on Windows, so the script saves safe local filenames and writes matching YOLO labels.

## Train Detection

Download YOLOv8 detection base weights and place them at:

```text
yolov8n.pt
```

Then train:

```powershell
python training\train_detection.py
```

The best detection model is copied to:

```text
backend/app/best_det.pt
```

After that, the backend automatically uses detection mode and returns real bounding boxes plus an annotated image.

## Backend Behavior

- If `backend/app/best_det.pt` exists: uses YOLOv8 detection with bounding boxes.
- If not: falls back to the existing classifier `backend/app/best.pt`.
- Non-leaf images are rejected before prediction.

## Accuracy Notes

PlantDoc is a real-world bounding-box dataset, but it is smaller and noisier than PlantVillage. For better accuracy:

- train on GPU if possible
- use `yolov8s.pt` or `yolov8m.pt` instead of `yolov8n.pt`
- increase epochs
- add your own field photos and annotations
