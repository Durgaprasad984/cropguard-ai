"""YOLOv8 inference for crop disease classification."""

import base64
import io
import os

import numpy as np
from PIL import Image, ImageDraw

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "best.pt")
DEFAULT_DETECTION_MODEL_PATH = os.path.join(APP_DIR, "best_det.pt")
YOLO_CONFIG_DIR = os.path.join(APP_DIR, ".ultralytics")
MPL_CONFIG_DIR = os.path.join(APP_DIR, ".matplotlib")

os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

from ultralytics import YOLO

from app.disease_info import CLASS_NAMES, get_disease_info

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    DEFAULT_DETECTION_MODEL_PATH if os.path.exists(DEFAULT_DETECTION_MODEL_PATH) else DEFAULT_MODEL_PATH,
)
REJECT_CLASS = "x_Removed_from_Healthy_leaves"
VALID_CLASS_NAMES = set(CLASS_NAMES)

MIN_PLANT_PIXEL_RATIO = 0.08
MIN_TOP_CONFIDENCE = 0.55
MIN_CONFIDENCE_MARGIN = 0.12
DETECTION_CONFIDENCE = 0.25

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    if getattr(model, "task", None) not in {"classify", "detect"}:
        raise RuntimeError(f"Expected YOLOv8 classify/detect model, got task={getattr(model, 'task', None)}")

    print(f"Loaded YOLOv8 crop {model.task} model from {MODEL_PATH}")
except Exception as exc:
    raise RuntimeError(f"Could not load crop disease model: {exc}") from exc


def unsupported_response(reason: str, message: str) -> dict:
    return {
        "success": False,
        "model_type": "classification",
        "num_detections": 0,
        "predictions": [],
        "annotated_image": None,
        "top_disease": None,
        "message": message,
        "rejection_reason": reason,
    }


def plant_pixel_ratio(image: Image.Image) -> float:
    """
    Estimate whether the image contains enough leaf-like color.
    This only rejects obvious out-of-dataset objects; YOLO still decides the disease class.
    """
    small = image.convert("RGB").resize((160, 160))
    arr = np.asarray(small).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    saturation = arr.max(axis=-1) - arr.min(axis=-1)

    green = (g > r * 0.90) & (g > b * 1.05) & (g > 0.18) & (saturation > 0.08)
    yellow = (r > 0.35) & (g > 0.28) & (b < 0.35) & (np.abs(r - g) < 0.25) & (saturation > 0.08)
    brown = (r > 0.20) & (g > 0.12) & (b < 0.22) & (r >= g * 0.90) & (g >= b * 1.10) & (saturation > 0.06)

    return float((green | yellow | brown).mean())


def get_class_name(index: int) -> str:
    if hasattr(model, "names") and index in model.names:
        return model.names[index]
    if index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return f"Class_{index}"


def draw_boxes(image: Image.Image, boxes, class_names) -> str:
    """Draw bounding boxes for future detection models and return a base64 JPEG."""
    draw = ImageDraw.Draw(image)
    colors = ["#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#3b82f6"]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label_name = get_class_name(cls_id) if cls_id < len(class_names) else "Unknown"
        label = f"{label_name} {conf:.0%}"
        color = colors[i % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 3, y1 - 17), label, fill="white")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()


def prediction_payload(class_name: str, confidence: float) -> dict:
    info = get_disease_info(class_name)
    return {
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence:.1%}",
        "severity": info.get("severity", "Unknown"),
        "treatment": info.get("treatment", []),
        "prevention": info.get("prevention", []),
        "description": info.get("description", ""),
    }


def detection_payload(class_name: str, confidence: float, box) -> dict:
    payload = prediction_payload(class_name, confidence)
    payload["bbox"] = {
        "x1": int(box.xyxy[0][0]),
        "y1": int(box.xyxy[0][1]),
        "x2": int(box.xyxy[0][2]),
        "y2": int(box.xyxy[0][3]),
    }
    return payload


def run_inference(image: Image.Image, raw_bytes: bytes) -> dict:
    """Run YOLOv8 classification and return a structured prediction response."""
    del raw_bytes

    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    if plant_pixel_ratio(image) < MIN_PLANT_PIXEL_RATIO:
        return unsupported_response(
            "not_leaf_image",
            "Please upload a clear crop leaf image. Non-leaf objects are not supported by this PlantVillage model.",
        )

    result = model(image, conf=DETECTION_CONFIDENCE, verbose=False)[0]
    predictions = []

    if getattr(model, "task", None) == "detect":
        if result.boxes is None or len(result.boxes) == 0:
            return unsupported_response(
                "no_leaf_detection",
                "No supported plant leaf was detected. Try a clearer photo with the leaf filling more of the image.",
            )

        annotated_img = image.copy()
        b64_annotated = draw_boxes(annotated_img, result.boxes, list(model.names.values()))

        for box in result.boxes:
            class_name = get_class_name(int(box.cls[0]))
            confidence = float(box.conf[0])
            predictions.append(detection_payload(class_name, confidence, box))

        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "success": True,
            "model_type": "detection",
            "num_detections": len(predictions),
            "predictions": predictions,
            "annotated_image": b64_annotated,
            "top_disease": predictions[0]["class_name"] if predictions else None,
            "message": "Detection completed.",
            "rejection_reason": None,
        }

    if result.probs is None:
        return unsupported_response(
            "no_classification",
            "The loaded model did not return classification probabilities.",
        )

    probs = result.probs.data.tolist()
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    top_idx = top_indices[0]
    second_idx = top_indices[1] if len(top_indices) > 1 else None
    top_conf = float(probs[top_idx])
    second_conf = float(probs[second_idx]) if second_idx is not None else 0.0
    top_class = get_class_name(top_idx)

    if top_class == REJECT_CLASS or top_class not in VALID_CLASS_NAMES:
        return unsupported_response(
            "unsupported_class",
            "This image does not match a supported PlantVillage crop disease class.",
        )

    if top_conf < MIN_TOP_CONFIDENCE or (top_conf - second_conf) < MIN_CONFIDENCE_MARGIN:
        return unsupported_response(
            "low_confidence",
            "The model is not confident enough to identify this leaf. Try a closer, brighter image of one leaf.",
        )

    for idx in top_indices:
        conf = float(probs[idx])
        class_name = get_class_name(idx)
        if conf < 0.01 or class_name not in VALID_CLASS_NAMES:
            continue
        predictions.append(prediction_payload(class_name, conf))

    return {
        "success": True,
        "model_type": "classification",
        "num_detections": len(predictions),
        "predictions": predictions,
        "annotated_image": None,
        "top_disease": predictions[0]["class_name"] if predictions else None,
        "message": "Prediction completed.",
        "rejection_reason": None,
    }
