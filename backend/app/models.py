"""Pydantic models for API request/response validation"""

from pydantic import BaseModel
from typing import List, Optional, Dict


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SinglePrediction(BaseModel):
    class_name: str
    confidence: float
    confidence_pct: str
    severity: str
    description: str
    treatment: List[str]
    prevention: List[str]
    bbox: Optional[BoundingBox] = None


class PredictionResponse(BaseModel):
    success: bool
    model_type: str                        # "detection" or "classification"
    num_detections: int
    predictions: List[SinglePrediction]
    annotated_image: Optional[str] = None  # base64 JPEG with drawn boxes
    top_disease: Optional[str] = None
    message: Optional[str] = None
    rejection_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
