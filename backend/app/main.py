"""
Crop Disease Detection API - FastAPI Backend
YOLOv8-powered plant disease detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
from PIL import Image
import numpy as np

from app.inference import run_inference
from app.models import PredictionResponse, HealthResponse

app = FastAPI(
    title="🌿 Crop Disease Detection API",
    description="YOLOv8-powered crop disease detection using PlantVillage dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-deployed-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    return {"status": "ok", "message": "Crop Disease Detection API is running 🌿"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "message": "Model loaded and ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a crop/plant image and get disease predictions.
    Accepts: JPEG, PNG, WEBP
    Returns: Disease class, confidence, bounding boxes, treatment tips
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload JPEG, PNG, or WEBP."
        )

    # Read and validate image
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Run YOLOv8 inference
    result = run_inference(image, contents)
    return result


@app.post("/predict/base64")
async def predict_base64(payload: dict):
    """Accept base64-encoded image (for web clients that prefer JSON)"""
    try:
        image_data = base64.b64decode(payload["image"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        result = run_inference(image, image_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
