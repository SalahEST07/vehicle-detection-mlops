
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import tempfile
import os

from backend.app.services.inference import predict_image

app = FastAPI(
    title="Vehicle Detection API",
    description="REST API for detecting cars, motorcycles, buses, and trucks using YOLO26n.",
    version="1.0.0"
)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    filename: str
    total_detections: int
    detections: List[Detection]


@app.get("/")
async def root():
    return {
        "message": "Vehicle Detection API is running",
        "model": "YOLO26n",
        "classes": ["car", "motorcycle", "bus", "truck"]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.0, le=1.0)
):
    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WEBP images are supported."
        )

    suffix = Path(file.filename or "image.jpg").suffix.lower()

    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image file extension."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        detections = predict_image(temp_path, confidence)

        return {
            "filename": file.filename,
            "total_detections": len(detections),
            "detections": detections
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {str(error)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
