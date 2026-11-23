from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import tempfile
import os
import uuid

from backend.app.services.inference import predict_image, predict_video
app = FastAPI(
    title="Vehicle Detection API",
    description="REST API for detecting cars, motorcycles, buses, and trucks using YOLO26n.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "outputs" / "videos"

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

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


class VideoPredictionResponse(BaseModel):
    filename: str
    processed_frames: int
    detection_counts: dict
    output_filename: str
    video_url: str


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


@app.get("/videos/{filename}")
async def get_video(filename: str):
    video_path = OUTPUT_VIDEO_DIR / filename

    if not video_path.is_file() or video_path.parent != OUTPUT_VIDEO_DIR:
        raise HTTPException(
            status_code=404,
            detail="Video not found."
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=filename
    )

@app.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.0, le=1.0)
):
    allowed_types = [
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "video/quicktime"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, AVI, and MOV videos are supported."
        )

    suffix = Path(file.filename or "video.mp4").suffix.lower()

    if suffix not in [".mp4", ".avi", ".mov"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video file extension."
        )

    input_path = None
    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_VIDEO_DIR / output_filename

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            input_path = temp_file.name

            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        result = predict_video(
            video_path=input_path,
            output_path=str(output_path),
            confidence=confidence
        )

        return {
            "filename": file.filename,
            "processed_frames": result["processed_frames"],
            "detection_counts": result["detection_counts"],
            "output_filename": output_filename,
            "video_url": f"http://127.0.0.1:8000/videos/{output_filename}"
        }

    except Exception as error:
        if output_path.exists():
            output_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Video prediction failed: {str(error)}"
        )

    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        await file.close()

