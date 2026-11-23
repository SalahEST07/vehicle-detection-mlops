from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import time
import threading
from collections import Counter
import tempfile
import os
import uuid
from backend.app.services.inference import predict_image, predict_video




# =========================
# Inference Monitoring
# =========================

monitoring_data = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_request_time": 0.0,
    "detection_counts": Counter({
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    })
}

monitoring_lock = threading.Lock()


def record_request(success: bool, duration: float, detections=None):
    with monitoring_lock:
        monitoring_data["total_requests"] += 1

        if success:
            monitoring_data["successful_requests"] += 1
        else:
            monitoring_data["failed_requests"] += 1

        monitoring_data["total_request_time"] += duration

        if detections:
            for detection in detections:
                class_name = detection.get("class_name")

                if class_name in monitoring_data["detection_counts"]:
                    monitoring_data["detection_counts"][class_name] += 1
app = FastAPI(
    title="Vehicle Detection API",
    description="REST API for detecting cars, motorcycles, buses, and trucks using YOLO26n.",
    version="1.0.0"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_VIDEO_DIR = PROJECT_ROOT / "outputs" / "videos"

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/metrics")
async def get_metrics():
    with monitoring_lock:
        total = monitoring_data["total_requests"]

        average_time = (
            monitoring_data["total_request_time"] / total
            if total > 0 else 0
        )

        return {
            "total_requests": total,
            "successful_requests": monitoring_data["successful_requests"],
            "failed_requests": monitoring_data["failed_requests"],
            "average_request_time_seconds": round(average_time, 4),
            "detection_counts": dict(monitoring_data["detection_counts"])
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.0, le=1.0)
):
    temp_path = None
    start_time = time.time()
    request_recorded = False

    try:
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        detections = predict_image(temp_path, confidence)

        record_request(
            success=True,
            duration=time.time() - start_time,
            detections=detections
        )
        request_recorded = True

        return {
            "filename": file.filename,
            "total_detections": len(detections),
            "detections": detections
        }

    except Exception as error:
        if not request_recorded:
            record_request(
                success=False,
                duration=time.time() - start_time
            )

        if isinstance(error, HTTPException):
            raise error

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
    input_path = None
    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_VIDEO_DIR / output_filename
    start_time = time.time()
    request_recorded = False

    try:
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            input_path = temp_file.name

            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        result = predict_video(
            video_path=input_path,
            output_path=str(output_path),
            confidence=confidence
        )

        # Record successful video request
        record_request(
            success=True,
            duration=time.time() - start_time,
            detections=[
                {"class_name": class_name}
                for class_name, count in result["detection_counts"].items()
                for _ in range(count)
            ]
        )
        request_recorded = True

        return {
            "filename": file.filename,
            "processed_frames": result["processed_frames"],
            "detection_counts": result["detection_counts"],
            "output_filename": output_filename,
            "video_url": f"http://127.0.0.1:8000/videos/{output_filename}"
        }

    except Exception as error:
        if not request_recorded:
            record_request(
                success=False,
                duration=time.time() - start_time
            )

        if output_path.exists():
            output_path.unlink()

        if isinstance(error, HTTPException):
            raise error

        raise HTTPException(
            status_code=500,
            detail=f"Video prediction failed: {str(error)}"
        )

    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        await file.close()


