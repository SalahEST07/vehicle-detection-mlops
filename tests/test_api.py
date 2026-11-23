
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Vehicle Detection API is running"
    assert response.json()["classes"] == [
        "car", "motorcycle", "bus", "truck"
    ]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_reject_unsupported_image_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"test content", "text/plain")}
    )

    assert response.status_code == 400
    assert "Only JPEG, PNG, and WEBP" in response.json()["detail"]


def test_reject_invalid_confidence():
    response = client.post(
        "/predict?confidence=1.5",
        files={"file": ("test.jpg", b"fake image", "image/jpeg")}
    )

    assert response.status_code == 422


def test_image_prediction_with_mocked_model(monkeypatch):
    from backend.app import main

    fake_detections = [
        {
            "class_id": 0,
            "class_name": "car",
            "confidence": 0.95,
            "bbox": {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 120.0
            }
        }
    ]

    monkeypatch.setattr(
        main,
        "predict_image",
        lambda image_path, confidence: fake_detections
    )

    response = client.post(
        "/predict",
        files={"file": ("car.jpg", b"fake image", "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json()["total_detections"] == 1
    assert response.json()["detections"][0]["class_name"] == "car"

def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "average_request_time_seconds" in data
    assert "detection_counts" in data

    assert isinstance(data["detection_counts"], dict)

    assert set(data["detection_counts"].keys()) == {
        "car", "motorcycle", "bus", "truck"
    }