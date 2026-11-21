
from pathlib import Path
from ultralytics import YOLO


# Project root: G:\vehicle-detection-mlops
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_best.onnx"

# Load the ONNX model
model = YOLO(
    str(MODEL_PATH),
    task="detect"
)

def predict_image(image_path: str, confidence: float = 0.25):
    """
    Run vehicle detection on an image.
    """

    results = model.predict(
        source=image_path,
        conf=confidence,
        imgsz=640,
         device="cpu",
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            score = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(score, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

    return detections
