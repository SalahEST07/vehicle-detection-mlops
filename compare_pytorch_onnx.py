
import mlflow
import numpy as np
import onnxruntime as ort
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "64576917d0154403beb6d938ff511577"
IMAGE_PATH = "bus.jpg"
ONNX_PATH = "models/yolo26n_best.onnx"

# Download original PyTorch weights
client = MlflowClient()
pt_path = client.download_artifacts(
    run_id=RUN_ID,
    path="model/best.pt"
)

# PyTorch inference
print("Running PyTorch inference...")
pt_model = YOLO(pt_path, task="detect")
pt_results = pt_model.predict(
    IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    verbose=False
)

pt_detections = []

for box in pt_results[0].boxes:
    pt_detections.append({
        "class": pt_model.names[int(box.cls[0])],
        "confidence": float(box.conf[0])
    })

# ONNX Runtime inference through Ultralytics
print("\nRunning ONNX inference...")
onnx_model = YOLO(ONNX_PATH, task="detect")
onnx_results = onnx_model.predict(
    IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    verbose=False
)

onnx_detections = []

for box in onnx_results[0].boxes:
    onnx_detections.append({
        "class": onnx_model.names[int(box.cls[0])],
        "confidence": float(box.conf[0])
    })

# Compare results
print("\n--- PYTORCH DETECTIONS ---")
for detection in pt_detections:
    print(detection)

print("\n--- ONNX DETECTIONS ---")
for detection in onnx_detections:
    print(detection)

print("\nDetection counts:")
print("PyTorch:", len(pt_detections))
print("ONNX:", len(onnx_detections))
