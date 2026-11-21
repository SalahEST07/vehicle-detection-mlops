
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "64576917d0154403beb6d938ff511577"

client = MlflowClient()

print("Downloading best.pt from MLflow...")

model_path = client.download_artifacts(
    run_id=RUN_ID,
    path="model/best.pt"
)

print("Loading YOLO26n...")
model = YOLO(model_path, task="detect")

print("\nExporting to ONNX...")

onnx_path = model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=17
)

print("\nExport completed!")
print("ONNX model:", onnx_path)
print("File exists:", os.path.isfile(onnx_path))

if os.path.isfile(onnx_path):
    print("File size:", round(os.path.getsize(onnx_path) / (1024 * 1024), 2), "MB")
