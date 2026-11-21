
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "data" / "vehicle.yaml"

print("=== DATASET CHECK ===")

data = check_det_dataset(str(DATA_YAML))

print("Dataset configuration: OK")
print("Training images:", data["train"])
print("Validation images:", data["val"])
print("Number of classes:", data["nc"])
print("Class names:", data["names"])

assert data["nc"] == 4, "Expected exactly 4 classes"

print("\n=== GPU CHECK ===")

assert torch.cuda.is_available(), "CUDA is not available"

gpu = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print("GPU:", gpu)
print(f"Total GPU memory: {memory:.2f} GB")
print(f"Currently allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.1f} MB")

print("\n=== MODEL CHECK ===")

model = YOLO("yolo26n.pt")

print("Model loaded successfully")
print("Model task:", model.task)

print("\nAll preflight checks passed!")
