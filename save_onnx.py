
import os
import shutil

source = r"C:\Users\SBS\AppData\Local\Temp\tmplgo06v2i\best.onnx"

destination_dir = "models"
destination = os.path.join(destination_dir, "yolo26n_best.onnx")

os.makedirs(destination_dir, exist_ok=True)

if not os.path.isfile(source):
    raise FileNotFoundError(f"ONNX source not found: {source}")

shutil.copy2(source, destination)

print("ONNX model saved successfully!")
print("Destination:", os.path.abspath(destination))
print("Size:", round(os.path.getsize(destination) / (1024 * 1024), 2), "MB")
