
import onnx
import onnxruntime as ort
import os

MODEL_PATH = "models/yolo26n_best.onnx"

print("Checking ONNX file...")

# 1. Check file
assert os.path.isfile(MODEL_PATH), "ONNX file not found!"

# 2. Validate ONNX structure
model = onnx.load(MODEL_PATH)
onnx.checker.check_model(model)

print("ONNX structure: VALID")

# 3. Load with ONNX Runtime
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("\nONNX Runtime loading: SUCCESS")

# 4. Inspect input and output
print("\nModel inputs:")
for item in session.get_inputs():
    print("Name:", item.name)
    print("Shape:", item.shape)
    print("Type:", item.type)

print("\nModel outputs:")
for item in session.get_outputs():
    print("Name:", item.name)
    print("Shape:", item.shape)
    print("Type:", item.type)

print("\nONNX validation completed successfully!")
