
import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "64576917d0154403beb6d938ff511577"
IMAGE_PATH = "bus.jpg"

client = MlflowClient()

# Download the registered model's original artifact
model_path = client.download_artifacts(
    run_id=RUN_ID,
    path="model/best.pt"
)

# Load YOLO model
model = YOLO(model_path, task="detect")

# Run inference
print("Running vehicle detection...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=True
)

# Display detection results
for result in results:
    print("\nDetected objects:")

    if len(result.boxes) == 0:
        print("No vehicles detected.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                f"{model.names[class_id]} "
                f"- Confidence: {confidence:.3f}"
            )

print("\nInference completed!")
print("Annotated image saved in the Ultralytics runs/detect directory.")
