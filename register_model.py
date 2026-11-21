
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "64576917d0154403beb6d938ff511577"
MODEL_NAME = "Vehicle_Detection_YOLO26n"

client = MlflowClient()

# Get the existing baseline run
run = client.get_run(RUN_ID)

# Build the exact artifact URI
model_source = (
    f"{run.info.artifact_uri}/model/best.pt"
)

# Create a model version from the existing artifact
version = client.create_model_version(
    name=MODEL_NAME,
    source=model_source,
    run_id=RUN_ID
)

print("\nModel version created!")
print("Model name:", MODEL_NAME)
print("Version:", version.version)
print("Status:", version.status)
print("Source:", version.source)
