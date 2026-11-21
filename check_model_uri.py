
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "64576917d0154403beb6d938ff511577"

client = mlflow.tracking.MlflowClient()

def show_artifacts(path=""):
    artifacts = client.list_artifacts(RUN_ID, path)

    for artifact in artifacts:
        print(
            f"{artifact.path} | "
            f"Directory: {artifact.is_dir} | "
            f"Size: {artifact.file_size}"
        )

        if artifact.is_dir:
            show_artifacts(artifact.path)

print("MLflow model artifacts:")
show_artifacts("model")
