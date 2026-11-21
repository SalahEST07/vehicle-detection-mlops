
import os
from pathlib import Path
from multiprocessing import freeze_support

# Configure MLflow before importing Ultralytics
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Vehicle_Detection_COCO"
os.environ["MLFLOW_RUN"] = "YOLO26n_baseline"

from ultralytics import YOLO, settings


def main():
    # Enable MLflow integration
    settings.update({"mlflow": True})

    # Project paths
    project_dir = Path(__file__).resolve().parent.parent
    data_yaml = project_dir / "data" / "vehicle.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    print("Loading pretrained YOLO26n model...")

    model = YOLO("yolo26n.pt")

    print("Starting YOLO26n training...")

    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=15,
        seed=42,
        deterministic=True,
        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        project=str(project_dir / "training" / "results"),
        name="yolo26n_baseline",
        exist_ok=True
    )

    print("\nTraining completed!")

    weights_dir = (
        project_dir
        / "training"
        / "results"
        / "yolo26n_baseline"
        / "weights"
    )

    print("Best model:", weights_dir / "best.pt")
    print("Last model:", weights_dir / "last.pt")


if __name__ == "__main__":
    freeze_support()
    main()
