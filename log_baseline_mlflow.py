
import os
import mlflow

PROJECT_DIR = r"G:\vehicle-detection-mlops"

RUN_DIR = os.path.join(
    PROJECT_DIR,
    "training",
    "results",
    "yolo26n_baseline"
)

WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")

# Connect to the running MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Vehicle_Detection_COCO")

# Training parameters
params = {
    "model": "YOLO26n",
    "epochs": 50,
    "image_size": 640,
    "batch_size": 16,
    "seed": 42,
    "optimizer": "Ultralytics default",
    "dataset": "COCO 2017 vehicle subset",
    "classes": "car,motorcycle,bus,truck",
    "training_device": "NVIDIA RTX 5060 Ti",
    "ultralytics_version": "8.4.198"
}

# Validation metrics from the completed baseline
metrics = {
    "precision": 0.731,
    "recall": 0.628,
    "mAP50": 0.698,
    "mAP50_95": 0.494
}

with mlflow.start_run(run_name="YOLO26n_baseline"):

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Log the best trained model weights
    best_model = os.path.join(WEIGHTS_DIR, "best.pt")
    if os.path.exists(best_model):
        mlflow.log_artifact(best_model, artifact_path="model")
    else:
        raise FileNotFoundError(best_model)

    # Log the last checkpoint if available
    last_model = os.path.join(WEIGHTS_DIR, "last.pt")
    if os.path.exists(last_model):
        mlflow.log_artifact(last_model, artifact_path="model")

    # Log evaluation and training artifacts
    artifact_names = [
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "args.yaml"
    ]

    for filename in artifact_names:
        path = os.path.join(RUN_DIR, filename)
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path="evaluation")

    print("Baseline successfully logged to MLflow!")
    print("Run ID:", mlflow.active_run().info.run_id)
