
import mlflow

# Connect to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create or retrieve our experiment
experiment_name = "Vehicle_Detection_COCO"
experiment_id = mlflow.set_experiment(experiment_name).experiment_id

# Create a small test run
with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("project", "Vehicle Detection")
    mlflow.log_param("model", "YOLO26n")
    mlflow.log_metric("connection_test", 1)

print("MLflow connection successful!")
print("Experiment ID:", experiment_id)
print("Tracking URI:", mlflow.get_tracking_uri())
