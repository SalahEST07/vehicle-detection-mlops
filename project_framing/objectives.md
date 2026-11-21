
# Project: Vehicle Detection and Classification

## 1. Problem Statement

Vehicle detection is an important computer vision
task with applications in intelligent transportation
systems and traffic monitoring.

The objective of this project is to develop a
computer vision application capable of detecting
and classifying four types of vehicles:

- Car
- Motorcycle
- Bus
- Truck

The system will use a pretrained YOLO model and
transfer learning to perform object detection.

The final application will support image uploads,
video processing and webcam-based detection.

The solution will be industrialized using:

- DVC for data versioning
- MLflow for experiment tracking and model registry
- ONNX for model export and optimized inference
- FastAPI for the REST API
- React for the frontend
- Docker Compose for containerization
- GitHub Actions for CI/CD

## 2. General Objective

Develop and industrialize a vehicle detection
application using deep learning and MLOps tools.

## 3. Specific Objectives

1. Identify and prepare a suitable vehicle dataset.
2. Analyze dataset quality and class distribution.
3. Fine-tune a pretrained YOLO model.
4. Evaluate model performance using precision,
   recall and mean Average Precision.
5. Deploy the model through a web application
   supported by an MLOps pipeline.
