
from pathlib import Path
from ultralytics import YOLO


# Project root: G:\vehicle-detection-mlops
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_best.onnx"

# Lazy model loading
model = None


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}"
            )

        model = YOLO(
            str(MODEL_PATH),
            task="detect"
        )

    return model

def predict_image(image_path: str, confidence: float = 0.25):
    """
    Run vehicle detection on an image.
    """
    model = get_model()
    results = model.predict(
        source=image_path,
        conf=confidence,
        imgsz=640,
         device="cpu",
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            score = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(score, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })




    return detections





def predict_video(video_path: str, output_path: str, confidence: float = 0.25):
    """
    Track vehicles across video frames using ByteTrack.
    Assign each tracking ID a stable class using majority voting.
    """
    model = get_model()
    import cv2
    import csv
    from collections import Counter, defaultdict

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video properties.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video.")

    frame_count = 0

    # Store every observed class for each tracking ID.
    track_classes = defaultdict(list)

    diagnostic_path = Path(output_path).with_name("tracking_diagnostics.csv")

    diagnostic_file = open(
        diagnostic_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    diagnostic_writer = csv.writer(diagnostic_file)

    diagnostic_writer.writerow([
        "frame", "track_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2"
    ])

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                source=frame,
                conf=confidence,
                imgsz=640,
                device="cpu",
                agnostic_nms=True,
                tracker=str(PROJECT_ROOT / "tracking" / "bytetrack_custom.yaml"),
                persist=True,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()

                for track_id, class_id, score, coordinates in zip(
                    track_ids, class_ids, confidences, boxes
                ):
                    class_name = model.names[class_id]

                    if class_name in ["car", "motorcycle", "bus", "truck"]:
                        track_classes[track_id].append(class_name)

                    diagnostic_writer.writerow([
                        frame_count,
                        track_id,
                        class_name,
                        round(score, 4),
                        *[round(value, 2) for value in coordinates]
                    ])

            annotated_frame = result.plot()
            writer.write(annotated_frame)

            frame_count += 1

    finally:
        cap.release()
        writer.release()
        diagnostic_file.close()

    # Assign one final class to each tracking ID.
    detection_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    for track_id, classes in track_classes.items():
        if classes:
            final_class = Counter(classes).most_common(1)[0][0]
            detection_counts[final_class] += 1

    return {
        "processed_frames": frame_count,
        "detection_counts": detection_counts,
        "output_path": output_path
    }


