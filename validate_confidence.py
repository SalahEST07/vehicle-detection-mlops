
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    MODEL_PATH = r"G:\vehicle-detection-mlops\training\results\yolo26n_baseline\weights\best.pt"
    DATA_PATH = r"G:\vehicle-detection-mlops\data\vehicle.yaml"

    model = YOLO(MODEL_PATH)

    thresholds = [0.001, 0.01, 0.05, 0.10, 0.25]

    for conf in thresholds:
        print(f"\n{'='*50}")
        print(f"VALIDATION CONFIDENCE: {conf}")
        print(f"{'='*50}")

        metrics = model.val(
            data=DATA_PATH,
            imgsz=640,
            batch=16,
            workers=0,
            device=0,
            conf=conf,
            iou=0.7,
            plots=True,
            project=r"G:\vehicle-detection-mlops\training\confidence_analysis",
            name=f"conf_{conf}",
            exist_ok=True
        )

        print("\nRESULTS:")
        print("Confidence:", conf)
        print("Precision:", metrics.box.mp)
        print("Recall:", metrics.box.mr)
        print("mAP50:", metrics.box.map50)
        print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    freeze_support()
    main()
