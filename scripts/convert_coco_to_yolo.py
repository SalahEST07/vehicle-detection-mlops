import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data")
ANNOTATIONS_DIR = DATA_DIR / "annotations"
LABELS_DIR = DATA_DIR / "labels"

# Original COCO category ID -> new YOLO class ID
CLASS_MAPPING = {
    3: 0,  # car
    4: 1,  # motorcycle
    6: 2,  # bus
    8: 3,  # truck
}

CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}


def load_manifest(split):
    path = DATA_DIR / f"{split}_manifest.csv"

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def convert_split(split):
    print(f"\nConverting {split.upper()} annotations")
    print("=" * 55)

    annotation_file = (
        ANNOTATIONS_DIR / f"instances_{split}2017.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    manifest = load_manifest(split)

    # Keep only images selected in our manifest.
    selected_ids = {
        int(row["image_id"]) for row in manifest
    }

    image_info = {
        image["id"]: image for image in coco["images"]
        if image["id"] in selected_ids
    }

    # Group YOLO labels by image.
    image_labels = defaultdict(list)

    class_counts = Counter()
    skipped_crowd = 0
    invalid_boxes = 0
    ignored_categories = 0

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        if image_id not in selected_ids:
            continue

        if category_id not in CLASS_MAPPING:
            ignored_categories += 1
            continue

        if ann.get("iscrowd", 0) == 1:
            skipped_crowd += 1
            continue

        image = image_info[image_id]
        img_width = image["width"]
        img_height = image["height"]

        x, y, box_width, box_height = ann["bbox"]

        # Clip the bounding box to the image boundaries.
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(img_width), x + box_width)
        y2 = min(float(img_height), y + box_height)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            invalid_boxes += 1
            continue

        # Convert to normalized YOLO coordinates.
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        norm_width = box_width / img_width
        norm_height = box_height / img_height

        yolo_class = CLASS_MAPPING[category_id]

        line = (
            f"{yolo_class} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{norm_width:.6f} "
            f"{norm_height:.6f}"
        )

        image_labels[image_id].append(line)
        class_counts[yolo_class] += 1

    output_dir = LABELS_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a label file for every selected image,
    # including negative images with no target objects.
    missing_image_ids = []

    for row in manifest:
        image_id = int(row["image_id"])
        filename = Path(row["file_name"]).stem
        label_path = output_dir / f"{filename}.txt"

        if image_id not in image_info:
            missing_image_ids.append(image_id)
            continue

        labels = image_labels.get(image_id, [])

        with open(label_path, "w", encoding="utf-8") as f:
            if labels:
                f.write("\n".join(labels) + "\n")

    print(f"Manifest images: {len(manifest):,}")
    print(f"Images with target labels: {len(image_labels):,}")
    print(f"Label files created: {len(list(output_dir.glob('*.txt'))):,}")

    print("\nYOLO object counts:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {name:12s}: {class_counts[class_id]:,}")

    print(f"\nIgnored other categories: {ignored_categories:,}")
    print(f"Skipped crowd annotations: {skipped_crowd:,}")
    print(f"Invalid boxes: {invalid_boxes:,}")
    print(f"Missing image metadata: {len(missing_image_ids):,}")

    return {
        "manifest_count": len(manifest),
        "label_count": len(list(output_dir.glob("*.txt"))),
        "class_counts": class_counts,
        "missing": missing_image_ids,
    }


if __name__ == "__main__":
    train_result = convert_split("train")
    val_result = convert_split("val")

    print("\n" + "=" * 55)
    print("CONVERSION FINISHED")
    print("=" * 55)

    print(
        "Training label files:",
        train_result["label_count"]
    )
    print(
        "Validation label files:",
        val_result["label_count"]
    )

    print(
        "Total target objects:",
        sum(train_result["class_counts"].values())
        + sum(val_result["class_counts"].values())
    )