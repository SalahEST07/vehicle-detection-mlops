import json
import csv
import random
from pathlib import Path

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data\annotations")
OUTPUT_DIR = Path(r"G:\vehicle-detection-mlops\data")

TARGET_IDS = {3, 4, 6, 8}

NEGATIVE_RATIO = 0.10
RANDOM_SEED = 42


def create_manifest(split):
    annotation_file = DATA_DIR / f"instances_{split}2017.json"

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Group target annotations by image ID.
    image_targets = {}

    for ann in annotations:
        category_id = ann["category_id"]

        if category_id in TARGET_IDS:
            image_id = ann["image_id"]

            if image_id not in image_targets:
                image_targets[image_id] = []

            image_targets[image_id].append(category_id)

    # Separate positive and negative images.
    positive_images = []
    negative_images = []

    for img in images:
        image_id = img["id"]

        if image_id in image_targets:
            positive_images.append(img)
        else:
            negative_images.append(img)

    # Select a reproducible sample of negative images.
    negative_count = round(len(positive_images) * NEGATIVE_RATIO)

    rng = random.Random(RANDOM_SEED)
    selected_negatives = rng.sample(
        negative_images,
        min(negative_count, len(negative_images))
    )

    selected_images = []

    for img in positive_images:
        selected_images.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "coco_url": img.get("coco_url", ""),
            "width": img["width"],
            "height": img["height"],
            "has_target": True,
            "category_ids": ",".join(
                map(str, sorted(set(image_targets[img["id"]])))
            ),
            "target_object_count": len(image_targets[img["id"]]),
        })

    for img in selected_negatives:
        selected_images.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "coco_url": img.get("coco_url", ""),
            "width": img["width"],
            "height": img["height"],
            "has_target": False,
            "category_ids": "",
            "target_object_count": 0,
        })

    # Save the manifest.
    output_file = OUTPUT_DIR / f"{split}_manifest.csv"

    fieldnames = [
        "image_id",
        "file_name",
        "coco_url",
        "width",
        "height",
        "has_target",
        "category_ids",
        "target_object_count",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_images)

    print(f"\n{split.upper()} SPLIT")
    print("-" * 40)
    print(f"Positive images: {len(positive_images):,}")
    print(f"Negative images selected: {len(selected_negatives):,}")
    print(f"Total selected images: {len(selected_images):,}")
    print(f"Manifest saved: {output_file}")


if __name__ == "__main__":
    create_manifest("train")
    create_manifest("val")