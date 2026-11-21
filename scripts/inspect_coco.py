import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data\annotations")

TARGET_CLASSES = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


def inspect_split(filename):
    path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"Inspecting: {filename}")
    print(f"{'=' * 60}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    images = data["images"]
    annotations = data["annotations"]

    # Count objects belonging to each target class.
    object_counts = Counter()

    # Track which target classes appear in each image.
    image_classes = defaultdict(set)

    for annotation in annotations:
        category_id = annotation["category_id"]

        if category_id in TARGET_CLASSES:
            class_name = TARGET_CLASSES[category_id]
            object_counts[class_name] += 1

            image_id = annotation["image_id"]
            image_classes[image_id].add(class_name)

    # Count images containing each class.
    image_counts = Counter()

    for classes in image_classes.values():
        for class_name in classes:
            image_counts[class_name] += 1

    print(f"Total images: {len(images):,}")
    print(f"Total annotations: {len(annotations):,}")

    print("\nTarget vehicle object counts:")
    for class_name in TARGET_CLASSES.values():
        print(f"  {class_name:12s}: {object_counts[class_name]:,}")

    print("\nImages containing each vehicle class:")
    for class_name in TARGET_CLASSES.values():
        print(f"  {class_name:12s}: {image_counts[class_name]:,}")

    print(
        "\nImages containing at least one target vehicle: "
        f"{len(image_classes):,}"
    )

    # Images containing more than one distinct target class.
    multi_class_images = sum(
        1 for classes in image_classes.values()
        if len(classes) > 1
    )

    print(
        "Images containing multiple target vehicle classes: "
        f"{multi_class_images:,}"
    )


if __name__ == "__main__":
    inspect_split("instances_train2017.json")
    inspect_split("instances_val2017.json")