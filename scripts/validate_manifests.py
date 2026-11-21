import csv
from pathlib import Path

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data")

TRAIN_FILE = DATA_DIR / "train_manifest.csv"
VAL_FILE = DATA_DIR / "val_manifest.csv"


def load_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


train = load_manifest(TRAIN_FILE)
val = load_manifest(VAL_FILE)

train_ids = {row["image_id"] for row in train}
val_ids = {row["image_id"] for row in val}

train_names = {row["file_name"] for row in train}
val_names = {row["file_name"] for row in val}

print("=" * 55)
print("MANIFEST VALIDATION")
print("=" * 55)

print(f"Training images: {len(train):,}")
print(f"Validation images: {len(val):,}")

print("\nDuplicate checks:")
print(f"Duplicate training IDs: {len(train) - len(train_ids)}")
print(f"Duplicate validation IDs: {len(val) - len(val_ids)}")
print(f"Overlapping image IDs: {len(train_ids & val_ids)}")
print(f"Overlapping filenames: {len(train_names & val_names)}")

print("\nPositive/negative checks:")
for name, rows in [("Training", train), ("Validation", val)]:
    positives = sum(row["has_target"] == "True" for row in rows)
    negatives = sum(row["has_target"] == "False" for row in rows)

    print(f"{name}: {positives:,} positive, {negatives:,} negative")

print("\nURL checks:")
for name, rows in [("Training", train), ("Validation", val)]:
    missing = sum(not row["coco_url"] for row in rows)
    print(f"{name}: {missing} missing image URLs")

print("\nExpected class IDs: 3, 4, 6, 8")
print("Validation complete.")