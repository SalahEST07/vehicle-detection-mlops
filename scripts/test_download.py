import csv
import urllib.request
from pathlib import Path
from PIL import Image

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data")

BATCH_SIZE = 10


def load_manifest(split):
    path = DATA_DIR / f"{split}_manifest.csv"

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def download_test_batch(split):
    rows = load_manifest(split)[:BATCH_SIZE]

    output_dir = DATA_DIR / "images" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    print(f"\nTesting {split.upper()} downloads")
    print("-" * 45)

    for row in rows:
        filename = row["file_name"]
        url = row["coco_url"]
        output_path = output_dir / filename

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()

            output_path.write_bytes(content)

            # Verify that the file is a readable image.
            with Image.open(output_path) as img:
                img.verify()

            # Reopen to read dimensions.
            with Image.open(output_path) as img:
                width, height = img.size

            if width != int(row["width"]) or height != int(row["height"]):
                raise ValueError("Image dimensions do not match COCO metadata")

            size_mb = output_path.stat().st_size / (1024 * 1024)

            print(
                f"OK: {filename} | "
                f"{width}x{height} | {size_mb:.2f} MB"
            )

            successful += 1

        except Exception as error:
            failed += 1

            if output_path.exists():
                output_path.unlink()

            print(f"FAILED: {filename} | {error}")

    print(f"\n{split.upper()} summary")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    return successful, failed


if __name__ == "__main__":
    train_ok, train_failed = download_test_batch("train")
    val_ok, val_failed = download_test_batch("val")

    print("\n" + "=" * 45)
    print("TOTAL TEST RESULTS")
    print("=" * 45)
    print(f"Successful downloads: {train_ok + val_ok}")
    print(f"Failed downloads: {train_failed + val_failed}")