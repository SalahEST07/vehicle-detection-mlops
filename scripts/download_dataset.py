import csv
import time
import urllib.request
import threading

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data")

MAX_RETRIES = 3
MAX_WORKERS = 8
TIMEOUT = 60

progress_lock = threading.Lock()
completed = 0
downloaded = 0
skipped = 0
failed = 0


def load_manifest(split):
    path = DATA_DIR / f"{split}_manifest.csv"

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_image(path, row):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size

        return (
            width == int(row["width"])
            and height == int(row["height"])
        )
    except Exception:
        return False


def download_one(row, split):
    global downloaded, skipped, failed, completed

    output_dir = DATA_DIR / "images" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / row["file_name"]

    result = "failed"

    # Reuse an existing image if it passes verification.
    if output_path.exists() and verify_image(output_path, row):
        result = "skipped"
    else:
        if output_path.exists():
            output_path.unlink()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                request = urllib.request.Request(
                    row["coco_url"],
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                with urllib.request.urlopen(
                    request, timeout=TIMEOUT
                ) as response:
                    content = response.read()

                # Write to a temporary file first.
                temp_path = output_path.with_suffix(".tmp")
                temp_path.write_bytes(content)

                if not verify_image(temp_path, row):
                    temp_path.unlink(missing_ok=True)
                    raise ValueError("Invalid image or wrong dimensions")

                temp_path.replace(output_path)

                result = "downloaded"
                break

            except Exception as error:
                if attempt == MAX_RETRIES:
                    print(
                        f"\nFAILED: {row['file_name']} | {error}"
                    )
                else:
                    time.sleep(attempt * 2)

        if result == "failed":
            output_path.unlink(missing_ok=True)

    with progress_lock:
        completed += 1

        if result == "downloaded":
            downloaded += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

        if completed % 100 == 0 or completed == total_images:
            print(
                f"\rProgress: {completed:,}/{total_images:,} | "
                f"Downloaded: {downloaded:,} | "
                f"Skipped: {skipped:,} | "
                f"Failed: {failed:,}",
                end="",
                flush=True
            )

    return result


def download_split(split):
    global completed, downloaded, skipped, failed, total_images

    rows = load_manifest(split)

    completed = 0
    downloaded = 0
    skipped = 0
    failed = 0
    total_images = len(rows)

    print(f"\n\nDownloading {split.upper()} images")
    print("=" * 60)
    print(f"Images in manifest: {total_images:,}")
    print(f"Parallel workers: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_one, row, split)
            for row in rows
        ]

        for future in as_completed(futures):
            future.result()

    print("\n\nSplit summary:")
    print(f"Downloaded: {downloaded:,}")
    print(f"Already valid: {skipped:,}")
    print(f"Failed: {failed:,}")

    return rows, failed


def main():
    all_failures = []

    for split in ["train", "val"]:
        rows, split_failures = download_split(split)

        if split_failures:
            output_dir = DATA_DIR / "images" / split

            for row in rows:
                path = output_dir / row["file_name"]

                if not path.exists() or not verify_image(path, row):
                    all_failures.append({
                        "split": split,
                        "file_name": row["file_name"],
                        "coco_url": row["coco_url"],
                    })

    failure_file = DATA_DIR / "download_failures.csv"

    with open(
        failure_file, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "file_name", "coco_url"]
        )
        writer.writeheader()
        writer.writerows(all_failures)

    print("\n" + "=" * 60)
    print("DOWNLOAD PROCESS FINISHED")
    print("=" * 60)
    print(f"Remaining failed images: {len(all_failures):,}")
    print(f"Failure report: {failure_file}")


if __name__ == "__main__":
    main()