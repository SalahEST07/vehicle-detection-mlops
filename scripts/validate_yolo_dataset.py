from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import random

DATA_DIR = Path(r"G:\vehicle-detection-mlops\data")

CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

random.seed(42)


def validate_split(split):
    image_dir = DATA_DIR / "images" / split
    label_dir = DATA_DIR / "labels" / split

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    invalid_lines = 0
    invalid_coordinates = 0
    invalid_classes = 0
    empty_labels = 0
    object_counts = Counter()

    for label_path in labels:
        content = label_path.read_text(encoding="utf-8").strip()

        if not content:
            empty_labels += 1
            continue

        for line in content.splitlines():
            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(
                    float, parts[1:]
                )
            except ValueError:
                invalid_lines += 1
                continue

            if class_id not in CLASS_NAMES:
                invalid_classes += 1
                continue

            values = [x_center, y_center, width, height]

            if not all(0 <= value <= 1 for value in values):
                invalid_coordinates += 1
                continue

            if width <= 0 or height <= 0:
                invalid_coordinates += 1
                continue

            object_counts[class_id] += 1

    print(f"\n{'=' * 55}")
    print(f"{split.upper()} VALIDATION")
    print("=" * 55)

    print(f"Images: {len(images):,}")
    print(f"Labels: {len(labels):,}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Empty labels (negative images): {empty_labels:,}")

    print("\nObject counts:")
    for class_id, name in CLASS_NAMES.items():
        print(f"{name:12s}: {object_counts[class_id]:,}")

    print("\nErrors:")
    print(f"Invalid label lines: {invalid_lines}")
    print(f"Invalid class IDs: {invalid_classes}")
    print(f"Invalid coordinates: {invalid_coordinates}")

    return {
        "images": images,
        "labels": labels,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "invalid_lines": invalid_lines,
        "invalid_classes": invalid_classes,
        "invalid_coordinates": invalid_coordinates,
        "object_counts": object_counts,
    }


def create_contact_sheet(split, images, max_samples=12):
    output_dir = DATA_DIR / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer images that contain vehicle annotations.
    label_dir = DATA_DIR / "labels" / split

    positive_images = [
        p for p in images
        if (label_dir / f"{p.stem}.txt").read_text(
            encoding="utf-8"
        ).strip()
    ]

    samples = random.sample(
        positive_images,
        min(max_samples, len(positive_images))
    )

    tile_width = 400
    tile_height = 280
    columns = 3
    rows = (len(samples) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "white"
    )

    draw_sheet = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(samples):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # Resize while preserving aspect ratio.
        image.thumbnail((tile_width, tile_height - 35))

        scale_x = image.width / original_width
        scale_y = image.height / original_height

        draw = ImageDraw.Draw(image)

        label_path = label_dir / f"{image_path.stem}.txt"

        for line in label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = (xc - bw / 2) * original_width * scale_x
            y1 = (yc - bh / 2) * original_height * scale_y
            x2 = (xc + bw / 2) * original_width * scale_x
            y2 = (yc + bh / 2) * original_height * scale_y

            draw.rectangle(
                (x1, y1, x2, y2),
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 15)),
                CLASS_NAMES[class_id],
                fill="yellow",
                stroke_width=1,
                stroke_fill="black"
            )

        x_offset = (index % columns) * tile_width
        y_offset = (index // columns) * tile_height

        sheet.paste(image, (x_offset, y_offset))

        draw_sheet.text(
            (x_offset + 5, y_offset + tile_height - 25),
            image_path.name,
            fill="black"
        )

    output_path = output_dir / f"{split}_contact_sheet.jpg"
    sheet.save(output_path, quality=90)

    print(f"\nContact sheet saved: {output_path}")


if __name__ == "__main__":
    train = validate_split("train")
    val = validate_split("val")

    create_contact_sheet("train", train["images"])
    create_contact_sheet("val", val["images"])

    print("\nValidation scripts completed.")