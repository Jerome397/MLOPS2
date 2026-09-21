from pathlib import Path
from PIL import Image
import shutil


RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

CLASS_NAMES = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles-Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable-Fruit",
}

SPLITS = ["training", "evaluation", "validation"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare_directories():
    for directory in [PROCESSED_DIR, MINI_DIR]:
        if directory.exists():
            shutil.rmtree(directory)

    for split in SPLITS:
        for class_name in CLASS_NAMES.values():

            (PROCESSED_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

            (MINI_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


def process_split(split):
    source_dir = RAW_DIR / split

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Could not find {source_dir}"
        )

    mini_counts = {
        class_name: 0
        for class_name in CLASS_NAMES.values()
    }

    image_files = sorted(
        file
        for file in source_dir.iterdir()
        if file.suffix.lower() in VALID_EXTENSIONS
    )

    for image_path in image_files:

        category_id = image_path.stem.split("_")[0]

        if category_id not in CLASS_NAMES:
            print(f"Skipping unknown file: {image_path.name}")
            continue

        class_name = CLASS_NAMES[category_id]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            processed_path = (
                PROCESSED_DIR
                / split
                / class_name
                / image_path.name
            )

            image.save(processed_path)

            if mini_counts[class_name] < MINI_LIMIT:

                mini_path = (
                    MINI_DIR
                    / split
                    / class_name
                    / image_path.name
                )

                image.save(mini_path)
                mini_counts[class_name] += 1

    print(f"Finished {split}")


def main():
    print("Preparing Food-11 dataset...")

    prepare_directories()

    for split in SPLITS:
        process_split(split)

    print("Dataset preparation completed.")


if __name__ == "__main__":
    main()