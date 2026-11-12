"""Prepare deterministic ImageFolder datasets without modifying raw images."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]
SPLITS = ["training", "validation", "evaluation"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def category_from_filename(filename: str) -> str:
    prefix = Path(filename).name.split("_", 1)[0]
    if not prefix.isdigit() or not 0 <= int(prefix) < len(CLASSES):
        raise ValueError(f"Invalid Food-11 filename: {filename}")
    return CLASSES[int(prefix)]


def resize_image(img: Image.Image, size: int = 128) -> Image.Image:
    if size < 1:
        raise ValueError("size must be positive")
    return (
        ImageOps.exif_transpose(img).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    )


def prepare(root: Path, mini_limit: int = 100):
    if not 1 <= mini_limit <= 100:
        raise ValueError("mini-limit must be between 1 and 100")
    raw = root / "food11_raw"
    for split in SPLITS:
        if not (raw / split).is_dir():
            raise FileNotFoundError(raw / split)
    # Refuse stale outputs rather than silently mixing old and new data.
    outputs = [root / "food11_processed", root / "food11_processed_mini"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Processed outputs already exist; archive them before rebuilding")
    for out in outputs:
        for split in SPLITS:
            for category in CLASSES:
                (out / split / category).mkdir(parents=True)
    seen = {}
    report = {"counts": {}, "excluded_cross_split_duplicates": []}
    # Keep evaluation first, then validation: remove leakage from training.
    for split in reversed(SPLITS):
        counts = dict.fromkeys(CLASSES, 0)
        for source in sorted((raw / split).iterdir()):
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            category = category_from_filename(source.name)
            with Image.open(source) as image:
                image = resize_image(image)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                report["excluded_cross_split_duplicates"].append(str(source))
                continue
            seen[digest] = split
            target = outputs[0] / split / category / (source.stem + ".jpg")
            image.save(target, quality=95)
            if counts[category] < mini_limit:
                (outputs[1] / split / category / target.name).write_bytes(target.read_bytes())
            counts[category] += 1
        report["counts"][split] = counts
        print(split, counts, flush=True)
    Path("lab/data_preparation.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--mini-limit", type=int, default=100)
    args = parser.parse_args()
    prepare(args.data_root, args.mini_limit)


if __name__ == "__main__":
    main()
