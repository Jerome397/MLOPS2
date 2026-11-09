import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image


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


SPLITS = [
    "training",
    "validation",
    "evaluation",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def print_result(
    name: str,
    passed: bool,
    details: str = "",
):
    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    message = (
        f"{status} - {name}"
    )

    if details:
        message += (
            f": {details}"
        )

    print(message)


def image_files(
    folder: Path,
):
    if not folder.exists():
        return []

    return [
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]


def check_folders(
    root: Path,
):
    passed = True

    expected = set(
        CLASSES
    )

    for split in SPLITS:

        split_dir = (
            root
            / split
        )

        if not split_dir.exists():

            passed = False

            print_result(
                f"{split} folders",
                False,
                "split does not exist",
            )

            continue

        actual = {
            path.name
            for path
            in split_dir.iterdir()
            if path.is_dir()
        }

        split_ok = (
            actual == expected
        )

        passed = (
            passed
            and split_ok
        )

        print_result(
            f"{split} folders",
            split_ok,
            (
                f"{len(actual)} "
                f"class folders"
            ),
        )

    return passed


def check_images(
    root: Path,
):
    passed = True

    checked = 0

    for split in SPLITS:
        for class_name in CLASSES:

            folder = (
                root
                / split
                / class_name
            )

            for path in image_files(
                folder
            ):

                try:

                    with Image.open(
                        path
                    ) as img:

                        img.load()

                        valid = (
                            img.mode == "RGB"
                            and img.size
                            == (128, 128)
                        )

                except Exception:

                    valid = False

                if not valid:

                    passed = False

                    print(
                        "Invalid image:",
                        path,
                    )

                checked += 1

    print_result(
        "image format",
        passed,
        (
            f"{checked} images "
            f"checked"
        ),
    )

    return passed


def check_minimum(
    root: Path,
    minimum: int,
):
    passed = True

    for split in SPLITS:
        for class_name in CLASSES:

            count = len(
                image_files(
                    root
                    / split
                    / class_name
                )
            )

            if count < minimum:

                passed = False

                print(
                    "Too few images:",
                    split,
                    class_name,
                    count,
                )

    print_result(
        "minimum images per class",
        passed,
        (
            f"minimum={minimum}"
        ),
    )

    return passed


def check_duplicates(
    root: Path,
):
    passed = True

    seen = {}

    for split in SPLITS:
        for class_name in CLASSES:

            folder = (
                root
                / split
                / class_name
            )

            for path in image_files(
                folder
            ):

                digest = (
                    hashlib
                    .md5(
                        path.read_bytes()
                    )
                    .hexdigest()
                )

                if digest in seen:

                    old_split, old_path = (
                        seen[digest]
                    )

                    if (
                        old_split
                        != split
                    ):

                        passed = False

                        print(
                            "Duplicate across splits:",
                            old_path,
                            "<->",
                            path,
                        )

                else:

                    seen[digest] = (
                        split,
                        path,
                    )

    print_result(
        "no duplicates across splits",
        passed,
    )

    return passed


def class_shares(
    root: Path,
    split: str,
):

    counts = {}

    total = 0

    for class_name in CLASSES:

        count = len(
            image_files(
                root
                / split
                / class_name
            )
        )

        counts[
            class_name
        ] = count

        total += count

    if total == 0:
        return {
            class_name: 0.0
            for class_name in CLASSES
        }

    return {
        class_name:
        counts[class_name] / total
        for class_name in CLASSES
    }


def check_distribution(
    root: Path,
    tau: float,
):

    train_shares = class_shares(
        root,
        "training",
    )

    val_shares = class_shares(
        root,
        "validation",
    )

    passed = True

    for class_name in CLASSES:

        difference = abs(
            train_shares[class_name]
            - val_shares[class_name]
        )

        if difference > tau:

            passed = False

            print(
                "Distribution mismatch:",
                class_name,
                f"{difference:.4f}",
            )

    print_result(
        "training/validation distribution",
        passed,
        f"tau={tau}",
    )

    return passed


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--min-per-class",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    root = args.dataset

    results = [
        check_folders(root),

        check_images(root),

        check_minimum(
            root,
            args.min_per_class,
        ),

        check_duplicates(root),

        check_distribution(
            root,
            args.tau,
        ),
    ]

    print()

    if all(results):

        print(
            "Dataset contract: PASS"
        )

        sys.exit(0)

    print(
        "Dataset contract: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()