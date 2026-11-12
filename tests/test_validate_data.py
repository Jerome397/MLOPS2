from PIL import Image

from src.food11.data import CLASSES, SPLITS
from src.food11.validate_data import (
    check_distribution,
    check_duplicates,
    check_folders,
    check_images,
    check_minimum,
)


def test_contract_detects_leakage_and_corruption(tmp_path):
    for split_id, split in enumerate(SPLITS):
        for class_id, category in enumerate(CLASSES):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (split_id * 80, class_id * 20, 30)).save(folder / "x.png")
    assert check_folders(tmp_path)
    assert check_images(tmp_path)
    assert check_minimum(tmp_path, 1)
    assert check_duplicates(tmp_path)
    assert check_distribution(tmp_path, 0.05)
    source = tmp_path / "training" / "Bread" / "x.png"
    target = tmp_path / "evaluation" / "Bread" / "leak.png"
    target.write_bytes(source.read_bytes())
    assert not check_duplicates(tmp_path)
    target.write_bytes(b"corrupt")
    assert not check_images(tmp_path)
