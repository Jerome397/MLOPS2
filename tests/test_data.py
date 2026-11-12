import pytest
from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"
    assert category_from_filename("3_120.jpg") == "Egg"


@pytest.mark.parametrize("name", ["-1_2.jpg", "11_2.jpg", "wrong.jpg"])
def test_invalid_category(name):
    with pytest.raises(ValueError):
        category_from_filename(name)


def test_resize_image_shape_and_mode():
    image = Image.new("L", (512, 384))
    output = resize_image(image)
    assert output.size == (128, 128)
    assert output.mode == "RGB"
    assert image.size == (512, 384)


def test_eleven_classes():
    assert len(set(CLASSES)) == 11
    assert CLASSES == sorted(CLASSES)  # ImageFolder output ordering
