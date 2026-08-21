"""Unit tests for src/data_preprocessing.py"""
import csv
import os

import pytest
from PIL import Image

from src.data_preprocessing import (
    CLASSES,
    list_image_paths,
    split_dataset,
    write_manifest,
    get_transforms,
)


@pytest.fixture
def tiny_raw_dir(tmp_path):
    """Create a tiny fake raw dataset: 4 cat images, 4 dog images."""
    for cls in CLASSES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(4):
            img = Image.new("RGB", (50, 50), color=(i * 10, i * 10, i * 10))
            img.save(cls_dir / f"{cls}_{i}.jpg")
    return str(tmp_path)


def test_list_image_paths_finds_all_images(tiny_raw_dir):
    samples = list_image_paths(tiny_raw_dir)
    assert len(samples) == 8
    labels = {label for _, label in samples}
    assert labels == {0, 1}  # cat=0, dog=1


def test_list_image_paths_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_image_paths(str(tmp_path))


def test_split_dataset_respects_fractions(tiny_raw_dir):
    samples = list_image_paths(tiny_raw_dir)
    train, val, test = split_dataset(samples, train_frac=0.5, val_frac=0.25, seed=1)
    # 4 samples per class * 0.5 = 2 train per class -> 4 total; etc.
    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 2
    # No overlap between splits
    all_paths = [p for p, _ in train] + [p for p, _ in val] + [p for p, _ in test]
    assert len(all_paths) == len(set(all_paths))


def test_split_dataset_is_stratified(tiny_raw_dir):
    samples = list_image_paths(tiny_raw_dir)
    train, _, _ = split_dataset(samples, train_frac=0.5, val_frac=0.25, seed=1)
    labels_in_train = [label for _, label in train]
    # Both classes should be represented in the training split
    assert set(labels_in_train) == {0, 1}


def test_write_manifest_creates_valid_csv(tmp_path, tiny_raw_dir):
    samples = list_image_paths(tiny_raw_dir)
    out_path = write_manifest("train", samples, str(tmp_path))
    assert os.path.exists(out_path)
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(samples)
    assert set(rows[0].keys()) == {"filepath", "label"}


def test_get_transforms_produces_correct_tensor_shape(tiny_raw_dir):
    transform = get_transforms(train=False)
    img = Image.open(os.path.join(tiny_raw_dir, "cats", "cats_0.jpg")).convert("RGB")
    tensor = transform(img)
    # Resized to 224x224, 3 channels
    assert tensor.shape == (3, 224, 224)
