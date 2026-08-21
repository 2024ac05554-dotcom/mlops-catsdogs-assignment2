"""
Data preprocessing utilities for the Cats vs Dogs classifier.

Responsibilities:
  - Load raw images from data/raw/{cats,dogs}
  - Resize to 224x224 RGB (standard CNN input size)
  - Split into train/val/test (default 80/10/10)
  - Provide torchvision-style augmentation transforms for training
  - Persist the processed split as a manifest CSV under data/processed
    (this is what gets tracked by DVC, see dvc.yaml / data/processed.dvc)
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
CLASSES = ["cats", "dogs"]  # label index 0 = cat, 1 = dog


def list_image_paths(raw_dir: str) -> List[Tuple[str, int]]:
    """Return list of (filepath, label) tuples from data/raw/{cats,dogs}."""
    samples = []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), label_idx))
    if not samples:
        raise FileNotFoundError(
            f"No images found under {raw_dir}/cats or {raw_dir}/dogs. "
            "Run scripts/generate_dummy_data.py or add the real dataset."
        )
    return samples


def split_dataset(
    samples: List[Tuple[str, int]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """Stratified 80/10/10 (default) split by class."""
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train, val, test = [], [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_manifest(split_name: str, items: List[Tuple[str, int]], processed_dir: str) -> str:
    """Write a CSV manifest (filepath,label) for a split. This CSV is the
    lightweight artifact tracked by DVC instead of duplicating images."""
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, f"{split_name}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(items)
    return out_path


def get_transforms(train: bool) -> transforms.Compose:
    """Standard ImageNet-style normalization + augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@dataclass
class CatsDogsDataset(Dataset):
    manifest_csv: str
    train: bool = False

    def __post_init__(self):
        with open(self.manifest_csv) as f:
            reader = csv.DictReader(f)
            self.samples = [(row["filepath"], int(row["label"])) for row in reader]
        self.transform = get_transforms(self.train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def preprocess_and_split(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """End-to-end: load raw images, split, write manifests. Returns dict of paths."""
    samples = list_image_paths(raw_dir)
    train, val, test = split_dataset(samples)
    paths = {
        "train": write_manifest("train", train, processed_dir),
        "val": write_manifest("val", val, processed_dir),
        "test": write_manifest("test", test, processed_dir),
    }
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return paths


if __name__ == "__main__":
    preprocess_and_split()
