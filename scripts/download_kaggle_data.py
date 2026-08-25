"""
Downloads the Kaggle Dog and Cat Classification dataset via kagglehub, then
reorganizes whatever folder layout it comes in into the structure the rest
of this pipeline expects:

    data/raw/cats/*.jpg
    data/raw/dogs/*.jpg

An optional --limit caps images per class so the demo stays fast.

Usage:
    python scripts/download_kaggle_data.py --limit 500
"""
import argparse
import os
import shutil

import kagglehub

DEST_ROOT = "data/raw"


def find_image_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        lower_dir = dirpath.lower()
        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lower_name = fname.lower()
            if "cat" in lower_dir or lower_name.startswith("cat"):
                label = "cats"
            elif "dog" in lower_dir or lower_name.startswith("dog"):
                label = "dogs"
            else:
                continue
            yield os.path.join(dirpath, fname), label


def main(limit: int):
    print("Downloading dataset from Kaggle (this may take a few minutes)...")
    dataset_path = kagglehub.dataset_download(
        "bhavikjikadara/dog-and-cat-classification-dataset"
    )
    print(f"Downloaded to: {dataset_path}")

    os.makedirs(os.path.join(DEST_ROOT, "cats"), exist_ok=True)
    os.makedirs(os.path.join(DEST_ROOT, "dogs"), exist_ok=True)

    counts = {"cats": 0, "dogs": 0}
    for src_path, label in find_image_files(dataset_path):
        if limit and counts[label] >= limit:
            continue
        dest_name = f"{label[:-1]}_{counts[label]:05d}.jpg"
        dest_path = os.path.join(DEST_ROOT, label, dest_name)
        try:
            shutil.copyfile(src_path, dest_path)
            counts[label] += 1
        except Exception as e:
            print(f"WARN: skipped {src_path}: {e}")

        if limit and counts["cats"] >= limit and counts["dogs"] >= limit:
            break

    print(f"Copied {counts['cats']} cat images and {counts['dogs']} dog images "
          f"into {DEST_ROOT}/cats and {DEST_ROOT}/dogs")

    if counts["cats"] == 0 or counts["dogs"] == 0:
        print("\nWARNING: one or both classes have 0 images. Inspect layout with:")
        print(f'  Get-ChildItem -Recurse -Directory "{dataset_path}" | Select-Object FullName')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500,
                         help="Max images per class to copy (0 = copy all).")
    args = parser.parse_args()
    main(args.limit)
