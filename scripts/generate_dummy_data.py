"""
Generates a small synthetic dataset of 'cat' and 'dog' images so the full
pipeline (preprocessing -> training -> tracking -> packaging -> CI/CD) can be
exercised end-to-end without downloading the real Kaggle dataset.

For the real assignment submission, DELETE the generated files and replace
data/raw/cats and data/raw/dogs with the actual images from:
https://www.kaggle.com/datasets/salader/dogs-vs-cats  (or equivalent)

Usage:
    python scripts/generate_dummy_data.py --per_class 60
"""
import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw


def make_image(label: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    size = 256
    img = Image.new("RGB", (size, size), color=(
        rng.randint(150, 255), rng.randint(150, 255), rng.randint(150, 255)
    ))
    draw = ImageDraw.Draw(img)

    # Cats: rounder features / triangle "ears". Dogs: longer snout / floppy ears.
    body_color = (rng.randint(0, 120), rng.randint(0, 120), rng.randint(0, 120))
    cx, cy = size // 2, size // 2

    if label == "cats":
        r = rng.randint(60, 90)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=body_color)
        # pointy ears (triangles)
        draw.polygon([(cx - r, cy - r), (cx - r + 30, cy - r - 40), (cx - r + 60, cy - r)], fill=body_color)
        draw.polygon([(cx + r - 60, cy - r), (cx + r - 30, cy - r - 40), (cx + r, cy - r)], fill=body_color)
    else:  # dogs
        rw, rh = rng.randint(70, 100), rng.randint(50, 80)
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=body_color)
        # floppy ears (ellipses on the sides)
        draw.ellipse([cx - rw - 30, cy - 20, cx - rw + 10, cy + 60], fill=body_color)
        draw.ellipse([cx + rw - 10, cy - 20, cx + rw + 30, cy + 60], fill=body_color)

    # add noise so images aren't trivially identical
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(seed).integers(-15, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(per_class: int, out_dir: str):
    for label in ["cats", "dogs"]:
        cls_dir = os.path.join(out_dir, label)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(per_class):
            img = make_image(label, seed=i if label == "cats" else 10_000 + i)
            img.save(os.path.join(cls_dir, f"{label[:-1]}_{i:04d}.jpg"), quality=90)
    print(f"Generated {per_class} images per class under {out_dir}/{{cats,dogs}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=60)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()
    main(args.per_class, args.out_dir)
