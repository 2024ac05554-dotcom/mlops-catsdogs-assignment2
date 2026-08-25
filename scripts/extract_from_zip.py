"""
Extracts cat/dog images directly from a downloaded Kaggle dataset zip file
straight into data/raw/cats and data/raw/dogs.

Pass --zip_path pointing at your downloaded archive.
"""
import argparse
import os
import zipfile

DEST_ROOT = "data/raw"


def main(zip_path: str, limit: int):
    os.makedirs(os.path.join(DEST_ROOT, "cats"), exist_ok=True)
    os.makedirs(os.path.join(DEST_ROOT, "dogs"), exist_ok=True)

    counts = {"cats": 0, "dogs": 0}

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"Zip contains {len(names)} entries. Scanning for images...")

        for name in names:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lower_name = name.lower()

            if "cat" in lower_name:
                label = "cats"
            elif "dog" in lower_name:
                label = "dogs"
            else:
                continue

            if limit and counts[label] >= limit:
                continue

            dest_name = f"{label[:-1]}_{counts[label]:05d}.jpg"
            dest_path = os.path.join(DEST_ROOT, label, dest_name)

            try:
                with zf.open(name) as src, open(dest_path, "wb") as dst:
                    data = src.read()
                    if len(data) == 0:
                        continue
                    dst.write(data)
                counts[label] += 1
            except Exception as e:
                print(f"WARN: skipped {name}: {e}")

            if limit and counts["cats"] >= limit and counts["dogs"] >= limit:
                break

    print(f"Extracted {counts['cats']} cat images and {counts['dogs']} dog images "
          f"into {DEST_ROOT}/cats and {DEST_ROOT}/dogs")

    if counts["cats"] == 0 or counts["dogs"] == 0:
        print("WARNING: 0 images for one class. Inspect the zip contents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    main(args.zip_path, args.limit)
