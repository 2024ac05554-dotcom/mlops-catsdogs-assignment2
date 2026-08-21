"""
Post-deployment model performance tracking (M5).

Sends a small batch of requests (from the held-out test manifest, treated
here as 'real or simulated production traffic') to the deployed inference
service, compares predictions against the true labels, and logs accuracy +
per-request latency. Results are written to monitoring/performance_log.csv
and can be re-run periodically (e.g. via a cron job / scheduled Action) to
watch for drift.

Usage:
    python scripts/track_performance.py --base_url http://localhost:8000 --n 20
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import requests

LOG_PATH = "monitoring/performance_log.csv"


def load_labeled_samples(manifest_csv: str, n: int):
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[:n]


def main(args):
    samples = load_labeled_samples(args.manifest, args.n)
    if not samples:
        print(f"No samples found in {args.manifest}")
        return

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)

    correct, total, latencies = 0, 0, []
    label_map = {"0": "cat", "1": "dog"}

    with open(LOG_PATH, "a", newline="") as logf:
        writer = csv.writer(logf)
        if write_header:
            writer.writerow(["timestamp", "filepath", "true_label", "predicted_label",
                              "correct", "latency_ms"])

        for row in samples:
            filepath, true_label = row["filepath"], label_map[row["label"]]
            with open(filepath, "rb") as img_file:
                start = time.time()
                resp = requests.post(
                    f"{args.base_url}/predict",
                    files={"file": (os.path.basename(filepath), img_file, "image/jpeg")},
                    timeout=10,
                )
                latency_ms = (time.time() - start) * 1000

            if resp.status_code != 200:
                print(f"WARN: request for {filepath} failed with {resp.status_code}")
                continue

            pred_label = resp.json()["label"]
            is_correct = pred_label == true_label
            correct += int(is_correct)
            total += 1
            latencies.append(latency_ms)

            writer.writerow([
                datetime.now(timezone.utc).isoformat(), filepath, true_label,
                pred_label, is_correct, round(latency_ms, 2),
            ])

    if total:
        accuracy = correct / total
        avg_latency = sum(latencies) / len(latencies)
        print(f"Evaluated {total} live requests")
        print(f"Post-deployment accuracy: {accuracy:.2%}")
        print(f"Average latency: {avg_latency:.2f} ms")
        print(f"Detailed log appended to: {LOG_PATH}")
    else:
        print("No successful requests were evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8000")
    parser.add_argument("--manifest", type=str, default="data/processed/test.csv")
    parser.add_argument("--n", type=int, default=20)
    main(parser.parse_args())
