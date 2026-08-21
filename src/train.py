"""
Train the baseline CNN on Cats vs Dogs and track the experiment with MLflow.

Logs:
  - hyperparameters (params)
  - per-epoch train/val loss & accuracy (metrics)
  - a loss-curve plot (artifact)
  - a confusion matrix plot on the test set (artifact)
  - the trained model, serialized as a .pt file (artifact + registered model)

Usage:
    python -m src.train --epochs 5 --batch_size 16 --lr 1e-3
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.data_preprocessing import CatsDogsDataset, preprocess_and_split
from src.model import build_model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_confusion_matrix(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preprocess + split (writes manifests tracked by DVC)
    manifests = preprocess_and_split(raw_dir=args.raw_dir, processed_dir=args.processed_dir)

    train_ds = CatsDogsDataset(manifests["train"], train=True)
    val_ds = CatsDogsDataset(manifests["val"], train=False)
    test_ds = CatsDogsDataset(manifests["test"], train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("cats-vs-dogs-classification")

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "Adam",
            "model": "SimpleCNN",
            "image_size": 224,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        })

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            }, step=epoch)
            print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Test set evaluation + confusion matrix artifact
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        y_true, y_pred = evaluate_confusion_matrix(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=["cat", "dog"]).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title("Confusion Matrix (Test Set)")
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        fig_cm.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path)
        plt.close(fig_cm)

        # Loss curve artifact
        fig_loss, ax_loss = plt.subplots(figsize=(5, 4))
        ax_loss.plot(history["train_loss"], label="train_loss")
        ax_loss.plot(history["val_loss"], label="val_loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("Loss Curve")
        ax_loss.legend()
        loss_path = os.path.join(MODEL_DIR, "loss_curve.png")
        fig_loss.savefig(loss_path, bbox_inches="tight")
        mlflow.log_artifact(loss_path)
        plt.close(fig_loss)

        # Save trained model in standard serialized format (.pt)
        model_path = os.path.join(MODEL_DIR, "model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(
            model,
            name="pytorch-model",
            input_example=next(iter(test_loader))[0][:1].numpy(),
        )

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Model saved to: {model_path}")

        # DVC-trackable metrics file (separate from MLflow, for `dvc metrics diff`)
        import json
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")
        with open(metrics_path, "w") as mf:
            json.dump({
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "final_train_acc": history["train_acc"][-1],
                "final_val_acc": history["val_acc"][-1],
            }, mf, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--run_name", type=str, default="baseline-cnn")
    main(parser.parse_args())
