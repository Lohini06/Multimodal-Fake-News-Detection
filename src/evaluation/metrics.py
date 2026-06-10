import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

def evaluate(model, test_loader, device: torch.device, config: dict):
    """Run model on test set and return all predictions and labels."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            logits = model(images, texts)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all classification metrics."""
    metrics = {
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "f1_macro":  round(f1_score(labels, preds, average="macro"), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
        "precision": round(precision_score(labels, preds, average="macro"), 4),
        "recall":    round(recall_score(labels, preds, average="macro"), 4),
    }
    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    print("\n📋 Classification Report:")
    print(classification_report(labels, preds, target_names=["Real", "Fake"]))
    return metrics


def save_metrics(metrics: dict, config: dict):
    """Save metrics to JSON."""
    path = os.path.join(config["outputs"]["results_dir"], "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {path}")


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, config: dict):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(config["outputs"]["results_dir"], "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")


def plot_training_history(history: dict, config: dict):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss", color="#7F77DD")
    ax1.plot(history["val_loss"],   label="Val Loss",   color="#1D9E75")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc", color="#7F77DD")
    ax2.plot(history["val_acc"],   label="Val Acc",   color="#1D9E75")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(config["outputs"]["results_dir"], "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Training curves saved to {path}")