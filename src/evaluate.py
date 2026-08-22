"""
evaluate.py
Evaluation metrics and plots for multi-class pest species classification.

Accuracy is more meaningful here than it was for the ~1%-positive exoplanet
task, but with 10-15 classes of differing frequency (IP102's long tail),
macro-F1 (unweighted average across classes) is still reported alongside it
so a few dominant classes can't hide poor performance on rarer ones.
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Runs the model over a DataLoader, returns (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_labels, all_preds = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def print_report(y_true, y_pred, class_names, model_name: str = "Model"):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"--- {model_name} : classification report ---")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str = "Model", save_path: str | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6), max(5, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def compare_models(results: dict, save_path: str | None = None):
    """
    results: {"Baseline": (y_true, y_pred), "CNN": (y_true, y_pred), ...}
    Bar chart comparing accuracy AND macro-F1 across models.
    """
    names = list(results.keys())
    accs = [accuracy_score(y_true, y_pred) for (y_true, y_pred) in results.values()]
    f1s = [f1_score(y_true, y_pred, average="macro") for (y_true, y_pred) in results.values()]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#888888")
    ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#2b6cb0")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Baseline vs. CNN")
    ax.legend()
    for i, v in enumerate(accs):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(f1s):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
