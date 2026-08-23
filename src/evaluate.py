"""Full evaluation: baselines + MLP (with/without augmentation), permutation
importance, RPM-band failure analysis, and all report figures.
See PROJECT_PLAN.md §6-§7.

Run from src/: `python3 evaluate.py`
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from baselines import train_full_baseline, train_rpm_only_baseline
from preprocess import FEATURE_COLS, prepare_data
from train import train_model

SEED = 42
FIG_DIR = "../outputs/figures"
METRICS_PATH = "../outputs/metrics.json"
RPM_IDX = FEATURE_COLS.index("Engine rpm")


def mlp_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": {"class_0": float(precision[0]), "class_1": float(precision[1])},
        "recall": {"class_0": float(recall[0]), "class_1": float(recall[1])},
        "f1": {"class_0": float(f1[0]), "class_1": float(f1[1])},
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def permutation_importance_mlp(model, X_test, y_test, baseline_auc, n_repeats=10, seed=SEED):
    rng = np.random.default_rng(seed)
    importances = {}
    for i, col in enumerate(FEATURE_COLS):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, i])
            probs = mlp_probs(model, X_perm)
            auc = roc_auc_score(y_test, probs)
            drops.append(baseline_auc - auc)
        importances[col] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return importances


def rpm_band_accuracy(model, X_test, y_test, scaler, n_bands=4):
    rpm_scaled = X_test[:, RPM_IDX]
    rpm_raw = rpm_scaled * scaler.scale_[RPM_IDX] + scaler.mean_[RPM_IDX]
    quartile_edges = np.quantile(rpm_raw, np.linspace(0, 1, n_bands + 1))

    probs = mlp_probs(model, X_test)
    preds = (probs >= 0.5).astype(int)

    bands = []
    for i in range(n_bands):
        lo, hi = quartile_edges[i], quartile_edges[i + 1]
        if i == n_bands - 1:
            mask = (rpm_raw >= lo) & (rpm_raw <= hi)
        else:
            mask = (rpm_raw >= lo) & (rpm_raw < hi)
        band_acc = accuracy_score(y_test[mask], preds[mask]) if mask.sum() > 0 else None
        bands.append({
            "band": f"Q{i+1}",
            "rpm_range": [float(lo), float(hi)],
            "n": int(mask.sum()),
            "accuracy": float(band_acc) if band_acc is not None else None,
        })
    return bands


def plot_confusion_matrices(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_test, results, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(histories, path):
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist["train_loss"], label="Train loss")
        ax.plot(hist["val_loss"], label="Val loss")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(importances, path):
    cols = list(importances.keys())
    means = [importances[c]["mean"] for c in cols]
    stds = [importances[c]["std"] for c in cols]
    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        [cols[i] for i in order],
        [means[i] for i in order],
        xerr=[stds[i] for i in order],
        color="steelblue",
    )
    ax.set_xlabel("Mean AUC drop when shuffled")
    ax.set_title("Permutation Importance (MLP, no augmentation)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = prepare_data()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    pos_weight, scaler = data["pos_weight"], data["scaler"]

    print("Training baselines...")
    lr_full = train_full_baseline(X_train, y_train)
    lr_rpm = train_rpm_only_baseline(X_train, y_train, rpm_col_idx=RPM_IDX)

    print("Training MLP (no augmentation)...")
    mlp_plain, hist_plain = train_model(
        X_train, y_train, X_val, y_val, pos_weight, use_augmentation=False
    )

    print("Training MLP (with Gaussian jitter augmentation)...")
    mlp_aug, hist_aug = train_model(
        X_train, y_train, X_val, y_val, pos_weight, use_augmentation=True
    )

    print("Evaluating on test set...")
    results = {
        "Logistic Regression (full)": evaluate_predictions(y_test, lr_full.predict_proba(X_test)[:, 1]),
        "Logistic Regression (rpm only)": evaluate_predictions(y_test, lr_rpm.predict_proba(X_test[:, [RPM_IDX]])[:, 1]),
        "MLP (no augmentation)": evaluate_predictions(y_test, mlp_probs(mlp_plain, X_test)),
        "MLP (with augmentation)": evaluate_predictions(y_test, mlp_probs(mlp_aug, X_test)),
    }

    print("Computing permutation importance (main MLP)...")
    importances = permutation_importance_mlp(
        mlp_plain, X_test, y_test, results["MLP (no augmentation)"]["roc_auc"]
    )

    print("Computing RPM-band failure analysis...")
    bands = rpm_band_accuracy(mlp_plain, X_test, y_test, scaler)

    print("Generating figures...")
    plot_confusion_matrices(results, os.path.join(FIG_DIR, "confusion_matrices.png"))
    plot_roc_curves(y_test, results, os.path.join(FIG_DIR, "roc_curves.png"))
    plot_training_curves(
        {"MLP (no augmentation)": hist_plain, "MLP (with augmentation)": hist_aug},
        os.path.join(FIG_DIR, "training_curves.png"),
    )
    plot_permutation_importance(importances, os.path.join(FIG_DIR, "permutation_importance.png"))

    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {k: v for k, v in res.items() if k not in ("y_pred", "y_prob")}
    metrics_out["permutation_importance"] = importances
    metrics_out["rpm_band_accuracy"] = bands
    metrics_out["dataset"] = {
        "n_total": int(len(y_train) + len(y_val) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n=== Summary (test set) ===")
    for name, res in results.items():
        print(f"{name}: AUC={res['roc_auc']:.3f}, Acc={res['accuracy']:.3f}, "
              f"Recall(0)={res['recall']['class_0']:.3f}, Recall(1)={res['recall']['class_1']:.3f}")
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
