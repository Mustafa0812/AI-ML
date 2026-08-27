"""Full evaluation for the C-MAPSS FD001 RUL project: baselines + LSTM,
window-length ablation, PHM08 scoring, RUL-band failure analysis,
permutation importance, and all report figures.

Run from src/: `python evaluate.py`
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines import train_cycle_only_baseline, train_snapshot_baseline
from preprocess import FEATURE_COLS, WINDOW_SIZE, prepare_data
from train import train_model

SEED = 42
FIG_DIR = "../outputs/figures"
METRICS_PATH = "../outputs/metrics.json"
RUL_BANDS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125)]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def phm_score(y_true, y_pred):
    """NASA/PHM08 asymmetric scoring function. d = predicted - true.
    Late/dangerous predictions (d>=0, overestimating remaining life) are
    penalized more steeply (divisor 10) than early/conservative ones
    (d<0, divisor 13) — overestimating remaining life is the safety-critical
    failure mode (maintenance could be scheduled after the engine fails)."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return {"total": float(s.sum()), "mean": float(s.mean())}


def _phm_score_sanity_check():
    """d=+10 (late by 10) must score higher than d=-10 (early by 10)."""
    late = phm_score(np.array([100.0]), np.array([110.0]))["total"]   # d=+10
    early = phm_score(np.array([100.0]), np.array([90.0]))["total"]   # d=-10
    assert late > early, f"PHM score sign error: late={late}, early={early}"
    assert abs(late - 1.71828) < 1e-3, f"late score {late} != expected 1.718"
    assert abs(early - 1.15799) < 1e-3, f"early score {early} != expected 1.158"


def lstm_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def rul_band_error(y_true, y_pred, bands=RUL_BANDS):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = []
    for lo, hi in bands:
        mask = (y_true >= lo) & (y_true <= hi) if hi == bands[-1][1] else (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        out.append({
            "band": f"{lo}-{hi}",
            "n": n,
            "rmse": rmse(y_true[mask], y_pred[mask]) if n > 0 else None,
            "mae": mae(y_true[mask], y_pred[mask]) if n > 0 else None,
        })
    return out


def permutation_importance_lstm(model, X_test, y_test, baseline_rmse, n_repeats=10, seed=SEED):
    """Shuffle each sensor channel's full time series across the batch
    dimension (preserving within-window temporal structure), measure the
    RMSE *increase* (bigger = more important — the sign is flipped relative
    to an AUC-drop convention, since here a higher loss after shuffling
    means the model relied on that channel)."""
    rng = np.random.default_rng(seed)
    n = X_test.shape[0]
    importances = {}
    for i, col in enumerate(FEATURE_COLS):
        increases = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, :, i] = X_test[rng.permutation(n), :, i]
            preds = lstm_predict(model, X_perm)
            increases.append(rmse(y_test, preds) - baseline_rmse)
        importances[col] = {"mean": float(np.mean(increases)), "std": float(np.std(increases))}
    return importances


def build_trajectory(model, unit_df, test_rul_value, unit_last_cycle, window_size=WINDOW_SIZE):
    """Predicted RUL at every cycle of one test unit (trailing window ending
    at each cycle, left-padded for early cycles), alongside the reconstructed
    true RUL curve. RUL decreases by exactly 1/cycle, so the true RUL at any
    earlier cycle t is recoverable from the one given ground-truth value:
    true_RUL(t) = test_rul_value + (last_cycle - t), capped at 125.
    """
    X = unit_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    cycles = unit_df["cycle"].to_numpy()
    n = len(X)
    windows = []
    for end in range(1, n + 1):
        start = end - window_size
        if start < 0:
            pad = np.repeat(X[[0]], -start, axis=0)
            windows.append(np.concatenate([pad, X[:end]], axis=0))
        else:
            windows.append(X[start:end])
    preds = lstm_predict(model, np.stack(windows).astype(np.float32))
    true_rul = np.clip(test_rul_value + (unit_last_cycle - cycles), a_min=None, a_max=125)
    return cycles, true_rul, preds


def plot_pred_vs_actual(results, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [0, 130]
    ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect prediction")
    for name, res in results.items():
        ax.scatter(res["y_true"], res["y_pred"], alpha=0.6, s=25, label=f"{name} (RMSE={res['rmse']:.1f})")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Predicted vs. True RUL (test units)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(histories, path):
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist["train_loss"], label="Train loss (MSE)")
        ax.plot(hist["val_loss"], label="Val loss (MSE)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rul_band_error(bands, path):
    labels = [b["band"] for b in bands]
    rmses = [b["rmse"] if b["rmse"] is not None else 0 for b in bands]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, rmses, color="steelblue")
    ax.set_xlabel("True RUL band (cycles-to-failure)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("LSTM Error by RUL Band (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(importances, path):
    cols = list(importances.keys())
    means = [importances[c]["mean"] for c in cols]
    stds = [importances[c]["std"] for c in cols]
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([cols[i] for i in order], [means[i] for i in order],
            xerr=[stds[i] for i in order], color="steelblue")
    ax.set_xlabel("Mean RMSE increase when shuffled (cycles)")
    ax.set_title("Permutation Importance (LSTM)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_trajectories(model, data, path, n_units=3, seed=SEED):
    rng = np.random.default_rng(seed)
    test_df = data["test_df"]
    units = test_df["unit"].unique()
    sample_units = rng.choice(units, size=n_units, replace=False)

    fig, axes = plt.subplots(1, n_units, figsize=(5 * n_units, 4))
    if n_units == 1:
        axes = [axes]
    for ax, unit in zip(axes, sample_units):
        unit_df = test_df[test_df["unit"] == unit].reset_index(drop=True)
        unit_idx = list(units).index(unit)
        test_rul_value = data["y_test"][unit_idx]
        last_cycle = unit_df["cycle"].max()
        cycles, true_rul, preds = build_trajectory(model, unit_df, test_rul_value, last_cycle)
        ax.plot(cycles, true_rul, label="True RUL", color="black")
        ax.plot(cycles, preds, label="Predicted RUL", color="steelblue", alpha=0.8)
        ax.set_title(f"Test unit {unit}", fontsize=10)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    _phm_score_sanity_check()
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading and windowing data (window=30)...")
    data = prepare_data()
    n_features = len(data["feature_cols"])

    print("Training baselines...")
    lr_snap = train_snapshot_baseline(data["X_train_snap"], data["y_train_snap"])
    lr_cycle = train_cycle_only_baseline(data["cycle_train_snap"], data["y_train_snap"])

    print("Training LSTM (window=30, main model)...")
    lstm_main, hist_main = train_model(
        data["X_train_seq"], data["y_train_seq"], data["X_val_seq"], data["y_val_seq"],
        input_dim=n_features,
    )

    print("Training LSTM (window=15, ablation)...")
    data_w15 = prepare_data(window_size=15)
    lstm_w15, hist_w15 = train_model(
        data_w15["X_train_seq"], data_w15["y_train_seq"], data_w15["X_val_seq"], data_w15["y_val_seq"],
        input_dim=n_features,
    )

    print("Evaluating on test set...")
    y_test = data["y_test"]
    preds_snap = lr_snap.predict(data["X_test_snap"])
    preds_cycle = lr_cycle.predict(data["cycle_test_snap"])
    preds_lstm = lstm_predict(lstm_main, data["X_test_seq"])
    preds_lstm_w15 = lstm_predict(lstm_w15, data_w15["X_test_seq"])

    def summarize(y_true, y_pred):
        return {
            "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "phm_score": phm_score(y_true, y_pred),
            "y_true": y_true, "y_pred": y_pred,
        }

    results = {
        "Linear (cycle only)": summarize(y_test, preds_cycle),
        "Linear (snapshot)": summarize(y_test, preds_snap),
        "LSTM (window=30, main model)": summarize(y_test, preds_lstm),
        "LSTM (window=15, ablation)": summarize(y_test, preds_lstm_w15),
    }

    # Unclipped-RUL transparency check: reload the raw (unclipped) test RUL
    # for the headline model only, reported alongside the clipped metric.
    import pandas as pd
    y_test_unclipped = pd.read_csv("../data/CMaps/RUL_FD001.txt", header=None, names=["RUL"])["RUL"].to_numpy()
    unclipped_rmse_main = rmse(y_test_unclipped, preds_lstm)

    print("Computing permutation importance (main LSTM)...")
    importances = permutation_importance_lstm(
        lstm_main, data["X_test_seq"], y_test, results["LSTM (window=30, main model)"]["rmse"]
    )

    print("Computing RUL-band failure analysis (main LSTM)...")
    bands = rul_band_error(y_test, preds_lstm)

    print("Generating figures...")
    plot_pred_vs_actual(
        {"Linear (snapshot)": results["Linear (snapshot)"], "LSTM (window=30)": results["LSTM (window=30, main model)"]},
        os.path.join(FIG_DIR, "pred_vs_actual.png"),
    )
    plot_training_curves(
        {"LSTM (window=30)": hist_main, "LSTM (window=15)": hist_w15},
        os.path.join(FIG_DIR, "training_curves.png"),
    )
    plot_rul_band_error(bands, os.path.join(FIG_DIR, "rul_band_error.png"))
    plot_permutation_importance(importances, os.path.join(FIG_DIR, "permutation_importance.png"))
    plot_sample_trajectories(lstm_main, data, os.path.join(FIG_DIR, "sample_trajectories.png"))

    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {k: v for k, v in res.items() if k not in ("y_true", "y_pred")}
    metrics_out["unclipped_test_rmse_main_model"] = unclipped_rmse_main
    metrics_out["permutation_importance"] = importances
    metrics_out["rul_band_error"] = bands
    metrics_out["dataset"] = {
        "n_train_windows": int(len(data["y_train_seq"])),
        "n_val_windows": int(len(data["y_val_seq"])),
        "n_test_units": int(len(y_test)),
        "n_train_units": len(data["train_units"]),
        "n_val_units": len(data["val_units"]),
        "n_features": n_features,
        "window_size_main": WINDOW_SIZE,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n=== Summary (test set, n=100 engines) ===")
    for name, res in results.items():
        print(f"{name}: RMSE={res['rmse']:.2f}, MAE={res['mae']:.2f}, "
              f"PHM score (total)={res['phm_score']['total']:.1f}")
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
