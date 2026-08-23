"""Data loading, splitting, scaling, and augmentation for the engine-condition project.

See PROJECT_PLAN.md §3 for the full rationale behind each step.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURE_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]
TARGET_COL = "Engine Condition"
DATA_PATH = "../data/engine_data.csv"


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def split_data(df, seed=SEED):
    """Stratified 70/15/15 train/val/test split on Engine Condition."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Fit StandardScaler on train only; transform all three splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def compute_pos_weight(y_train):
    """BCEWithLogitsLoss pos_weight = (# negative) / (# positive) in the training set."""
    y_train = np.asarray(y_train)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def add_gaussian_jitter(X, sigma=0.05, seed=SEED):
    """Add train-only Gaussian noise to standardized features (regularization).

    X is expected to already be StandardScaler-transformed, so sigma is in
    standardized units. Never apply this to val/test data.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise


def prepare_data(path=DATA_PATH, seed=SEED):
    """Full pipeline: load -> split -> scale -> pos_weight. Returns a dict."""
    df = load_data(path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=seed)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)
    pos_weight = compute_pos_weight(y_train)

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
        "pos_weight": pos_weight,
        "feature_cols": FEATURE_COLS,
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    print(f"pos_weight: {data['pos_weight'].item():.4f}")
    print(f"Train class balance: {np.bincount(data['y_train'])}")
