"""Data loading, RUL labeling, unit-level splitting, scaling, and windowing
for the C-MAPSS FD001 turbofan RUL project.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 42
RUL_CAP = 125
WINDOW_SIZE = 30
STRIDE = 1

TRAIN_PATH = "../data/CMaps/train_FD001.txt"
TEST_PATH = "../data/CMaps/test_FD001.txt"
RUL_PATH = "../data/CMaps/RUL_FD001.txt"

_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"sensor_{i}" for i in range(1, 22)]

# Constant in FD001 (std < 1e-3, verified on the raw training data): carry no
# information for a single-operating-condition, single-fault-mode subset.
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
DROP_COLS = ["op1", "op2", "op3"] + CONSTANT_SENSORS
FEATURE_COLS = [c for c in _COLS if c not in DROP_COLS and c not in ("unit", "cycle")]


def load_raw(path):
    """C-MAPSS .txt files are whitespace-separated with trailing whitespace
    on every line (produces spurious all-NaN columns with a naive sep=' ')."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_COLS)
    return df


def compute_train_rul(df, cap=RUL_CAP):
    """Piecewise-linear RUL target: RUL(t) = min(max_cycle(unit) - t, cap).

    Early-life cycles carry no degradation signal, so an uncapped target
    forces the model to distinguish near-identical healthy readings by an
    arbitrary large RUL value. Capping focuses capacity on the informative
    near-failure region (standard practice, e.g. Zheng et al. 2017).
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).clip(upper=cap)
    return rul.to_numpy(dtype=np.float32)


def load_test_rul(path=RUL_PATH, cap=RUL_CAP):
    """One ground-truth RUL per test unit, at that unit's last recorded cycle.

    Must come from this file, not be re-derived as max_cycle - t: test
    trajectories are arbitrarily truncated before failure.
    """
    rul = pd.read_csv(path, header=None, names=["RUL"])["RUL"].to_numpy(dtype=np.float32)
    return np.clip(rul, a_min=None, a_max=cap)


def split_units(unit_ids, test_size=0.20, seed=SEED):
    """Group-aware split so adjacent, highly-overlapping windows from the
    same engine never appear on both sides of train/val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(unit_ids))
    train_idx, val_idx = next(gss.split(idx, groups=unit_ids))
    train_units = set(np.asarray(unit_ids)[train_idx])
    val_units = set(np.asarray(unit_ids)[val_idx])
    assert train_units.isdisjoint(val_units)
    return train_units, val_units


def scale_features(train_df, *dfs):
    """Fit StandardScaler on train-unit rows only; transform train + any
    number of other dataframes (val, test) with those statistics."""
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    out = [scaler.transform(train_df[FEATURE_COLS])]
    out += [scaler.transform(d[FEATURE_COLS]) for d in dfs]
    return (*out, scaler)


def build_windows(df, rul, window_size=WINDOW_SIZE, stride=STRIDE):
    """Sliding windows per unit: X[i] = (window_size, n_features), y[i] =
    RUL at the window's last cycle. Left-pads (repeats the first row) any
    unit shorter than window_size so this stays safe if reused on subsets
    (FD002/FD004) where short trajectories can occur; FD001 never exercises
    padding since its shortest trajectory (31) exceeds window_size (30).
    """
    n_features = len(FEATURE_COLS)
    X_list, y_list = [], []
    for unit in df["unit"].unique():
        mask = (df["unit"] == unit).to_numpy()
        unit_X = df.loc[mask, FEATURE_COLS].to_numpy(dtype=np.float32)
        unit_y = rul[mask]
        n = len(unit_X)
        if n < window_size:
            # Single padded window per short unit; its label is the last
            # real (unpadded) cycle's RUL.
            pad = np.repeat(unit_X[[0]], window_size - n, axis=0)
            X_list.append(np.concatenate([pad, unit_X], axis=0))
            y_list.append(unit_y[-1])
            continue
        for end in range(window_size, n + 1, stride):
            X_list.append(unit_X[end - window_size:end])
            y_list.append(unit_y[end - 1])
    X = np.stack(X_list).astype(np.float32) if X_list else np.empty((0, window_size, n_features), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def build_last_window(df, window_size=WINDOW_SIZE):
    """One window per unit: its final window_size cycles (left-padded if the
    trajectory is shorter). Used for test-time evaluation, where exactly one
    prediction per unit is compared against RUL_FD001.txt."""
    X_list = []
    for unit in df["unit"].unique():
        unit_X = df.loc[df["unit"] == unit, FEATURE_COLS].to_numpy(dtype=np.float32)
        n = len(unit_X)
        if n < window_size:
            pad = np.repeat(unit_X[[0]], window_size - n, axis=0)
            unit_X = np.concatenate([pad, unit_X], axis=0)
        X_list.append(unit_X[-window_size:])
    return np.stack(X_list).astype(np.float32)


def prepare_data(train_path=TRAIN_PATH, test_path=TEST_PATH, rul_path=RUL_PATH,
                  seed=SEED, window_size=WINDOW_SIZE):
    """Full pipeline: load -> unit split -> scale (train-fit only) -> label -> window."""
    raw_train = load_raw(train_path)
    raw_test = load_raw(test_path)

    train_units, val_units = split_units(raw_train["unit"].unique(), seed=seed)
    train_df = raw_train[raw_train["unit"].isin(train_units)].reset_index(drop=True)
    val_df = raw_train[raw_train["unit"].isin(val_units)].reset_index(drop=True)

    train_scaled, val_scaled, test_scaled, scaler = scale_features(train_df, val_df, raw_test)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = raw_test.copy()
    train_df[FEATURE_COLS] = train_scaled
    val_df[FEATURE_COLS] = val_scaled
    test_df[FEATURE_COLS] = test_scaled

    train_rul = compute_train_rul(train_df)
    val_rul = compute_train_rul(val_df)
    test_rul = load_test_rul(rul_path)

    X_train_seq, y_train_seq = build_windows(train_df, train_rul, window_size=window_size)
    X_val_seq, y_val_seq = build_windows(val_df, val_rul, window_size=window_size)
    X_test_seq = build_last_window(test_df, window_size=window_size)

    # Snapshot (current-cycle-only, no window) views for the linear baseline —
    # same feature space and scaling as the LSTM, but no history.
    X_train_snap = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train_snap = train_rul
    X_test_snap = np.stack([
        test_df.loc[test_df["unit"] == u, FEATURE_COLS].to_numpy(dtype=np.float32)[-1]
        for u in test_df["unit"].unique()
    ])

    cycle_train_snap = train_df["cycle"].to_numpy(dtype=np.float32).reshape(-1, 1)
    cycle_test_snap = np.stack([
        test_df.loc[test_df["unit"] == u, "cycle"].to_numpy(dtype=np.float32)[-1]
        for u in test_df["unit"].unique()
    ]).reshape(-1, 1)

    return {
        "X_train_seq": X_train_seq, "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq, "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq, "y_test": test_rul,
        "X_train_snap": X_train_snap, "y_train_snap": y_train_snap,
        "X_test_snap": X_test_snap,
        "cycle_train_snap": cycle_train_snap, "cycle_test_snap": cycle_test_snap,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "train_units": train_units, "val_units": val_units,
        "test_df": test_df, "test_rul_lookup": test_rul,
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"Train windows: {data['X_train_seq'].shape}, Val windows: {data['X_val_seq'].shape}")
    print(f"Test windows: {data['X_test_seq'].shape}")
    print(f"Train/val units disjoint: {data['train_units'].isdisjoint(data['val_units'])}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
