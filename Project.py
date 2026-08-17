"""
CWRU Bearing Fault Classification — Exam 3 project.

4-class classification (Normal / Ball / Inner Race / Outer Race fault) from
12kHz drive-end accelerometer signal, load = 1hp only (see ML.md for why the
cross-load generalization experiment was dropped: dataset only has load-1 data).

Pipeline: raw signal -> fixed windows (z-score normalized) -> two models:
  - baseline: logistic regression on flattened raw windows
  - main: 2D-CNN on STFT spectrograms of the windows
No hand-crafted fault-frequency features are used anywhere (see ML.md's
"pure ML model" constraint).
"""
import os
import re

import numpy as np
import scipy.io as sio
from scipy.signal import stft

DATA_DIR = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\brjapon\cwru-bearing-datasets\versions\1\raw"
)

WINDOW_SIZE = 2048
WINDOW_HOP = 2048  # non-overlapping windows
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15  # remaining 0.15 -> test

LABELS = ["Normal", "Ball", "InnerRace", "OuterRace"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

FILENAME_LABEL_PATTERNS = [
    (re.compile(r"^Time_Normal"), "Normal"),
    (re.compile(r"^B\d"), "Ball"),
    (re.compile(r"^IR\d"), "InnerRace"),
    (re.compile(r"^OR\d"), "OuterRace"),
]


def label_for_filename(filename: str) -> str:
    for pattern, label in FILENAME_LABEL_PATTERNS:
        if pattern.match(filename):
            return label
    raise ValueError(f"Could not determine label for {filename}")


def load_signal(filepath: str) -> np.ndarray:
    """Load the drive-end (DE) accelerometer channel from a CWRU .mat file."""
    mat = sio.loadmat(filepath)
    file_id = re.search(r"_(\d+)\.mat$", filepath).group(1)
    key = f"X{file_id}_DE_time"
    if key not in mat:
        raise KeyError(f"{key} not found in {filepath}, keys={list(mat.keys())}")
    return mat[key].squeeze().astype(np.float64)


def windows_from_signal(signal: np.ndarray, window_size: int, hop: int) -> np.ndarray:
    n_windows = (len(signal) - window_size) // hop + 1
    windows = np.stack(
        [signal[i * hop: i * hop + window_size] for i in range(n_windows)]
    )
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / (std + 1e-8)


def build_dataset():
    """
    Returns windows/labels split by contiguous chunks *within each file*
    (train = first 70% of a file's windows, val = next 15%, test = last 15%)
    so adjacent, highly-correlated windows don't leak across the split.
    """
    splits = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".mat"):
            continue
        label = label_for_filename(filename)
        signal = load_signal(os.path.join(DATA_DIR, filename))
        windows = windows_from_signal(signal, WINDOW_SIZE, WINDOW_HOP)

        n = len(windows)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        splits["train"].append(windows[:n_train])
        splits["val"].append(windows[n_train:n_train + n_val])
        splits["test"].append(windows[n_train + n_val:])

        idx = LABEL_TO_IDX[label]
        split_labels["train"].append(np.full(n_train, idx))
        split_labels["val"].append(np.full(n_val, idx))
        split_labels["test"].append(np.full(n - n_train - n_val, idx))

    out = {}
    for split in ("train", "val", "test"):
        out[split] = (
            np.concatenate(splits[split]).astype(np.float32),
            np.concatenate(split_labels[split]).astype(np.int64),
        )
    return out


def windows_to_spectrograms(windows: np.ndarray, fs: int = 12000,
                             nperseg: int = 128, noverlap: int = 96) -> np.ndarray:
    """STFT magnitude spectrogram per window -> (N, 1, freq_bins, time_bins)."""
    _, _, Zxx = stft(windows, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
    mag = np.abs(Zxx).astype(np.float32)
    mag = np.log1p(mag)
    mean = mag.mean(axis=(1, 2), keepdims=True)
    std = mag.std(axis=(1, 2), keepdims=True)
    mag = (mag - mean) / (std + 1e-8)
    return mag[:, None, :, :]  # add channel dim


if __name__ == "__main__":
    data = build_dataset()
    for split, (X, y) in data.items():
        counts = {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))}
        print(f"{split}: {X.shape[0]} windows, class counts = {counts}")

    X_train_spec = windows_to_spectrograms(data["train"][0])
    print("Example spectrogram batch shape:", X_train_spec.shape)
