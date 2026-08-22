"""
dataset.py
PyTorch Dataset for plant leaf images, stratified splitting, and class-imbalance
handling via loss weighting (IP102's long tail means even the top-N selected
classes won't be perfectly balanced).

Design decisions worth citing in the report:
  - Data augmentation (flip/rotation/color jitter) is applied ONLY to the
    training split -- val/test use a deterministic resize+normalize only,
    so evaluation reflects real, unaugmented images.
  - Stratified split (same principle as the exoplanet project) so every
    class is represented proportionally in train/val/test.
  - Class weights (inverse frequency) are passed to CrossEntropyLoss rather
    than oversampling images -- simpler for multi-class, and avoids
    duplicating/interpolating actual photographs.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

IMG_SIZE = 128

# ImageNet-style normalization stats are a reasonable default even for a
# from-scratch model -- they just rescale channels to roughly zero-mean,
# unit-variance, not a form of hand-crafted feature engineering.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class LeafImageDataset(Dataset):
    """Wraps a DataFrame of (path, label_idx) rows with a torchvision transform."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def build_label_mapping(df: pd.DataFrame):
    """Sorted class name list <-> index mapping, shared across train/val/test."""
    class_names = sorted(df["class_name"].unique())
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, name_to_idx


def stratified_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42):
    """Three-way stratified split by class_name. Returns three DataFrames."""
    df_temp, df_test = train_test_split(
        df, test_size=test_size, stratify=df["class_name"], random_state=seed
    )
    relative_val_size = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_temp, test_size=relative_val_size, stratify=df_temp["class_name"], random_state=seed
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for nn.CrossEntropyLoss(weight=...).
    Up-weights rarer species so the loss doesn't just favor the most common ones.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for any absent class
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(df_train, df_val, df_test, batch_size: int = 32):
    train_ds = LeafImageDataset(df_train, train_transform)
    val_ds = LeafImageDataset(df_val, eval_transform)
    test_ds = LeafImageDataset(df_test, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
