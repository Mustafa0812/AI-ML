"""
preprocessing.py
Dataset discovery for PlantVillage (leaf disease / pest-damage images).

PlantVillage is distributed as "ImageFolder" style (every Kaggle mirror uses
this layout): data_dir/<class_name>/*.jpg, where class_name is typically
"CropName___DiseaseName" or "CropName___healthy" (e.g. "Tomato___Late_blight",
"Potato___healthy"). We scan recursively and use the immediate parent folder
as the class, so a split baked into the folder structure (e.g.
data_dir/train/<class_name>/*.jpg) is ignored -- we do our own stratified
split downstream anyway, same methodology as every earlier project iteration.

A second, more general annotation-file format (index/name + path/label list,
as used by some other benchmark releases) is also supported as a fallback in
case a different mirror/dataset ends up here instead.

No hand-crafted disease-lesion-shape features here -- only generic image I/O
and resizing/normalization (the image equivalent of the exoplanet project's
STFT step), satisfying the "pure ML" constraint from earlier in this project.
"""

from __future__ import annotations
import os
from collections import Counter
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _index_format_a(data_dir: str) -> pd.DataFrame:
    """ImageFolder-style: class = immediate parent directory name."""
    rows = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                class_name = os.path.basename(root)
                rows.append({"path": os.path.join(root, fname), "class_name": class_name})
    return pd.DataFrame(rows)


def _index_format_b(data_dir: str) -> pd.DataFrame:
    """Original IP102 repo format: classes.txt + train/val/test.txt annotation lists."""
    classes_path = os.path.join(data_dir, "classes.txt")
    idx_to_name = {}
    with open(classes_path, "r") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                idx_to_name[int(parts[0])] = parts[1]

    # images typically live under data_dir/images/, but fall back to data_dir/
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = data_dir

    rows = []
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        split_path = os.path.join(data_dir, split_file)
        if not os.path.exists(split_path):
            continue
        with open(split_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                rel_path, label_idx = parts[0], int(parts[1])
                full_path = os.path.join(images_dir, rel_path)
                class_name = idx_to_name.get(label_idx, str(label_idx))
                rows.append({"path": full_path, "class_name": class_name})
    return pd.DataFrame(rows)


def index_dataset(data_dir: str) -> pd.DataFrame:
    """
    Auto-detects which format is present in data_dir and returns a DataFrame
    with columns ['path', 'class_name'] covering every image found (all
    splits pooled -- we re-split ourselves in dataset.py).
    """
    if os.path.exists(os.path.join(data_dir, "classes.txt")):
        df = _index_format_b(data_dir)
        fmt = "B (original IP102 annotation format)"
    else:
        df = _index_format_a(data_dir)
        fmt = "A (ImageFolder-style directories)"

    if df.empty:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Expected either an ImageFolder-style "
            f"layout (data/<class_name>/*.jpg) or the original IP102 files "
            f"(classes.txt + train.txt/val.txt/test.txt + images/)."
        )
    print(f"Detected dataset format: {fmt}. Found {len(df)} images across "
          f"{df['class_name'].nunique()} classes.")
    return df


def select_top_classes(df: pd.DataFrame, n_classes: int = 12, min_per_class: int = 50) -> pd.DataFrame:
    """
    IP102 has a long-tailed distribution (some of the 102 classes have very
    few images). For a tractable exam project, keep only the n_classes most
    represented species with at least min_per_class images each.
    """
    counts = Counter(df["class_name"])
    eligible = [c for c, n in counts.items() if n >= min_per_class]
    top_classes = sorted(eligible, key=lambda c: counts[c], reverse=True)[:n_classes]

    filtered = df[df["class_name"].isin(top_classes)].reset_index(drop=True)
    print(f"Selected top {len(top_classes)} classes (of {df['class_name'].nunique()} total):")
    for c in top_classes:
        print(f"  {c}: {counts[c]} images")
    return filtered
