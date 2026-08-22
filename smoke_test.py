"""
smoke_test.py
Generates a small SYNTHETIC image dataset (ImageFolder-style: data/<class_name>/*.jpg)
with class-specific colored lesion patterns standing in for plant diseases /
pest damage, and runs the entire pipeline end-to-end: index -> select classes
-> split -> train baseline -> train CNN -> evaluate -> Grad-CAM.

This is purely to catch bugs before the real PlantVillage data arrives.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.utils import set_seed, get_device
from src.preprocessing import index_dataset, select_top_classes
from src.dataset import build_label_mapping, stratified_split, compute_class_weights, build_dataloaders
from src.models import LogisticRegressionBaseline, CNN2D
from src.train import train_model
from src.evaluate import get_predictions, print_report, plot_confusion_matrix, compare_models
from src.interpret import compute_gradcam, plot_gradcam_overlay


# class_name : (lesion_color, lesion_shape, n_spots)
SYNTH_CLASSES = {
    "Tomato___healthy":      {"color": None,              "n_spots": 0},
    "Tomato___Late_blight":  {"color": (60, 40, 20),       "n_spots": 3},   # dark brown blotches
    "Tomato___Leaf_Mold":    {"color": (180, 170, 60),     "n_spots": 4},   # yellow-green patches
    "Potato___healthy":      {"color": None,               "n_spots": 0},
    "Potato___Early_blight": {"color": (110, 70, 30),      "n_spots": 2},   # concentric brown rings
}
IMG_SIZE = 128


def make_synthetic_images(data_dir, n_per_class=40, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(data_dir, exist_ok=True)

    for class_name, spec in SYNTH_CLASSES.items():
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_per_class):
            # base leaf-green background with vein-like texture noise
            base = np.tile(np.array([50, 110, 40], dtype=np.uint8), (IMG_SIZE, IMG_SIZE, 1))
            noise = rng.integers(-20, 20, size=base.shape)
            arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)

            if spec["color"] is not None:
                for _ in range(spec["n_spots"]):
                    cx, cy = rng.integers(15, IMG_SIZE - 15, size=2)
                    r = rng.integers(6, 16)
                    color = tuple(int(c + rng.integers(-15, 15)) for c in spec["color"])
                    color = tuple(max(0, min(255, c)) for c in color)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

            img.save(os.path.join(class_dir, f"{class_name}_{i:03d}.jpg"))

    print(f"Synthetic images written to {data_dir}: {len(SYNTH_CLASSES)} classes x {n_per_class} images")


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = "data/_synthetic_smoke_test"
    os.makedirs("figures", exist_ok=True)
    make_synthetic_images(data_dir)

    df = index_dataset(data_dir)
    df = select_top_classes(df, n_classes=5, min_per_class=10)

    class_names, name_to_idx = build_label_mapping(df)
    df["label_idx"] = df["class_name"].map(name_to_idx)
    n_classes = len(class_names)
    print(f"Classes: {class_names}")

    df_train, df_val, df_test = stratified_split(df, seed=42)
    print(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    class_weights = compute_class_weights(df_train["label_idx"].tolist(), n_classes)
    print(f"Class weights: {class_weights}")

    train_loader, val_loader, test_loader = build_dataloaders(df_train, df_val, df_test, batch_size=8)

    print("\n=== Training baseline (logistic regression) ===")
    baseline = LogisticRegressionBaseline(n_classes, img_size=128)
    baseline, base_hist = train_model(
        baseline, train_loader, val_loader, device, epochs=5, lr=1e-3,
        class_weights=class_weights, verbose=True,
    )
    y_true_b, y_pred_b = get_predictions(baseline, test_loader, device)
    print_report(y_true_b, y_pred_b, class_names, "Baseline")

    print("\n=== Training CNN ===")
    cnn = CNN2D(n_classes, dropout=0.3)
    cnn, cnn_hist = train_model(
        cnn, train_loader, val_loader, device, epochs=5, lr=1e-3,
        class_weights=class_weights, verbose=True,
    )
    y_true_c, y_pred_c = get_predictions(cnn, test_loader, device)
    print_report(y_true_c, y_pred_c, class_names, "CNN")

    plot_confusion_matrix(y_true_c, y_pred_c, class_names, "CNN", save_path="figures/_smoke_cm.png")
    compare_models({"Baseline": (y_true_b, y_pred_b), "CNN": (y_true_c, y_pred_c)},
                    save_path="figures/_smoke_compare.png")
    print("Plots saved to figures/")

    # Grad-CAM on a diseased (not healthy) test sample so there's a lesion to check against
    diseased_idx = [i for i in range(len(test_loader.dataset))
                     if "healthy" not in class_names[test_loader.dataset[i][1]]]
    idx = diseased_idx[0] if diseased_idx else 0
    x_sample, y_sample = test_loader.dataset[idx]
    x_input = x_sample.unsqueeze(0)
    cam, pred_class, pred_prob = compute_gradcam(cnn, x_input, device)
    plot_gradcam_overlay(
        x_sample, cam, class_names[y_sample], class_names[pred_class], pred_prob,
        save_path="figures/_smoke_gradcam.png",
    )
    print(f"Grad-CAM OK. True={class_names[y_sample]}, Predicted={class_names[pred_class]} (p={pred_prob:.3f})")

    print("\n✅ SMOKE TEST PASSED — full pipeline runs end-to-end without errors.")


if __name__ == "__main__":
    main()
