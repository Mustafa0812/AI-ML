"""
interpret.py
Grad-CAM for the 2D-CNN: shows which region of a pest image the model
weighted most heavily when making its prediction -- the classic Grad-CAM
use case (this is the architecture/task Grad-CAM was originally designed
for, unlike the 1D adaptation used in the earlier exoplanet project).

Because CNN2D ends in global average pooling before the classifier head
(see models.py), the CAM weight for each channel is just the average
gradient of the target class's logit w.r.t. that channel's spatial feature
map -- then a weighted sum across channels, ReLU'd and upsampled back to the
original image resolution.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models import CNN2D
from src.dataset import NORM_MEAN, NORM_STD


def _unnormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) numpy array in [0, 1] for display."""
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def compute_gradcam(model: CNN2D, x: torch.Tensor, device: torch.device, target_class: int | None = None):
    """
    x: single sample, shape (1, 3, H, W).
    target_class: class index to explain; defaults to the model's own top prediction.

    Returns:
        cam_upsampled: np.ndarray, shape (H, W), values in [0, 1]
        pred_class: int, the predicted class index
        pred_prob: float, softmax probability of pred_class
    """
    model.eval()
    x = x.to(device)
    H, W = x.shape[-2], x.shape[-1]

    logits, feats = model.forward_with_features(x)   # feats: (1, C, H', W')
    feats.retain_grad()

    probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1).item() if target_class is None else target_class
    pred_prob = probs[0, pred_class].item()

    model.zero_grad()
    logits[0, pred_class].backward()

    grads = feats.grad                                     # (1, C, H', W')
    weights = grads.mean(dim=(2, 3), keepdim=True)          # GAP of gradients -> (1, C, 1, 1)
    cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))  # (1, 1, H', W')

    cam_upsampled = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam_upsampled = cam_upsampled.squeeze().detach().cpu().numpy()

    cam_min, cam_max = cam_upsampled.min(), cam_upsampled.max()
    cam_upsampled = (cam_upsampled - cam_min) / (cam_max - cam_min + 1e-8)

    return cam_upsampled, pred_class, pred_prob


def plot_gradcam_overlay(
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    true_class_name: str,
    pred_class_name: str,
    pred_prob: float,
    save_path: str | None = None,
):
    """
    img_tensor: (3, H, W) normalized tensor (as fed to the model).
    cam: (H, W) array in [0, 1] from compute_gradcam.
    Side-by-side: original image | Grad-CAM heatmap overlaid on the image.
    """
    img = _unnormalize(img_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    correct = "✓" if true_class_name == pred_class_name else "✗"
    fig.suptitle(f"True: {true_class_name} | Predicted: {pred_class_name} "
                 f"(p={pred_prob:.2f}) {correct}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
