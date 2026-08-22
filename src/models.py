"""
models.py
Baseline (logistic regression) and main model (2D-CNN, built from scratch --
not a pretrained network, keeping this "your own architecture") for
multi-class pest species classification.

Both models output raw logits over n_classes -- paired with
nn.CrossEntropyLoss, which combines softmax + categorical cross-entropy in
one numerically stable step (see ML_Fundamentals_Reference.md S1.2).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LogisticRegressionBaseline(nn.Module):
    """
    Flattens the raw (resized, normalized) image and applies one linear
    layer straight to n_classes logits. No convolution, no spatial structure
    exploited at all -- the reference point the CNN needs to beat.
    """

    def __init__(self, n_classes: int, img_size: int = 128, in_channels: int = 3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels * img_size * img_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.linear(x)  # (batch, n_classes) raw logits


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool, the repeating unit of the CNN."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, pool: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNN2D(nn.Module):
    """
    From-scratch 2D-CNN for pest image classification.

    4 conv blocks (16->32->64->128 channels) -> global average pooling ->
    dropout-regularized dense head -> n_classes logits.

    Global average pooling (rather than flatten) keeps the parameter count
    manageable and -- same reasoning as the exoplanet project's 1D-CNN -- is
    the architecture Grad-CAM is built for: self.conv_blocks output IS the
    spatial feature map interpret.py visualizes as a heatmap over the image.
    """

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv_blocks(x)     # (batch, 128, H', W') -- used for Grad-CAM
        pooled = self.gap(feats)        # (batch, 128, 1, 1)
        return self.classifier(pooled)  # (batch, n_classes) raw logits

    def forward_with_features(self, x: torch.Tensor):
        """Same as forward(), but also returns the last conv feature map for Grad-CAM."""
        feats = self.conv_blocks(x)
        pooled = self.gap(feats)
        logits = self.classifier(pooled)
        return logits, feats
