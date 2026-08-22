"""
train.py
Training loop shared by baseline and CNN.

Choices made here (see ML_Fundamentals_Reference.md for the underlying formulas):
  - Loss: CrossEntropyLoss (categorical cross-entropy over n_classes),
    optionally class-weighted for IP102's long-tailed distribution -- S1.2.
  - Optimizer: Adam (mini-batch GD + momentum + adaptive learning rates) --
    an extension of the mini-batch GD covered in S2.3, not covered by name
    in the course slides, worth flagging as such in the report.
  - Regularization: dropout + batch norm (inside the CNN) + early stopping
    on validation loss -- S8.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_correct = 0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            n_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            n_samples += X_batch.size(0)

    return total_loss / n_samples, n_correct / n_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    class_weights: torch.Tensor | None = None,
    patience: int = 6,
    verbose: bool = True,
):
    """
    Trains `model` with Adam + CrossEntropyLoss, early stopping on validation
    loss. Returns the best model (lowest val loss) and full training history
    (loss + accuracy per epoch) for the report's learning-curve figure.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model, history
