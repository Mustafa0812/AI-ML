"""Training loop for EngineConditionMLP. See PROJECT_PLAN.md §5.4-5.8."""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_model

SEED = 42


def _make_loader(X, y, batch_size, shuffle, seed=SEED):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    pos_weight,
    use_augmentation=False,
    jitter_sigma=0.05,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=200,
    patience=15,
    seed=SEED,
):
    """Mini-batch training with Adam, BCEWithLogitsLoss(pos_weight), early stopping on val loss."""
    torch.manual_seed(seed)
    model = build_model(seed=seed)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True, seed=seed)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            if use_augmentation:
                xb = xb + torch.randn_like(xb) * jitter_sigma
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    from preprocess import prepare_data

    data = prepare_data()
    model, history = train_model(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["pos_weight"]
    )
    print(f"Trained for {len(history['train_loss'])} epochs")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
