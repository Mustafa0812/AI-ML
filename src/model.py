"""MLP classifier for engine condition prediction. See PROJECT_PLAN.md §5.1-5.3."""
import torch
import torch.nn as nn

SEED = 42


class EngineConditionMLP(nn.Module):
    """6 -> 32 -> 16 -> 8 -> 1, BatchNorm+ReLU+Dropout, raw logit output.

    Sigmoid is applied inside BCEWithLogitsLoss during training (see train.py)
    and explicitly via torch.sigmoid() at inference time.
    """

    def __init__(self, input_dim=6, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(seed=SEED):
    torch.manual_seed(seed)
    return EngineConditionMLP()


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {n_params}")
