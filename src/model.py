"""LSTM regressor for turbofan Remaining Useful Life (RUL) prediction."""
import torch
import torch.nn as nn

SEED = 42


class TurbofanRULLSTM(nn.Module):
    """Sequence(30, 15) -> LSTM(hidden=64) -> FC head -> scalar RUL.

    No BatchNorm: normalizing across a recurrent hidden state mixes
    per-timestep statistics in ways that are inappropriate for sequence
    models (LayerNorm would be the correct alternative if normalization
    were needed). At this size, dropout + weight decay + early stopping
    are sufficient regularization.
    """

    def __init__(self, input_dim=15, hidden_size=64, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def build_model(seed=SEED, input_dim=15):
    torch.manual_seed(seed)
    return TurbofanRULLSTM(input_dim=input_dim)


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {n_params}")
