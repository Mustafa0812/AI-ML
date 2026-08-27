"""Linear regression baselines for RUL prediction."""
from sklearn.linear_model import LinearRegression


def train_snapshot_baseline(X_train_snap, y_train_snap):
    """Linear regression on the current-cycle sensor snapshot only (no
    history) — same scaled 15-feature space the LSTM sees, but with zero
    access to the preceding cycles. The head-to-head contrast against the
    LSTM isolates the value of temporal context."""
    reg = LinearRegression()
    reg.fit(X_train_snap, y_train_snap)
    return reg


def train_cycle_only_baseline(cycle_train, y_train_snap):
    """Trivial single-feature baseline: RUL regressed on elapsed cycle count
    alone, no sensor data at all — checks whether a naive "engines degrade
    linearly with time" heuristic is sufficient by itself."""
    reg = LinearRegression()
    reg.fit(cycle_train, y_train_snap)
    return reg
