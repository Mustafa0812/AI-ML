"""Logistic regression baselines. See PROJECT_PLAN.md §4."""
from sklearn.linear_model import LogisticRegression

SEED = 42


def train_full_baseline(X_train, y_train, seed=SEED):
    """Logistic regression on all 6 scaled features, class-weighted."""
    clf = LogisticRegression(class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)
    return clf


def train_rpm_only_baseline(X_train, y_train, rpm_col_idx=0, seed=SEED):
    """Single-feature logistic regression on Engine rpm only.

    Mirrors PROJECT_PLAN.md's "no single feature comes close to sufficient" check.
    """
    clf = LogisticRegression(class_weight="balanced", random_state=seed)
    clf.fit(X_train[:, [rpm_col_idx]], y_train)
    return clf
