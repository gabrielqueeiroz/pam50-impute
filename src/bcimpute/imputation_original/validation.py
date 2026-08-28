"""Internal validation strategies for subset selection (train-only)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

from .types import ValidationStrategy


def make_splitter(
    strategy: ValidationStrategy,
    *,
    n_splits: int,
    random_state: int,
    y_for_strata: np.ndarray | None = None,
):
    """
    Build a sklearn splitter for internal validation.

    StratifiedKFold uses ``y_for_strata`` only to organize folds (e.g. PAM50);
    it must never be used as an SVR feature.
    """
    strategy = strategy.lower().strip()  # type: ignore[assignment]
    if strategy == "loocv":
        return LeaveOneOut()
    if strategy == "kfold":
        return KFold(
            n_splits=max(2, int(n_splits)),
            shuffle=True,
            random_state=int(random_state),
        )
    if strategy == "stratified_kfold":
        if y_for_strata is None:
            raise ValueError(
                "stratified_kfold requires y_for_strata (external labels for fold "
                "organization only; not used as SVR predictors)."
            )
        return StratifiedKFold(
            n_splits=max(2, int(n_splits)),
            shuffle=True,
            random_state=int(random_state),
        )
    raise ValueError(f"Unknown validation strategy: {strategy!r}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
