"""SVR + scaler fitting for Original RFECA / RFACA."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_svr_pipeline(
    *,
    kernel: str = "linear",
    C: float = 1.0,
    epsilon: float = 0.1,
    use_scaler: bool = True,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svr", SVR(kernel=kernel, C=C, epsilon=epsilon)))
    return Pipeline(steps)


def fit_predict_oof(
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    *,
    y_strata: np.ndarray | None,
    kernel: str,
    C: float,
    epsilon: float,
    use_scaler: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Out-of-fold predictions on rows covered by the splitter.

    Each fold builds a *fresh* Pipeline (StandardScaler + SVR). The scaler is
    fit only on that fold's training rows via ``pipe.fit(X_train, y_train)``;
    validation rows are transformed/predicted without refitting the scaler.

    Returns (y_true_concat, y_pred_concat) in splitter yield order. For
    LeaveOneOut / KFold / StratifiedKFold each sample appears in exactly one
    test fold, so len(y_true) == n_samples when all folds succeed.
    """
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    seen: set[int] = set()
    split_iter = (
        splitter.split(X, y_strata) if y_strata is not None else splitter.split(X)
    )
    for train_idx, test_idx in split_iter:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        overlap = set(map(int, test_idx)).intersection(seen)
        if overlap:
            raise AssertionError(
                f"OOF leakage: test indices reused across folds: {sorted(overlap)[:5]}"
            )
        seen.update(map(int, test_idx))
        # New scaler+SVR per fold — never fit scaler before the split.
        pipe = make_svr_pipeline(
            kernel=kernel, C=C, epsilon=epsilon, use_scaler=use_scaler
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(np.asarray(pred, dtype=float))
    if not y_true_parts:
        return np.array([]), np.array([])
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def fit_final_svr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str,
    C: float,
    epsilon: float,
    use_scaler: bool,
) -> Pipeline:
    pipe = make_svr_pipeline(
        kernel=kernel, C=C, epsilon=epsilon, use_scaler=use_scaler
    )
    pipe.fit(X, y)
    return pipe
