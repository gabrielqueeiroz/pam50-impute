"""Correlation helpers for Original RFECA / RFACA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_abs_ranking(
    y: np.ndarray | pd.Series,
    X: pd.DataFrame,
    *,
    min_periods: int = 3,
) -> tuple[list[str], dict[str, float]]:
    """
    Rank predictor columns by descending |Pearson r| with target y.

    Uses pairwise complete observations per column. ``y`` is aligned to ``X``
    by position (same row order), not by pandas index labels.
    """
    if len(y) != len(X):
        raise ValueError("y and X must have the same number of rows.")
    # Position-based alignment: avoid index mismatch (RangeIndex vs sample_id).
    y_s = pd.Series(np.asarray(y, dtype=float), index=X.index)
    scores: dict[str, float] = {}
    for col in X.columns:
        r = y_s.corr(X[col], method="pearson", min_periods=min_periods)
        if pd.isna(r):
            continue
        scores[str(col)] = float(r)
    order = sorted(scores.keys(), key=lambda c: abs(scores[c]), reverse=True)
    return order, scores


def complete_predictor_mask(X: pd.DataFrame) -> pd.Series:
    """True for columns with zero NaN."""
    return ~X.isna().any(axis=0)
