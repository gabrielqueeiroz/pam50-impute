"""Notebook-faithful RFE / RFA wrappers (sklearn + feature_engine)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from feature_engine.selection import RecursiveFeatureAddition
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

SelectorKind = Literal["RFE", "RFA"]


def make_linear_svr() -> SVR:
    """Estimator used by the dissertation notebook for RFE and RFA."""
    return SVR(kernel="linear")


def select_features_rfe(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    """
    Apply sklearn RFE with linear SVR (notebook default: half of features).

    With a single column, RFE is skipped (matches notebook branch for 1 gene).
    """
    cols = [str(c) for c in X.columns]
    if len(cols) <= 1:
        return cols
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rfe = RFE(estimator=make_linear_svr())
    rfe.fit(X_arr, y_arr)
    return [c for c, keep in zip(cols, rfe.support_) if bool(keep)]


def select_features_rfa(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    scoring: str = "r2",
    threshold: float = 0.001,
) -> list[str]:
    """
    Apply feature_engine RecursiveFeatureAddition with linear SVR.

    Notebook call site: RecursiveFeatureAddition(SVR(kernel='linear'),
    scoring='r2', threshold=0.001). With a single column, RFA is skipped.
    """
    cols = [str(c) for c in X.columns]
    if len(cols) <= 1:
        return cols
    X_df = pd.DataFrame(np.asarray(X, dtype=float), columns=cols)
    y_arr = np.asarray(y, dtype=float)
    rfa = RecursiveFeatureAddition(
        estimator=make_linear_svr(),
        scoring=scoring,
        threshold=threshold,
    )
    rfa.fit(X_df, y_arr)
    out = list(rfa.get_feature_names_out())
    return [str(c) for c in out]


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    kind: SelectorKind,
    *,
    rfa_scoring: str = "r2",
    rfa_threshold: float = 0.001,
) -> list[str]:
    if kind == "RFE":
        return select_features_rfe(X, y)
    if kind == "RFA":
        return select_features_rfa(
            X, y, scoring=rfa_scoring, threshold=rfa_threshold
        )
    raise ValueError(f"Unknown selector kind: {kind!r}")
