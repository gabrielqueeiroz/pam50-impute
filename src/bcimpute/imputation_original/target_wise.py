"""TARGET-WISE matrix construction for OriginalRFECA (notebook-faithful)."""

from __future__ import annotations

import numpy as np
import pandas as pd


INPUT_PROTOCOL = "target_wise_complete_predictors"
MASK_SOURCE = "persisted_multivariate_mask_target_column"
PREDICTOR_VALUES = "original_complete_matrix"


def build_target_wise_matrix(
    X_original: pd.DataFrame,
    mask: np.ndarray,
    gene: str,
) -> pd.DataFrame:
    """
    Predictors = original values; NaN only on ``mask`` positions of ``gene``.

    Artificial NaNs from other genes in a multivariate mask are *not* applied.
    """
    if gene not in X_original.columns:
        raise KeyError(f"Gene {gene!r} not in columns.")
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != X_original.shape:
        raise ValueError(
            f"mask shape {mask_arr.shape} != X_original shape {X_original.shape}"
        )
    out = X_original.copy()
    # Ensure predictors stay original (overwrite any accidental NaNs from a
    # multivariate view by restoring from X_original — already a copy of it).
    col_i = list(X_original.columns).index(gene)
    out.iloc[mask_arr[:, col_i], col_i] = np.nan
    return out


def assert_predictors_finite(X_tw: pd.DataFrame, gene: str) -> None:
    """Raise if any predictor used by RFECA contains NaN/Inf."""
    preds = [c for c in X_tw.columns if c != gene]
    block = X_tw[preds]
    if not np.isfinite(block.to_numpy(dtype=float)).all():
        n_bad = int((~np.isfinite(block.to_numpy(dtype=float))).sum())
        raise AssertionError(
            f"TARGET-WISE violation: {n_bad} non-finite predictor values for gene={gene}."
        )


def target_column_mask(mask: np.ndarray, gene: str, columns: list[str]) -> np.ndarray:
    """Boolean vector (n_samples,) of artificial missingness for ``gene``."""
    mask_arr = np.asarray(mask, dtype=bool)
    j = list(columns).index(gene)
    return mask_arr[:, j]
