"""Subset generators and small utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def correlation_prefixes(ordered_candidates: list[str]) -> list[list[str]]:
    """
    Progressive prefixes of a |Pearson|-ranked candidate list (notebook get_X_values).

    Both OriginalRFECA and OriginalRFACA evaluate these prefixes; they differ only
    in the inner selector (sklearn RFE vs feature_engine RFA).
    """
    if not ordered_candidates:
        return []
    return [list(ordered_candidates[:k]) for k in range(1, len(ordered_candidates) + 1)]


def rfeca_subsets(ordered_candidates: list[str]) -> list[list[str]]:
    """Alias: RFECA evaluates the same correlation prefixes as the notebook."""
    return correlation_prefixes(ordered_candidates)


def rfaca_subsets(ordered_candidates: list[str]) -> list[list[str]]:
    """Alias: RFACA evaluates the same correlation prefixes as the notebook."""
    return correlation_prefixes(ordered_candidates)


def safe_mkdir(path: Path) -> Path:
    """Create directory; raise if path exists as a non-empty directory or file."""
    path = Path(path)
    if path.exists():
        if path.is_file():
            raise FileExistsError(f"Refusing to overwrite file: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty results directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def to_dataframe(X, feature_names: list[str] | None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        out.columns = out.columns.astype(str)
        return out
    if feature_names is None:
        raise ValueError(
            "ndarray input requires feature_names (or pass a pandas DataFrame)."
        )
    return pd.DataFrame(
        np.asarray(X, dtype=float), columns=[str(c) for c in feature_names]
    )


def column_means(X: pd.DataFrame) -> dict[str, float]:
    means = X.mean(axis=0, skipna=True)
    return {str(k): float(v) if np.isfinite(v) else 0.0 for k, v in means.items()}
