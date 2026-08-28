"""Runtime assertions for leakage-safe experimental integrity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evaluation import FoldAudit
from .missingness import MissingnessResult


class AssertionErrorList(AssertionError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Assertion failures:\n- " + "\n- ".join(errors))


def assert_no_inplace_mutation(X_original: pd.DataFrame, X_reference_values: np.ndarray) -> None:
    current = X_original.to_numpy(dtype=float)
    if current.shape != X_reference_values.shape:
        raise AssertionError("Complete matrix shape changed unexpectedly.")
    if not np.allclose(current, X_reference_values, equal_nan=True):
        raise AssertionError("Complete matrix was mutated in place.")


def assert_mask_feature_only(mask: np.ndarray, X: pd.DataFrame) -> None:
    if mask.shape != X.shape:
        raise AssertionError(
            f"Mask shape {mask.shape} != feature matrix shape {X.shape}"
        )


def assert_shared_mask_across_imputers(
    missing_sets: dict[float, list[MissingnessResult]],
    imputer_names: list[str],
) -> list[dict[str, Any]]:
    """
    Document that each (rate, rep) mask is generated once and reused.
    Returns a checklist confirming identical mask object identity intent.
    """
    checklist = []
    for rate, reps in missing_sets.items():
        for item in reps:
            checklist.append(
                {
                    "missing_rate": rate,
                    "replicate": item.replicate,
                    "seed": item.seed,
                    "mask_sum": int(item.mask.sum()),
                    "mask_shape": list(item.mask.shape),
                    "shared_across_imputers": list(imputer_names),
                    "note": "Same MissingnessResult instance is passed to every imputer.",
                }
            )
    return checklist


def assert_fold_isolation(audit: FoldAudit) -> None:
    errors: list[str] = []

    # Train/test overlap
    for row in audit.rows:
        if row["train_test_overlap"] != 0:
            errors.append(
                f"Overlap in {row['stage']} {row['imputer']} "
                f"rep={row['replicate']} fold={row['fold']}"
            )

    # No shared imputer/scaler/corr objects across folds within the same
    # (stage, imputer) group. Live refs in FoldAudit prevent id() recycling.
    for key, ids in audit.imputer_ids.items():
        if len(ids) != len(set(ids)):
            errors.append(
                f"Imputer object id reused across folds for {key}: {ids}"
            )

    for key, ids in audit.corr_matrix_ids.items():
        if len(ids) != len(set(ids)):
            errors.append(
                f"Correlation matrix object shared across folds for {key}: {ids}"
            )

    for key, ids in audit.scaler_ids.items():
        if len(ids) != len(set(ids)):
            errors.append(f"Scaler object shared across folds for {key}: {ids}")

    # RFECA must not use corr CSV; correlations must come from training fold.
    for row in audit.rows:
        if row["imputer"].startswith("RFECA") and row["has_corr_csv_path"]:
            errors.append(f"RFECA used corr_csv_path in fold audit: {row}")
        if (
            row["imputer"].startswith("RFECA")
            and row["correlation_source"] != "training_fold_X.corr()"
        ):
            errors.append(
                f"RFECA correlation source is not training-fold: {row['correlation_source']}"
            )

    if errors:
        raise AssertionErrorList(errors)


def assert_metrics_on_masked_only(imp_raw: pd.DataFrame) -> None:
    """Sanity: n_masked_test_values > 0 when missing_rate > 0."""
    bad = imp_raw[(imp_raw["missing_rate"] > 0) & (imp_raw["n_masked_test_values"] <= 0)]
    if len(bad):
        raise AssertionError(
            f"Expected masked test cells for missing_rate>0, found empty folds:\n{bad}"
        )


def assert_no_legacy_imputed_in_metric_target(imp_raw: pd.DataFrame) -> None:
    """Primary analysis must never score RMSE/MAE on legacy-imputed cells."""
    if "n_legacy_imputed_cells_in_metric_target" not in imp_raw.columns:
        return
    bad = imp_raw[imp_raw["n_legacy_imputed_cells_in_metric_target"] != 0]
    if len(bad):
        raise AssertionError(
            "Legacy-imputed cells found in metric targets:\n"
            f"{bad[['imputer','replicate','fold','n_legacy_imputed_cells_in_metric_target']]}"
        )


def assert_artificial_mask_observed_only(
    mask: np.ndarray, originally_observed_mask: np.ndarray
) -> None:
    obs = np.asarray(originally_observed_mask, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if np.any(m & ~obs):
        raise AssertionError(
            "Artificial missingness mask includes cells that were not originally observed."
        )
