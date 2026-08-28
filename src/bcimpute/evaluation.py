"""Cross-validated imputation and classification evaluation."""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import TARGET_LABELS
from .imputers import RFECAImputerSVR
from .pipelines import make_pipelines_with_imputer


class FoldAudit:
    """Collect per-fold isolation evidence for assertions/reporting."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        # Strong references prevent Python id() recycling from false positives.
        self._live_refs: list[Any] = []
        # Keys: (stage, imputer_name) -> list of object ids while still referenced.
        self.imputer_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.scaler_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.corr_matrix_ids: dict[tuple[str, str], list[int]] = defaultdict(list)

    def add(
        self,
        *,
        stage: str,
        imputer_name: str,
        missing_rate: float,
        replicate: int,
        fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fitted_imputer,
        pipeline=None,
    ) -> None:
        overlap = set(train_idx).intersection(set(test_idx))
        self._live_refs.append(fitted_imputer)
        if pipeline is not None:
            self._live_refs.append(pipeline)

        self.rows.append(
            {
                "stage": stage,
                "imputer": imputer_name,
                "missing_rate": missing_rate,
                "replicate": replicate,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_test_overlap": int(len(overlap)),
                "imputer_object_id": id(fitted_imputer),
                "correlation_source": getattr(
                    fitted_imputer, "_correlation_source_", "n/a"
                ),
                "has_corr_csv_path": bool(
                    getattr(fitted_imputer, "corr_csv_path", None)
                ),
            }
        )
        key = (stage, imputer_name)
        self.imputer_ids[key].append(id(fitted_imputer))
        if isinstance(fitted_imputer, RFECAImputerSVR):
            corr = getattr(fitted_imputer, "correlation_matrix_", None)
            if corr is not None:
                self._live_refs.append(corr)
                self.corr_matrix_ids[key].append(id(corr))
            w = getattr(fitted_imputer, "_last_transform_warnings_", [])
            for item in w:
                self.warnings.append(
                    f"{imputer_name}|rate={missing_rate}|rep={replicate}|fold={fold}|{item}"
                )

        if pipeline is not None:
            # Collect fitted StandardScaler objects from nested pipelines.
            for name, step in pipeline.named_steps.items():
                if isinstance(step, StandardScaler):
                    self._live_refs.append(step)
                    self.scaler_ids[(stage, f"{imputer_name}:{name}")].append(id(step))
                if hasattr(step, "named_estimators_"):
                    for est_name, est in step.named_estimators_.items():
                        if isinstance(est, StandardScaler):
                            self._live_refs.append(est)
                            self.scaler_ids[
                                (stage, f"{imputer_name}:ensemble.{est_name}")
                            ].append(id(est))
                        if hasattr(est, "named_steps"):
                            for sname, sstep in est.named_steps.items():
                                if isinstance(sstep, StandardScaler):
                                    self._live_refs.append(sstep)
                                    self.scaler_ids[
                                        (
                                            stage,
                                            f"{imputer_name}:ensemble.{est_name}.{sname}",
                                        )
                                    ].append(id(sstep))


def _as_frame(X, columns=None, index=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns, index=index)


def _correlation_preservation(
    X_true: np.ndarray,
    X_imp: np.ndarray,
) -> dict[str, float]:
    """
    Gene–gene correlation preservation on a held-out fold.

    Compares Pearson correlation of the complete ground-truth fold vs the
    imputed fold (all cells). Lower Frobenius / MAE = better preservation.
    RV coefficient near 1 = better agreement of correlation structure.
    """
    n_samples, n_features = X_true.shape
    if n_samples < 3 or n_features < 2:
        return {
            "corr_frobenius": float("nan"),
            "corr_frobenius_rel": float("nan"),
            "corr_mae_offdiag": float("nan"),
            "corr_rv": float("nan"),
        }

    # Row-wise nan-safe: imputed matrix should be finite after transform.
    if not np.isfinite(X_imp).all():
        X_imp = np.nan_to_num(X_imp, nan=np.nanmean(X_imp))

    c_true = np.corrcoef(X_true, rowvar=False)
    c_imp = np.corrcoef(X_imp, rowvar=False)
    if not np.isfinite(c_true).all() or not np.isfinite(c_imp).all():
        return {
            "corr_frobenius": float("nan"),
            "corr_frobenius_rel": float("nan"),
            "corr_mae_offdiag": float("nan"),
            "corr_rv": float("nan"),
        }

    diff = c_true - c_imp
    fro = float(np.linalg.norm(diff, ord="fro"))
    fro_true = float(np.linalg.norm(c_true, ord="fro"))
    fro_rel = fro / fro_true if fro_true > 1e-12 else float("nan")

    eye = np.eye(n_features, dtype=bool)
    mae_off = float(np.mean(np.abs(diff[~eye])))

    # RV coefficient between correlation matrices (vectorized Frobenius form).
    # RV(A,B) = <A,B>_F / (||A||_F ||B||_F)
    num = float(np.sum(c_true * c_imp))
    den = fro_true * float(np.linalg.norm(c_imp, ord="fro"))
    rv = num / den if den > 1e-12 else float("nan")

    return {
        "corr_frobenius": fro,
        "corr_frobenius_rel": fro_rel,
        "corr_mae_offdiag": mae_off,
        "corr_rv": rv,
    }


def run_imputation_cv(
    X_full: pd.DataFrame,
    X_missing: pd.DataFrame,
    mask: np.ndarray,
    y: pd.Series,
    imputers: dict,
    *,
    n_splits: int,
    random_state: int,
    missing_rate: float,
    replicate: int,
    audit: FoldAudit,
    originally_observed_mask: np.ndarray | None = None,
    n_eligible_cells: int | None = None,
    n_legacy_imputed_cells_in_cohort: int | None = None,
    target_cell_policy: str = "originally_observed_only",
) -> pd.DataFrame:
    """
    Evaluate imputation quality without leakage.

    Metrics are computed ONLY on artificially masked cells in the test fold.
    Under the primary policy those cells are a subset of originally observed cells.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict] = []

    X_full_arr = X_full.to_numpy(dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)
    obs_arr = (
        np.asarray(originally_observed_mask, dtype=bool)
        if originally_observed_mask is not None
        else np.ones_like(mask_arr, dtype=bool)
    )

    # Integrity: original complete matrix must remain finite.
    if not np.isfinite(X_full_arr).all():
        raise AssertionError("X_full contains non-finite values.")

    # Primary safeguard: no legacy-imputed cell in artificial mask.
    if target_cell_policy == "originally_observed_only":
        n_bad = int((mask_arr & ~obs_arr).sum())
        if n_bad:
            raise AssertionError(
                f"Artificial mask includes {n_bad} legacy-imputed cells "
                "(forbidden under originally_observed_only)."
            )

    for imputer_name, imputer in imputers.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_missing, y), start=1):
            if set(train_idx).intersection(set(test_idx)):
                raise AssertionError("Train/test index overlap detected.")

            X_train = X_missing.iloc[train_idx]
            X_test = X_missing.iloc[test_idx]

            imp = clone(imputer)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                imp.fit(X_train)
                X_test_imp = _as_frame(
                    imp.transform(X_test),
                    columns=X_full.columns,
                    index=X_test.index,
                )

            for w in caught:
                audit.warnings.append(
                    f"imputation|{imputer_name}|rep={replicate}|fold={fold}|{w.category.__name__}:{w.message}"
                )

            audit.add(
                stage="imputation",
                imputer_name=imputer_name,
                missing_rate=missing_rate,
                replicate=replicate,
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                fitted_imputer=imp,
            )

            m_test = mask_arr[test_idx]
            # Mask must only cover feature cells (shape equals X).
            if m_test.shape != X_test.shape:
                raise AssertionError("Mask shape mismatch vs features.")

            # Metric target: artificially masked AND originally observed
            obs_test = obs_arr[test_idx]
            metric_mask = m_test & obs_test
            n_legacy_in_target = int((m_test & ~obs_test).sum())
            if target_cell_policy == "originally_observed_only" and n_legacy_in_target != 0:
                raise AssertionError(
                    "Legacy-imputed cells present in RMSE/MAE target (must be 0)."
                )

            true_vals = X_full_arr[test_idx][metric_mask]
            pred_vals = X_test_imp.to_numpy(dtype=float)[metric_mask]

            if true_vals.size == 0:
                rmse = np.nan
                mae = np.nan
                n_masked = 0
            else:
                diff = pred_vals - true_vals
                rmse = float(np.sqrt(np.mean(diff**2)))
                mae = float(np.mean(np.abs(diff)))
                n_masked = int(true_vals.size)

            # Covariance / correlation structure on the full test fold
            # (truth vs imputed matrix), independent of the masked-cell RMSE.
            cov_metrics = _correlation_preservation(
                X_full_arr[test_idx],
                X_test_imp.to_numpy(dtype=float),
            )

            # Degenerate prediction check
            degenerate = False
            if pred_vals.size > 0 and np.nanstd(pred_vals) == 0 and n_masked > 1:
                degenerate = True
                audit.warnings.append(
                    f"degenerate_imputation|{imputer_name}|rep={replicate}|fold={fold}"
                )

            rows.append(
                {
                    "stage": "imputation",
                    "imputer": imputer_name,
                    "missing_rate": missing_rate,
                    "replicate": replicate,
                    "fold": fold,
                    "rmse": rmse,
                    "mae": mae,
                    "corr_frobenius": cov_metrics["corr_frobenius"],
                    "corr_frobenius_rel": cov_metrics["corr_frobenius_rel"],
                    "corr_mae_offdiag": cov_metrics["corr_mae_offdiag"],
                    "corr_rv": cov_metrics["corr_rv"],
                    "n_masked_test_values": n_masked,
                    "n_eligible_cells": int(
                        n_eligible_cells
                        if n_eligible_cells is not None
                        else obs_arr.sum()
                    ),
                    "n_artificially_masked_cells": int(mask_arr.sum()),
                    "n_legacy_imputed_cells_in_cohort": int(
                        n_legacy_imputed_cells_in_cohort
                        if n_legacy_imputed_cells_in_cohort is not None
                        else (~obs_arr).sum()
                    ),
                    "n_legacy_imputed_cells_in_metric_target": n_legacy_in_target,
                    "target_cell_policy": target_cell_policy,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "degenerate_constant_preds": degenerate,
                }
            )

    return pd.DataFrame(rows)


def run_classification_cv(
    X_missing: pd.DataFrame,
    y: pd.Series,
    imputers: dict,
    *,
    classifier_name: str,
    n_splits: int,
    random_state: int,
    missing_rate: float,
    replicate: int,
    audit: FoldAudit,
) -> pd.DataFrame:
    """Classify after imputation; scaler/imputer fitted per training fold only."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict] = []
    # Prefer canonical PAM50 order; fall back to observed labels if subset.
    observed = set(pd.Series(y).astype(str).unique())
    labels = [lab for lab in TARGET_LABELS if lab in observed] or sorted(observed)

    for imputer_name, imputer in imputers.items():
        pipelines = make_pipelines_with_imputer(imputer, random_state=random_state)
        if classifier_name not in pipelines:
            raise KeyError(
                f"Classifier {classifier_name!r} not in {list(pipelines)}"
            )
        base_pipe = pipelines[classifier_name]

        for fold, (train_idx, test_idx) in enumerate(cv.split(X_missing, y), start=1):
            if set(train_idx).intersection(set(test_idx)):
                raise AssertionError("Train/test index overlap detected.")

            X_train = X_missing.iloc[train_idx]
            X_test = X_missing.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fold class distribution (for report)
            train_dist = y_train.value_counts().to_dict()
            test_dist = y_test.value_counts().to_dict()

            pipe = clone(base_pipe)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

            for w in caught:
                audit.warnings.append(
                    f"classification|{imputer_name}|{classifier_name}|rep={replicate}|fold={fold}|{w.category.__name__}:{w.message}"
                )

            fitted_imp = pipe.named_steps["imputer"]
            audit.add(
                stage="classification",
                imputer_name=imputer_name,
                missing_rate=missing_rate,
                replicate=replicate,
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                fitted_imputer=fitted_imp,
                pipeline=pipe,
            )

            f1 = float(f1_score(y_test, preds, average="macro", labels=labels, zero_division=0))
            bal = float(balanced_accuracy_score(y_test, preds))
            per_f1 = f1_score(y_test, preds, average=None, labels=labels, zero_division=0)
            per_prec = precision_score(
                y_test, preds, average=None, labels=labels, zero_division=0
            )
            per_rec = recall_score(
                y_test, preds, average=None, labels=labels, zero_division=0
            )

            # Degenerate: predicting a single class
            degenerate = len(set(preds)) <= 1
            if degenerate:
                audit.warnings.append(
                    f"degenerate_classification|{imputer_name}|{classifier_name}|rep={replicate}|fold={fold}"
                )

            row: dict[str, Any] = {
                "stage": "classification",
                "imputer": imputer_name,
                "model": classifier_name,
                "missing_rate": missing_rate,
                "replicate": replicate,
                "fold": fold,
                "f1_macro": f1,
                "bal_acc": bal,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_class_distribution": str(
                    dict(sorted((str(k), int(v)) for k, v in train_dist.items()))
                ),
                "test_class_distribution": str(
                    dict(sorted((str(k), int(v)) for k, v in test_dist.items()))
                ),
                "n_unique_predictions": int(len(set(preds))),
                "degenerate_single_class_preds": degenerate,
            }
            for lab, f1_c, prec_c, rec_c in zip(labels, per_f1, per_prec, per_rec):
                row[f"f1_{lab}"] = float(f1_c)
                row[f"precision_{lab}"] = float(prec_c)
                row[f"recall_{lab}"] = float(rec_c)
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_imputation(raw: pd.DataFrame) -> pd.DataFrame:
    agg_kwargs: dict[str, tuple[str, str]] = {
        "rmse_mean": ("rmse", "mean"),
        "rmse_std": ("rmse", "std"),
        "mae_mean": ("mae", "mean"),
        "mae_std": ("mae", "std"),
        "n_rows": ("rmse", "count"),
        "n_masked_total": ("n_masked_test_values", "sum"),
    }
    for col, prefix in [
        ("corr_frobenius", "corr_frobenius"),
        ("corr_frobenius_rel", "corr_frobenius_rel"),
        ("corr_mae_offdiag", "corr_mae_offdiag"),
        ("corr_rv", "corr_rv"),
    ]:
        if col in raw.columns:
            agg_kwargs[f"{prefix}_mean"] = (col, "mean")
            agg_kwargs[f"{prefix}_std"] = (col, "std")

    g = raw.groupby(["imputer", "missing_rate"], as_index=False).agg(**agg_kwargs)
    return g


def summarize_classification(raw: pd.DataFrame) -> pd.DataFrame:
    g = (
        raw.groupby(["imputer", "missing_rate", "model"], as_index=False)
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            bal_mean=("bal_acc", "mean"),
            bal_std=("bal_acc", "std"),
            n_rows=("f1_macro", "count"),
        )
    )
    return g


def summarize_classification_per_class(raw: pd.DataFrame) -> pd.DataFrame:
    """Long-form mean±std of per-class F1/precision/recall when columns exist."""
    metric_cols = [
        c
        for c in raw.columns
        if c.startswith(("f1_", "precision_", "recall_")) and c != "f1_macro"
    ]
    if not metric_cols:
        return pd.DataFrame(
            columns=[
                "imputer",
                "missing_rate",
                "model",
                "subtype",
                "metric",
                "mean",
                "std",
                "n_rows",
            ]
        )

    records: list[dict[str, Any]] = []
    keys = ["imputer", "missing_rate", "model"]
    for (imputer, rate, model), grp in raw.groupby(keys, sort=False):
        for col in metric_cols:
            kind, subtype = col.split("_", 1)
            records.append(
                {
                    "imputer": imputer,
                    "missing_rate": rate,
                    "model": model,
                    "subtype": subtype,
                    "metric": kind,
                    "mean": float(grp[col].mean()),
                    "std": float(grp[col].std(ddof=1)) if len(grp) > 1 else float("nan"),
                    "n_rows": int(len(grp)),
                }
            )
    return pd.DataFrame(records)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_imputation_cv_target_wise(
    X_full: pd.DataFrame,
    mask: np.ndarray,
    y: pd.Series,
    imputer,
    *,
    n_splits: int,
    random_state: int,
    missing_rate: float,
    replicate: int,
    audit: FoldAudit | None = None,
    fold_limit: int | None = None,
    imputer_name: str = "OriginalRFECA",
    checkpoint_root: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Outer StratifiedKFold evaluation for TARGET-WISE OriginalRFECA.

    Kept for audit / sensitivity only. The principal benchmark protocol is
    ``run_imputation_repeated_mask_holdout_target_wise`` (evaluation_protocol=
    repeated_mask_holdout), which does not multiply cost by n_splits.
    """
    from pathlib import Path

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mask_arr = np.asarray(mask, dtype=bool)
    X_full_arr = X_full.to_numpy(dtype=float)
    if not np.isfinite(X_full_arr).all():
        raise AssertionError("X_full must be fully finite for TARGET-WISE.")
    if mask_arr.shape != X_full.shape:
        raise ValueError("mask shape mismatch vs X_full.")

    rows: list[dict] = []
    fold_audits: list[dict] = []
    audit = audit or FoldAudit()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_full, y), start=1):
        if fold_limit is not None and fold > int(fold_limit):
            break
        if set(train_idx).intersection(set(test_idx)):
            raise AssertionError("Train/test index overlap detected.")

        imp = clone(imputer)
        if checkpoint_root is not None:
            ck = Path(checkpoint_root) / f"fold_{fold}"
            ck.mkdir(parents=True, exist_ok=True)
            imp.checkpoint_dir = str(ck)

        X_tr = X_full.iloc[train_idx]
        mask_tr = mask_arr[train_idx]
        X_te = X_full.iloc[test_idx]
        mask_te = mask_arr[test_idx]

        imp.set_target_wise_context(X_tr, mask_tr, for_transform=False)
        t0 = time.perf_counter()
        imp.fit(X_tr)
        t_fit = time.perf_counter() - t0

        imp.set_target_wise_context(X_te, mask_te, for_transform=True)
        t1 = time.perf_counter()
        X_te_imp = imp.transform(X_te)
        t_tr = time.perf_counter() - t1

        audit.add(
            stage="imputation",
            imputer_name=imputer_name,
            missing_rate=missing_rate,
            replicate=replicate,
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            fitted_imputer=imp,
        )

        metric_mask = mask_te
        true_vals = X_full_arr[test_idx][metric_mask]
        pred_vals = X_te_imp.to_numpy(dtype=float)[metric_mask]
        if true_vals.size == 0:
            rmse = float("nan")
            mae = float("nan")
            n_masked = 0
        else:
            diff = pred_vals - true_vals
            rmse = float(np.sqrt(np.mean(diff**2)))
            mae = float(np.mean(np.abs(diff)))
            n_masked = int(true_vals.size)

        n_pred_nan = 0
        for gene, packed in getattr(imp, "_gene_models_", {}).items():
            _, predictors = packed
            j = list(X_full.columns).index(gene)
            rows_m = np.flatnonzero(mask_te[:, j])
            if rows_m.size == 0:
                continue
            block = X_te.iloc[rows_m][predictors].to_numpy(dtype=float)
            n_pred_nan += int((~np.isfinite(block)).sum())

        ga = imp.get_audit_dict()
        fold_audits.append(
            {
                "fold": fold,
                "fit_seconds": t_fit,
                "transform_seconds": t_tr,
                "audit": ga,
                "n_predictor_nans_at_impute": n_pred_nan,
                "svr_coverage": ga.get("svr_coverage"),
                "fallback_rate": ga.get("fallback_rate"),
            }
        )
        if n_pred_nan != 0:
            raise AssertionError(
                f"TARGET-WISE fold {fold}: predictor NaNs at impute ({n_pred_nan})."
            )

        rows.append(
            {
                "stage": "imputation",
                "imputer": imputer_name,
                "missing_rate": missing_rate,
                "replicate": replicate,
                "fold": fold,
                "rmse": rmse,
                "mae": mae,
                "n_masked_test_values": n_masked,
                "input_protocol": "target_wise_complete_predictors",
                "evaluation_protocol": "outer_cv",
                "svr_coverage": ga.get("svr_coverage"),
                "fallback_rate": ga.get("fallback_rate"),
                "fit_seconds": t_fit,
                "transform_seconds": t_tr,
                "n_rfe_fits_total": ga.get("n_rfe_fits_total"),
                "n_svr_fits_total": ga.get("n_svr_fits_total"),
            }
        )

    return pd.DataFrame(rows), fold_audits


def run_imputation_repeated_mask_holdout_target_wise(
    X_full: pd.DataFrame,
    mask: np.ndarray,
    imputer,
    *,
    missing_rate: float,
    replicate: int,
    seed: int | None = None,
    imputer_name: str = "OriginalRFECA",
    checkpoint_root: str | None = None,
    n_gene_workers: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Principal evaluation for TARGET-WISE OriginalRFECA.

    evaluation_protocol = repeated_mask_holdout

    For each gene, the persisted artificial-mask column is the external test
    set; non-masked target values are training labels. Internal leakage-safe
    CV selects the prefix only among training labels. True values at masked
    positions never enter correlation, ranking, RFE, prefix selection, or
    final SVR fit. Metrics are computed only on masked positions.

    No outer sample CV — cost is one fit per (rate × replicate), not × n_splits.

    ``n_gene_workers`` > 1 parallelizes independent genes only (methodology
    unchanged; BLAS threads remain 1 inside each worker).
    """
    if int(n_gene_workers) > 1:
        return _run_holdout_target_wise_parallel(
            X_full=X_full,
            mask=mask,
            imputer=imputer,
            missing_rate=missing_rate,
            replicate=replicate,
            seed=seed,
            imputer_name=imputer_name,
            checkpoint_root=checkpoint_root,
            n_gene_workers=int(n_gene_workers),
        )

    from pathlib import Path

    mask_arr = np.asarray(mask, dtype=bool)
    X_full_arr = X_full.to_numpy(dtype=float)
    if not np.isfinite(X_full_arr).all():
        raise AssertionError("X_full must be fully finite for TARGET-WISE.")
    if mask_arr.shape != X_full.shape:
        raise ValueError("mask shape mismatch vs X_full.")

    imp = clone(imputer)
    # OriginalRFECAImputer.__init__(**kwargs) — sklearn clone drops non-signature attrs.
    if getattr(imputer, "target_genes", None) is not None:
        imp.target_genes = list(imputer.target_genes)
    if getattr(imputer, "run_context", None) is not None:
        imp.run_context = dict(imputer.run_context)
    if checkpoint_root is not None:
        ck = Path(checkpoint_root)
        ck.mkdir(parents=True, exist_ok=True)
        imp.checkpoint_dir = str(ck)

    # Fit on full matrix with target-wise mask: masked y never used as labels.
    imp.set_target_wise_context(X_full, mask_arr, for_transform=False)
    t0 = time.perf_counter()
    imp.fit(X_full)
    t_fit = time.perf_counter() - t0

    imp.set_target_wise_context(X_full, mask_arr, for_transform=True)
    t1 = time.perf_counter()
    X_imp = imp.transform(X_full)
    t_tr = time.perf_counter() - t1

    return _finalize_holdout_metrics(
        X_full=X_full,
        X_full_arr=X_full_arr,
        mask_arr=mask_arr,
        X_imp=X_imp,
        imp=imp,
        missing_rate=missing_rate,
        replicate=replicate,
        seed=seed,
        imputer_name=imputer_name,
        t_fit=t_fit,
        t_tr=t_tr,
    )


def _finalize_holdout_metrics(
    *,
    X_full,
    X_full_arr,
    mask_arr,
    X_imp,
    imp,
    missing_rate,
    replicate,
    seed,
    imputer_name,
    t_fit,
    t_tr,
):
    from scipy.stats import pearsonr, spearmanr

    target_set = (
        set(imp.target_genes)
        if getattr(imp, "target_genes", None) is not None
        else set(X_full.columns.astype(str))
    )
    n_pred_nan = 0
    eval_mask = np.zeros_like(mask_arr, dtype=bool)
    for gene, packed in getattr(imp, "_gene_models_", {}).items():
        if gene not in target_set:
            continue
        _, predictors = packed
        j = list(X_full.columns).index(gene)
        rows_m = np.flatnonzero(mask_arr[:, j])
        eval_mask[:, j] = mask_arr[:, j]
        if rows_m.size == 0:
            continue
        block = X_full.iloc[rows_m][predictors].to_numpy(dtype=float)
        n_pred_nan += int((~np.isfinite(block)).sum())
    if n_pred_nan != 0:
        raise AssertionError(
            f"TARGET-WISE holdout: predictor NaNs at impute ({n_pred_nan})."
        )
    if not np.any(eval_mask):
        for j, gene in enumerate(X_full.columns):
            if str(gene) in target_set:
                eval_mask[:, j] = mask_arr[:, j]

    true_vals = X_full_arr[eval_mask]
    pred_vals = X_imp.to_numpy(dtype=float)[eval_mask]
    if true_vals.size == 0:
        rmse = float("nan")
        mae = float("nan")
        n_masked = 0
    else:
        diff = pred_vals - true_vals
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        n_masked = int(true_vals.size)

    ga = imp.get_audit_dict()
    audit_by_gene = {g["gene"]: g for g in ga.get("genes", [])}
    gene_metric_rows: list[dict] = []

    for j, gene in enumerate(X_full.columns):
        gene_s = str(gene)
        if gene_s not in target_set:
            continue
        m = mask_arr[:, j]
        if not np.any(m):
            continue
        tv = X_full_arr[m, j]
        pv = X_imp.to_numpy(dtype=float)[m, j]
        rec = audit_by_gene.get(gene_s, {})
        r2 = float("nan")
        pearson = float("nan")
        spearman = float("nan")
        if tv.size >= 2 and np.nanstd(tv) > 0 and np.nanstd(pv) > 0:
            ss_res = float(np.sum((tv - pv) ** 2))
            ss_tot = float(np.sum((tv - np.mean(tv)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            try:
                pearson = float(pearsonr(tv, pv)[0])
            except Exception:  # noqa: BLE001
                pearson = float("nan")
            try:
                spearman = float(spearmanr(tv, pv)[0])
            except Exception:  # noqa: BLE001
                spearman = float("nan")
        best_oof = next(
            (
                s["rmse"]
                for s in rec.get("subsets_evaluated", [])
                if s.get("prefix_len") == rec.get("winning_prefix_len")
            ),
            float("nan"),
        )
        gene_metric_rows.append(
            {
                "gene": gene_s,
                "rmse": float(np.sqrt(np.mean((pv - tv) ** 2))),
                "mae": float(np.mean(np.abs(pv - tv))),
                "r2": r2,
                "pearson": pearson,
                "spearman": spearman,
                "n_observed_train": int(rec.get("n_observed", int((~m).sum()))),
                "n_masked": int(m.sum()),
                "winning_prefix_len": int(rec.get("winning_prefix_len", 0)),
                "n_predictors_selected": int(rec.get("n_predictors_selected", 0)),
                "winning_predictors": "|".join(rec.get("winning_predictors", [])),
                "best_oof_rmse": best_oof,
                "selection_seconds": float(rec.get("selection_seconds", 0.0)),
                "final_fit_seconds": float(rec.get("final_fit_seconds", 0.0)),
                "status": str(rec.get("status", "unknown")),
                "n_candidates": len(rec.get("candidates", [])),
                "fallback_count": int(
                    sum(
                        1
                        for e in ga.get("fallback_events", [])
                        if e.get("gene") == gene_s
                    )
                ),
            }
        )

    row = {
        "stage": "imputation",
        "imputer": imputer_name,
        "missing_rate": missing_rate,
        "replicate": replicate,
        "seed": seed,
        "fold": 0,
        "rmse": rmse,
        "mae": mae,
        "n_masked_test_values": n_masked,
        "input_protocol": "target_wise_complete_predictors",
        "evaluation_protocol": "repeated_mask_holdout",
        "svr_coverage": ga.get("svr_coverage"),
        "fallback_rate": ga.get("fallback_rate"),
        "fit_seconds": t_fit,
        "transform_seconds": t_tr,
        "n_rfe_fits_total": ga.get("n_rfe_fits_total"),
        "n_svr_fits_total": ga.get("n_svr_fits_total"),
        "n_predictor_nans_at_impute": n_pred_nan,
        "n_gene_workers": getattr(imp, "_n_gene_workers_", 1),
    }
    detail = {
        "audit": ga,
        "per_gene_metrics": gene_metric_rows,
        "fitted_imputer": imp,
        "eval_mask": eval_mask,
    }
    return pd.DataFrame([row]), detail


def _run_holdout_target_wise_parallel(
    *,
    X_full: pd.DataFrame,
    mask: np.ndarray,
    imputer,
    missing_rate: float,
    replicate: int,
    seed: int | None,
    imputer_name: str,
    checkpoint_root: str | None,
    n_gene_workers: int,
) -> tuple[pd.DataFrame, dict]:
    """Gene-parallel TARGET-WISE holdout (same methodology; genes only)."""
    import json
    from pathlib import Path

    import joblib

    from bcimpute.imputation_original.parallel_genes import (
        default_imputer_kwargs,
        pin_blas_threads,
        run_genes_parallel,
    )
    from bcimpute.imputation_original.types import (
        FitAudit,
        GeneAuditRecord,
        SubsetEvaluation,
    )

    pin_blas_threads(1)
    mask_arr = np.asarray(mask, dtype=bool)
    X_full_arr = X_full.to_numpy(dtype=float)
    if not np.isfinite(X_full_arr).all():
        raise AssertionError("X_full must be fully finite for TARGET-WISE.")

    target_genes = (
        list(imputer.target_genes)
        if getattr(imputer, "target_genes", None) is not None
        else [str(c) for c in X_full.columns]
    )

    ck = Path(checkpoint_root) if checkpoint_root else None
    if ck is not None:
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "models").mkdir(parents=True, exist_ok=True)
        (ck / "genes").mkdir(parents=True, exist_ok=True)

    # Resume: genes already present in serial gene_models.joblib or models/*.joblib
    gene_models: dict = {}
    if ck is not None:
        serial_models = ck / "gene_models.joblib"
        if serial_models.exists():
            gene_models.update(joblib.load(serial_models))
        for p in sorted((ck / "models").glob("*.joblib")):
            gene_models[p.stem] = joblib.load(p)

    pending = [g for g in target_genes if g not in gene_models]
    # Also skip if audit says skipped without model
    if ck is not None:
        still = []
        for g in pending:
            audit_p = ck / "genes" / f"{g}.json"
            if audit_p.exists():
                payload = json.loads(audit_p.read_text(encoding="utf-8"))
                if str(payload.get("status", "")).startswith("skipped"):
                    continue
            still.append(g)
        pending = still

    n_done = len(target_genes) - len(pending)
    print(
        f"[parallel-genes] workers={n_gene_workers} "
        f"done={n_done} pending={len(pending)} "
        f"checkpoint={ck}",
        flush=True,
    )

    kwargs = default_imputer_kwargs()
    # Preserve key hyperparams from the provided imputer
    for key in (
        "validation_strategy",
        "n_splits",
        "random_state",
        "kernel",
        "C",
        "epsilon",
        "use_scaler",
        "min_train_samples",
        "max_candidates",
        "selection_protocol",
        "feature_names",
    ):
        if hasattr(imputer, key):
            kwargs[key] = getattr(imputer, key)

    t0 = time.perf_counter()
    if pending:
        ctx = dict(getattr(imputer, "run_context", None) or {})
        run = run_genes_parallel(
            X=X_full,
            mask=mask_arr,
            genes=pending,
            n_workers=n_gene_workers,
            seed=int(seed if seed is not None else ctx.get("seed", 42)),
            mechanism=str(ctx.get("mechanism", "mcar")),
            missing_rate=float(missing_rate),
            replicate=int(replicate),
            dataset=str(ctx.get("dataset", "metabric")),
            imputer_kwargs=kwargs,
            checkpoint_dir=str(ck) if ck is not None else None,
        )
        failed = [r for r in run.results if not r.get("ok")]
        if failed:
            raise RuntimeError(
                "Parallel gene workers failed: "
                + "; ".join(f"{r['gene']}:{r.get('error')}" for r in failed)
            )
        # Reload models written by workers
        if ck is not None:
            for g in pending:
                mp = ck / "models" / f"{g}.joblib"
                if mp.exists():
                    gene_models[g] = joblib.load(mp)
    t_fit = time.perf_counter() - t0

    # Build a fitted imputer shell for transform + audit
    imp = clone(imputer)
    if getattr(imputer, "target_genes", None) is not None:
        imp.target_genes = list(imputer.target_genes)
    else:
        imp.target_genes = list(target_genes)
    if getattr(imputer, "run_context", None) is not None:
        imp.run_context = dict(imputer.run_context)
    imp.feature_names_in_ = [str(c) for c in X_full.columns]
    imp._gene_models_ = {g: gene_models[g] for g in target_genes if g in gene_models}
    imp._n_gene_workers_ = n_gene_workers
    imp._fallback_values_ = {
        str(c): float(X_full[c].mean()) for c in X_full.columns
    }
    imp._n_svr_imputed_cells_ = 0
    imp._n_target_masked_cells_transform_ = 0
    imp.audit_ = FitAudit(
        method=getattr(imp, "method_name", "OriginalRFECA"),
        validation_strategy=str(kwargs.get("validation_strategy", "kfold")),
        random_state=int(kwargs.get("random_state", 42)),
        selection_protocol=str(kwargs.get("selection_protocol", "leakage_safe")),
        use_scaler=bool(kwargs.get("use_scaler", False)),
        svr_kernel=str(kwargs.get("kernel", "linear")),
        max_candidates=kwargs.get("max_candidates", 49),
    )
    # Restore gene audits
    for g in target_genes:
        rec = None
        if ck is not None:
            ap = ck / "genes" / f"{g}.json"
            # Prefer shared genes/ then worker_* fallback
            if not ap.exists():
                wdir = ck / f"worker_{g}" / "genes" / f"{g}.json"
                if wdir.exists():
                    ap = wdir
            if ap.exists():
                payload = json.loads(ap.read_text(encoding="utf-8"))
                rec = GeneAuditRecord(
                    gene=g,
                    n_observed=int(payload.get("n_observed", 0)),
                    n_missing=int(payload.get("n_missing", 0)),
                    candidates=list(payload.get("candidates", [])),
                    correlation_abs_order=list(payload.get("correlation_abs_order", [])),
                    correlation_values=dict(payload.get("correlation_values", {})),
                    subsets_evaluated=[],
                    winning_prefix_len=int(payload.get("winning_prefix_len", 0)),
                    winning_prefix_genes=list(payload.get("winning_prefix_genes", [])),
                    winning_predictors=list(payload.get("winning_predictors", [])),
                    n_predictors_selected=int(payload.get("n_predictors_selected", 0)),
                    validation_strategy=str(payload.get("validation_strategy", "")),
                    validation_n_splits=payload.get("validation_n_splits"),
                    svr_params=dict(payload.get("svr_params", {})),
                    selection_seconds=float(payload.get("selection_seconds", 0.0)),
                    final_fit_seconds=float(payload.get("final_fit_seconds", 0.0)),
                    selector_kind=str(payload.get("selector_kind", "RFE")),
                    n_prefixes_evaluated=int(payload.get("n_prefixes_evaluated", 0)),
                    n_rfe_fits=int(payload.get("n_rfe_fits", 0)),
                    n_svr_fits=int(payload.get("n_svr_fits", 0)),
                    status=str(payload.get("status", "ok")),
                    message=str(payload.get("message", "")),
                )
                for s in payload.get("subsets_evaluated", []):
                    rec.subsets_evaluated.append(
                        SubsetEvaluation(
                            prefix_len=int(s["prefix_len"]),
                            prefix_genes_final=list(s.get("prefix_genes_final", [])),
                            selected_features_final=list(
                                s.get("selected_features_final", [])
                            ),
                            n_predictors=int(s.get("n_predictors", 0)),
                            rmse=float(s.get("rmse", float("nan"))),
                            n_oof_predictions=int(s.get("n_oof_predictions", 0)),
                        )
                    )
        if rec is not None:
            imp.audit_.genes.append(rec)
            imp.audit_.n_rfe_fits_total += rec.n_rfe_fits
            imp.audit_.n_svr_fits_total += rec.n_svr_fits

    # Merge models into serial gene_models.joblib for future resume
    if ck is not None and gene_models:
        joblib.dump(gene_models, ck / "gene_models.joblib")

    imp.set_target_wise_context(X_full, mask_arr, for_transform=True)
    t1 = time.perf_counter()
    X_imp = imp.transform(X_full)
    t_tr = time.perf_counter() - t1

    return _finalize_holdout_metrics(
        X_full=X_full,
        X_full_arr=X_full_arr,
        mask_arr=mask_arr,
        X_imp=X_imp,
        imp=imp,
        missing_rate=missing_rate,
        replicate=replicate,
        seed=seed,
        imputer_name=imputer_name,
        t_fit=t_fit,
        t_tr=t_tr,
    )
