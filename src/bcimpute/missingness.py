"""Controlled missingness simulation (MCAR / MAR) without mutating the source matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MissingnessResult:
    X_missing: pd.DataFrame
    mask: np.ndarray  # True where value was artificially removed
    missing_rate: float
    replicate: int
    seed: int
    target_cell_policy: str = "originally_observed_only"
    n_eligible_cells: int = 0
    n_artificially_masked_cells: int = 0
    n_legacy_imputed_cells_in_cohort: int = 0
    missingness_mechanism: str = "mcar"


def _resolve_eligible(
    X_full: pd.DataFrame,
    originally_observed_mask: np.ndarray | pd.DataFrame | None,
    target_cell_policy: str,
) -> tuple[np.ndarray, int]:
    shape = X_full.shape
    if originally_observed_mask is None:
        eligible = np.ones(shape, dtype=bool)
        n_legacy = 0
    else:
        if isinstance(originally_observed_mask, pd.DataFrame):
            eligible = originally_observed_mask.reindex(
                index=X_full.index, columns=X_full.columns
            ).to_numpy(dtype=bool)
        else:
            eligible = np.asarray(originally_observed_mask, dtype=bool)
        if eligible.shape != shape:
            raise ValueError(
                f"originally_observed_mask shape {eligible.shape} != X shape {shape}"
            )
        n_legacy = int((~eligible).sum())

    if target_cell_policy == "all_complete_cells":
        eligible = np.ones(shape, dtype=bool)

    return eligible, n_legacy


def _apply_mask(
    X_full: pd.DataFrame,
    mask: np.ndarray,
    *,
    target_cell_policy: str,
    n_eligible: int,
    n_legacy: int,
    mechanism: str = "mcar",
) -> tuple[pd.DataFrame, dict]:
    values = X_full.to_numpy(dtype=float, copy=True)
    values[mask] = np.nan
    X_missing = pd.DataFrame(values, index=X_full.index, columns=X_full.columns)
    meta = {
        "target_cell_policy": target_cell_policy,
        "n_eligible_cells": n_eligible,
        "n_artificially_masked_cells": int(mask.sum()),
        "n_legacy_imputed_cells_in_cohort": n_legacy,
        "missingness_mechanism": mechanism,
    }
    return X_missing, meta


def add_missing_values_mcar(
    X: pd.DataFrame,
    missing_rate: float,
    seed: int,
    *,
    originally_observed_mask: np.ndarray | pd.DataFrame | None = None,
    target_cell_policy: str = "originally_observed_only",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Create a *copy* of X with MCAR missingness.

    - Never modifies X in place.
    - Mask applies only to feature cells (X columns), never to labels.
    - Default policy restricts artificial masking to originally observed cells.
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
    if target_cell_policy not in {"originally_observed_only", "all_complete_cells"}:
        raise ValueError(f"Unknown target_cell_policy: {target_cell_policy}")

    X_full = X.copy()
    rng = np.random.default_rng(seed)
    eligible, n_legacy = _resolve_eligible(
        X_full, originally_observed_mask, target_cell_policy
    )
    n_eligible = int(eligible.sum())
    mask = np.zeros(X_full.shape, dtype=bool)

    if missing_rate == 0.0 or n_eligible == 0:
        n_masked = 0
    else:
        n_masked = int(round(missing_rate * n_eligible))
        n_masked = max(0, min(n_masked, n_eligible))
        flat_idx = np.flatnonzero(eligible.ravel())
        chosen = rng.choice(flat_idx, size=n_masked, replace=False)
        mask.ravel()[chosen] = True

    if target_cell_policy == "originally_observed_only" and originally_observed_mask is not None:
        if np.any(mask & ~eligible):
            raise AssertionError(
                "Artificial mask targeted non-observed (legacy-imputed) cells."
            )

    X_missing, meta = _apply_mask(
        X_full,
        mask,
        target_cell_policy=target_cell_policy,
        n_eligible=n_eligible,
        n_legacy=n_legacy,
        mechanism="mcar",
    )
    return X_missing, mask, meta


def add_missing_values_mar(
    X: pd.DataFrame,
    missing_rate: float,
    seed: int,
    *,
    originally_observed_mask: np.ndarray | pd.DataFrame | None = None,
    target_cell_policy: str = "originally_observed_only",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Create a *copy* of X with MAR missingness (exact cell count).

    Mechanism (defensible, label-free):
      For eligible cell (i, j), propensity depends on the *other* genes in sample i
      (row mean excluding column j), not on X[i, j] itself and not on PAM50 labels.
      Stochastic Gumbel noise breaks ties across replicates; the top-n eligible cells
      by score are masked so the realized rate matches MCAR's exact-count contract.

    This is MAR (not MNAR): P(R_ij=1 | X) = f(X_i, -j).
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
    if target_cell_policy not in {"originally_observed_only", "all_complete_cells"}:
        raise ValueError(f"Unknown target_cell_policy: {target_cell_policy}")

    X_full = X.copy()
    rng = np.random.default_rng(seed)
    eligible, n_legacy = _resolve_eligible(
        X_full, originally_observed_mask, target_cell_policy
    )
    n_eligible = int(eligible.sum())
    mask = np.zeros(X_full.shape, dtype=bool)

    if missing_rate == 0.0 or n_eligible == 0:
        X_missing, meta = _apply_mask(
            X_full,
            mask,
            target_cell_policy=target_cell_policy,
            n_eligible=n_eligible,
            n_legacy=n_legacy,
            mechanism="mar",
        )
        meta["mar_predictor"] = "leave_one_out_row_mean_abs_z_plus_gumbel"
        return X_missing, mask, meta

    n_masked = int(round(missing_rate * n_eligible))
    n_masked = max(0, min(n_masked, n_eligible))

    values = X_full.to_numpy(dtype=float)
    n_samples, n_features = values.shape
    # Row means of all genes; leave-one-out mean = (n*mu - x_j) / (n-1)
    row_sum = values.sum(axis=1, keepdims=True)
    if n_features < 2:
        raise ValueError("MAR requires at least 2 feature columns.")
    leave_one_out = (row_sum - values) / float(n_features - 1)

    # Standardize predictors per column for stable scores across gene scales.
    col_mu = leave_one_out.mean(axis=0, keepdims=True)
    col_sd = leave_one_out.std(axis=0, keepdims=True)
    col_sd = np.where(col_sd < 1e-12, 1.0, col_sd)
    z = (leave_one_out - col_mu) / col_sd

    # Higher |z| → more extreme co-expression context → higher missing propensity.
    # Gumbel noise → softmax-like stochastic ranking (exact top-n after noise).
    noise = rng.gumbel(loc=0.0, scale=1.0, size=z.shape)
    scores = np.abs(z) + noise
    scores = np.where(eligible, scores, -np.inf)

    flat_scores = scores.ravel()
    # argpartition for top-n among finite scores
    if n_masked == n_eligible:
        chosen = np.flatnonzero(eligible.ravel())
    else:
        # Only consider eligible positions
        eligible_flat = np.flatnonzero(np.isfinite(flat_scores))
        top_local = np.argpartition(-flat_scores[eligible_flat], n_masked - 1)[:n_masked]
        chosen = eligible_flat[top_local]

    mask.ravel()[chosen] = True

    if target_cell_policy == "originally_observed_only" and originally_observed_mask is not None:
        if np.any(mask & ~eligible):
            raise AssertionError(
                "Artificial mask targeted non-observed (legacy-imputed) cells."
            )
    if int(mask.sum()) != n_masked:
        raise AssertionError(f"MAR mask sum {mask.sum()} != expected {n_masked}")

    X_missing, meta = _apply_mask(
        X_full,
        mask,
        target_cell_policy=target_cell_policy,
        n_eligible=n_eligible,
        n_legacy=n_legacy,
        mechanism="mar",
    )
    meta["mar_predictor"] = "leave_one_out_row_mean_abs_z_plus_gumbel"
    return X_missing, mask, meta


def missingness_seed(
    base_seed: int,
    missing_rate: float,
    replicate: int,
    *,
    mechanism: str = "mcar",
    scheme: str = "v2",
) -> int:
    """
    Deterministic seed for a (mechanism, rate, replicate) triple.

    Schemes
    -------
    v2 (default)
        Collision-free for the study grid (rates in {0,0.05,0.1,0.2,0.3}, reps 0..9+):
        ``base + mech_offset + replicate * 1_000_003 + round(rate * 1_000_000)``.
    legacy
        Pre-2026-07 formula ``base + 1000 + replicate + round(rate*100) + mech_offset``.
        Collides across some (rate, rep) pairs (e.g. 5%@rep5 ↔ 10%@rep0). Keep only
        to reproduce masks from frozen artifacts under that scheme.
    """
    mechanism = mechanism.lower().strip()
    scheme = scheme.lower().strip()
    mech_offset = 0 if mechanism == "mcar" else 17_000
    if scheme == "legacy":
        return int(
            base_seed + 1000 + replicate + int(round(missing_rate * 100)) + mech_offset
        )
    if scheme == "v2":
        rate_key = int(round(float(missing_rate) * 1_000_000))
        return int(base_seed + mech_offset + int(replicate) * 1_000_003 + rate_key)
    raise ValueError(f"Unknown missingness seed scheme: {scheme!r}")


def generate_missingness_sets(
    X: pd.DataFrame,
    missing_rates: list[float],
    n_repetitions: int,
    base_seed: int,
    *,
    originally_observed_mask: np.ndarray | pd.DataFrame | None = None,
    target_cell_policy: str = "originally_observed_only",
    mechanism: str = "mcar",
    seed_scheme: str = "v2",
) -> dict[float, list[MissingnessResult]]:
    """Generate identical missingness scenarios reusable across all imputers."""
    mechanism = mechanism.lower().strip()
    if mechanism not in {"mcar", "mar"}:
        raise ValueError(f"Unknown missingness mechanism: {mechanism!r}")

    adder = add_missing_values_mcar if mechanism == "mcar" else add_missing_values_mar
    out: dict[float, list[MissingnessResult]] = {}
    for rate in missing_rates:
        out[rate] = []
        for rep in range(n_repetitions):
            seed = missingness_seed(
                base_seed,
                rate,
                rep,
                mechanism=mechanism,
                scheme=seed_scheme,
            )
            X_missing, mask, meta = adder(
                X,
                rate,
                seed,
                originally_observed_mask=originally_observed_mask,
                target_cell_policy=target_cell_policy,
            )
            # Expected count check
            if rate > 0 and meta["n_eligible_cells"] > 0:
                expected = int(round(rate * meta["n_eligible_cells"]))
                expected = max(0, min(expected, meta["n_eligible_cells"]))
                if mask.sum() != expected:
                    raise AssertionError(
                        f"mask sum {mask.sum()} != expected {expected} "
                        f"for mechanism={mechanism} rate={rate} rep={rep}"
                    )
            out[rate].append(
                MissingnessResult(
                    X_missing=X_missing,
                    mask=mask,
                    missing_rate=rate,
                    replicate=rep,
                    seed=seed,
                    target_cell_policy=meta["target_cell_policy"],
                    n_eligible_cells=meta["n_eligible_cells"],
                    n_artificially_masked_cells=meta["n_artificially_masked_cells"],
                    n_legacy_imputed_cells_in_cohort=meta[
                        "n_legacy_imputed_cells_in_cohort"
                    ],
                    missingness_mechanism=mechanism,
                )
            )
    return out
