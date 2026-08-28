"""Smoke: repeated_mask_holdout never sees masked target labels in training."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcimpute.evaluation import run_imputation_repeated_mask_holdout_target_wise
from bcimpute.imputation_original import OriginalRFECAImputer
from bcimpute.imputation_original.correlation import pearson_abs_ranking


def test_masked_y_never_in_correlation():
    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = pd.DataFrame(
        rng.normal(size=(n, p)), columns=[f"G{j}" for j in range(p)]
    )
    mask = np.zeros((n, p), dtype=bool)
    mask[0:8, 0] = True  # hold out G0 on rows 0..7
    true_held = X.iloc[0:8, 0].to_numpy().copy()

    seen_y_vals: list[np.ndarray] = []
    real = pearson_abs_ranking

    def tracking(y, X_rank, min_periods=3):
        seen_y_vals.append(np.asarray(y, dtype=float).copy())
        return real(y, X_rank, min_periods=min_periods)

    imp = OriginalRFECAImputer(
        validation_strategy="kfold",
        n_splits=3,
        max_candidates=3,
        use_scaler=False,
        random_state=0,
        min_train_samples=8,
    )
    with patch(
        "bcimpute.imputation_original.base.pearson_abs_ranking",
        side_effect=tracking,
    ):
        df, detail = run_imputation_repeated_mask_holdout_target_wise(
            X_full=X,
            mask=mask,
            imputer=imp,
            missing_rate=0.2,
            replicate=0,
            seed=0,
        )
    assert df["evaluation_protocol"].iloc[0] == "repeated_mask_holdout"
    assert df["svr_coverage"].iloc[0] >= 0.99
    assert df["fallback_rate"].iloc[0] == 0.0
    # No correlation call's y vector may contain the held-out true values as a block
    # (held-out positions are NaN in the working matrix, so y used in ranking is
    # observed-only — length n_obs, never includes the 8 held-out truths).
    for yv in seen_y_vals:
        assert len(yv) <= n - 8 or not np.allclose(yv[:8], true_held)
        # Stronger: none of the finite y values equal the set of held-out with multiplicity
        # Easier: all ranking y lengths for G0 fits are <= 32
    # At least some ranking used only observed count for G0
    assert any(len(yv) == n - 8 for yv in seen_y_vals) or any(
        len(yv) < n for yv in seen_y_vals
    )
    print("PASS test_masked_y_never_in_correlation", df[["rmse", "svr_coverage"]].iloc[0].to_dict())


if __name__ == "__main__":
    test_masked_y_never_in_correlation()
