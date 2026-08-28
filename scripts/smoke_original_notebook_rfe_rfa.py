#!/usr/bin/env python3
"""Smoke: OriginalRFECA/RFACA on a tiny synthetic matrix + gene walkthrough JSON."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcimpute.imputation_original import (  # noqa: E402
    OriginalRFACAImputer,
    OriginalRFECAImputer,
)
from bcimpute.imputation_original.utils import safe_mkdir, write_json  # noqa: E402


def _synth(n: int = 40, p: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    cols = {}
    for j in range(p):
        w = rng.normal(size=2)
        cols[f"G{j}"] = z @ w + rng.normal(scale=0.15, size=n)
    return pd.DataFrame(cols)


def main() -> int:
    out_dir = ROOT / "artifacts" / "original_rfeca_rfaca" / "smoke_notebook_rfe_rfa_20260730"
    safe_mkdir(out_dir)

    rng = np.random.default_rng(0)
    X = _synth(n=40, p=6, seed=0)
    X_miss = X.mask(rng.random(X.shape) < 0.15)

    common = dict(
        validation_strategy="kfold",
        n_splits=3,
        random_state=42,
        min_train_samples=8,
        max_candidates=4,
        record_fold_details=True,
    )

    t0 = time.perf_counter()
    a = OriginalRFECAImputer(**common)
    out_a = a.fit_transform(X_miss)
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = OriginalRFACAImputer(**common)
    out_b = b.fit_transform(X_miss)
    t_b = time.perf_counter() - t0

    # Walkthrough gene: first ok gene with rich audit
    gene = next(g.gene for g in a.audit_.genes if g.status == "ok")
    ga = next(g for g in a.audit_.genes if g.gene == gene)
    gb = next(g for g in b.audit_.genes if g.gene == gene)

    walkthrough = {
        "gene": gene,
        "n_observed": ga.n_observed,
        "n_missing": ga.n_missing,
        "candidate_pool": ga.candidates,
        "correlation_order_full_observed": ga.correlation_abs_order,
        "correlation_values": ga.correlation_values,
        "RFECA": {
            "prefix_rmses": [
                {
                    "prefix_len": s.prefix_len,
                    "prefix": s.prefix_genes_final,
                    "oof_rmse": s.rmse,
                    "n_oof": s.n_oof_predictions,
                    "fold_selected": [
                        {
                            "fold": fd.fold,
                            "corr_order": fd.correlation_order,
                            "prefix": fd.prefix_genes,
                            "selected": fd.selected_features,
                        }
                        for fd in s.fold_details
                    ],
                }
                for s in ga.subsets_evaluated
            ],
            "winning_prefix_len": ga.winning_prefix_len,
            "winning_prefix": ga.winning_prefix_genes,
            "winning_selected_after_RFE": ga.winning_predictors,
            "message": ga.message,
        },
        "RFACA": {
            "winning_prefix_len": gb.winning_prefix_len,
            "winning_prefix": gb.winning_prefix_genes,
            "winning_selected_after_RFA": gb.winning_predictors,
            "message": gb.message,
        },
        "subsets_distinct": ga.winning_predictors != gb.winning_predictors,
        "imputations_allclose": bool(
            np.allclose(out_a.to_numpy(), out_b.to_numpy(), equal_nan=True, atol=1e-10)
        ),
        "timing_seconds": {"OriginalRFECA": t_a, "OriginalRFACA": t_b},
        "n_nan_remaining": {
            "RFECA": int(out_a.isna().sum().sum()),
            "RFACA": int(out_b.isna().sum().sum()),
        },
    }
    write_json(out_dir / "smoke_walkthrough.json", walkthrough)
    write_json(
        out_dir / "smoke_summary.json",
        {
            "shape": list(X_miss.shape),
            "max_candidates": 4,
            "n_splits": 3,
            "rfeca_ok_genes": sum(1 for g in a.audit_.genes if g.status == "ok"),
            "rfaca_ok_genes": sum(1 for g in b.audit_.genes if g.status == "ok"),
            "timing_seconds": walkthrough["timing_seconds"],
            "subsets_distinct_on_walkthrough_gene": walkthrough["subsets_distinct"],
        },
    )
    print(f"Wrote {out_dir}")
    print(
        f"gene={gene} RFECA={ga.winning_predictors} RFACA={gb.winning_predictors} "
        f"distinct={walkthrough['subsets_distinct']} t_a={t_a:.2f}s t_b={t_b:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
