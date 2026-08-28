#!/usr/bin/env python3
"""
max_candidates sensitivity for Original RFECA/RFACA (imputation only).

- Shared MCAR mask across all configs (METABRIC, rate=0.2, 1 rep, seed fixed).
- Labels: RFECA_maxcand{K} / RFACA_maxcand{K} (not bare Original*).
- Writes under artifacts/original_rfeca_rfaca/ only (never overwrites legacy).

Usage:
  python experiments/run_maxcand_sensitivity.py --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.config import ARTIFACT_ROOT, PAM50_GENES  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.evaluation import FoldAudit, run_imputation_cv, summarize_imputation  # noqa: E402
from bcimpute.imputation_original import (  # noqa: E402
    OriginalRFACAImputer,
    OriginalRFECAImputer,
)
from bcimpute.imputation_original.utils import safe_mkdir, write_json  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402

MAXCAND_GRID = [5, 10, 20, 49]
NAMESPACE = ARTIFACT_ROOT / "original_rfeca_rfaca"


def _label(kind: str, k: int) -> str:
    return f"{kind}_maxcand{k}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--grid",
        type=int,
        nargs="+",
        default=MAXCAND_GRID,
        help="max_candidates values to sweep",
    )
    args = parser.parse_args()

    if not args.confirm:
        print("Refusing to start without --confirm.")
        print(f"Would sweep max_candidates={args.grid} on METABRIC MCAR rate={args.rate}")
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = safe_mkdir(
        NAMESPACE / f"metabric_mcar_maxcand_sensitivity_{run_id}"
    )
    print(f"Output: {out_dir}", flush=True)

    cohort = load_cohort("metabric")
    obs = cohort.originally_observed_mask
    assert obs is not None

    missing_sets = generate_missingness_sets(
        cohort.X,
        missing_rates=[float(args.rate)],
        n_repetitions=1,
        base_seed=int(args.seed),
        originally_observed_mask=obs,
        target_cell_policy="originally_observed_only",
        mechanism="mcar",
        seed_scheme="v2",
    )
    item = missing_sets[float(args.rate)][0]
    # Persist shared mask once
    np.save(out_dir / "shared_mask.npy", item.mask)
    write_json(
        out_dir / "shared_mask_meta.json",
        {
            "missing_rate": item.missing_rate,
            "replicate": item.replicate,
            "seed": item.seed,
            "mechanism": "mcar",
            "seed_scheme": "v2",
            "n_masked": int(item.mask.sum()),
            "cohort": "metabric",
            "n_samples": cohort.n_samples,
            "n_genes": cohort.n_features,
        },
    )

    meta = {
        "run_id": run_id,
        "grid": list(args.grid),
        "n_splits": int(args.n_splits),
        "validation": "kfold",
        "original_rfeca_n_splits": 5,
        "note": "Shared mask; imputation-only; labels RFECA/RFACA_maxcandK",
    }
    write_json(out_dir / "config_snapshot.json", meta)

    all_imp_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    imputed_vectors: dict[str, np.ndarray] = {}
    t_wall0 = time.perf_counter()

    for k in args.grid:
        k = int(k)
        # Cap at n_genes-1
        k_eff = min(k, len(PAM50_GENES) - 1)
        imputers = {
            _label("RFECA", k): OriginalRFECAImputer(
                validation_strategy="kfold",
                n_splits=5,
                random_state=int(args.seed),
                max_candidates=k_eff,
                feature_names=list(PAM50_GENES),
            ),
            _label("RFACA", k): OriginalRFACAImputer(
                validation_strategy="kfold",
                n_splits=5,
                random_state=int(args.seed),
                max_candidates=k_eff,
                feature_names=list(PAM50_GENES),
            ),
        }
        print(f"\n=== max_candidates={k} (eff={k_eff}) ===", flush=True)
        t0 = time.perf_counter()
        audit = FoldAudit()
        imp_df = run_imputation_cv(
            X_full=cohort.X,
            X_missing=item.X_missing,
            mask=item.mask,
            y=cohort.y,
            imputers=imputers,
            n_splits=int(args.n_splits),
            random_state=int(args.seed),
            missing_rate=float(args.rate),
            replicate=0,
            audit=audit,
            originally_observed_mask=obs.to_numpy(),
            target_cell_policy="originally_observed_only",
        )
        imp_df["max_candidates"] = k
        imp_df["max_candidates_eff"] = k_eff
        imp_df["seed"] = item.seed
        all_imp_rows.append(imp_df)

        # Gene audit on full masked matrix (same mask)
        for name, base_imp in imputers.items():
            audit_imp = clone(base_imp)
            t_g0 = time.perf_counter()
            audit_imp.fit(item.X_missing)
            # Also transform once to capture imputed values on masked cells
            X_imp = audit_imp.transform(item.X_missing)
            t_g = time.perf_counter() - t_g0
            genes = audit_imp.audit_.genes
            ok = [g for g in genes if g.status == "ok"]
            skipped = [g for g in genes if g.status != "ok"]
            ks = [g.n_predictors_selected for g in ok]
            at_cap = sum(1 for g in ok if g.n_predictors_selected == k_eff)
            n_cand_means = [len(g.candidates) for g in genes]
            # Imputed values only on artificially masked cells
            flat_mask = item.mask.ravel()
            vals = X_imp.to_numpy(dtype=float).ravel()[flat_mask]
            imputed_vectors[name] = vals

            write_json(
                out_dir / f"gene_audit_{name}.json",
                {
                    "imputer": name,
                    "max_candidates": k,
                    "max_candidates_eff": k_eff,
                    "fit_transform_seconds": t_g,
                    "audit": audit_imp.get_audit_dict(),
                },
            )

            # Per-method RMSE from CV summary later; store gene stats now
            summary_rows.append(
                {
                    "imputer": name,
                    "kind": "RFECA" if name.startswith("RFECA") else "RFACA",
                    "max_candidates": k,
                    "max_candidates_eff": k_eff,
                    "n_genes_total": len(genes),
                    "n_genes_ok": len(ok),
                    "n_genes_skipped": len(skipped),
                    "skip_statuses": {
                        s: sum(1 for g in skipped if g.status == s)
                        for s in sorted({g.status for g in skipped})
                    },
                    "mean_n_candidates": float(np.mean(n_cand_means)) if n_cand_means else float("nan"),
                    "mean_winner_k": float(np.mean(ks)) if ks else float("nan"),
                    "median_winner_k": float(np.median(ks)) if ks else float("nan"),
                    "winner_k_hist": {str(i): int(sum(1 for x in ks if x == i)) for i in range(1, k_eff + 1)},
                    "n_winner_at_cap": int(at_cap),
                    "prop_winner_at_cap": float(at_cap / len(ok)) if ok else float("nan"),
                    "gene_audit_fit_transform_s": t_g,
                }
            )
            print(
                f"  {name}: ok={len(ok)} skip={len(skipped)} "
                f"mean_k={np.mean(ks) if ks else float('nan'):.2f} "
                f"at_cap={at_cap}/{len(ok)} ({100*at_cap/max(len(ok),1):.1f}%) "
                f"audit_s={t_g:.1f}",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        print(f"  config wall_s={elapsed:.1f}", flush=True)

    imp_raw = pd.concat(all_imp_rows, ignore_index=True)
    imp_raw.to_csv(out_dir / "exp1_imputation_raw.csv", index=False)
    imp_sum = summarize_imputation(imp_raw)
    imp_sum.to_csv(out_dir / "exp1_imputation_summary.csv", index=False)

    # Attach CV RMSE to summary_rows
    rmse_map = {
        (r["imputer"],): (r["rmse_mean"], r["rmse_std"])
        for _, r in imp_sum.iterrows()
    }
    for row in summary_rows:
        key = (row["imputer"],)
        if key in rmse_map:
            row["rmse_mean"], row["rmse_std"] = rmse_map[key]

    # Pairwise correlation of imputed masked cells across configs (same kind)
    corr_rows = []
    names = list(imputed_vectors.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va, vb = imputed_vectors[a], imputed_vectors[b]
            if va.size == 0:
                continue
            c = float(np.corrcoef(va, vb)[0, 1])
            corr_rows.append({"imputer_a": a, "imputer_b": b, "pearson_imputed_cells": c})

    pd.DataFrame(summary_rows).to_csv(out_dir / "maxcand_sensitivity_summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out_dir / "imputed_value_correlations.csv", index=False)
    write_json(
        out_dir / "maxcand_sensitivity_report.json",
        {
            "status": "PASS",
            "wall_clock_seconds": time.perf_counter() - t_wall0,
            "output_dir": str(out_dir),
            "shared_mask_seed": item.seed,
            "summary": summary_rows,
            "imputed_correlations": corr_rows,
        },
    )
    print(f"\nDONE wall_s={time.perf_counter()-t_wall0:.1f}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
