#!/usr/bin/env python3
"""
Baseline per-gene RMSE/MAE collection (imputation-only).

Scenario (frozen baseline protocol):
  - Methods: SimpleMean, KNN(k=5,dist), MissForest
  - Mechanisms: MCAR + MAR
  - Rate: 20%
  - Replicates: 0–4
  - Seeds: legacy (same as metabric_full_* 2026-07 artifacts)
  - CV: StratifiedKFold(5, shuffle=True, random_state=42)
  - No classification

Outputs under artifacts/baseline_gene_metrics_<timestamp>/:
  - config_snapshot.json
  - seed_audit.csv
  - masks/<mech>/rate_0.20/rep_<r>/mask.npz
  - per_gene_raw.csv          (fold-level gene metrics)
  - per_gene_summary.csv      (mean over folds×reps)
  - fold_aggregate.csv        (global RMSE/MAE for sanity vs freeze)
  - progress.jsonl
  - DONE.json / REPORT.md

Example:
  python experiments/run_baseline_gene_metrics.py --dry-run
  python experiments/run_baseline_gene_metrics.py --confirm --workers 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.config import PAM50_GENES  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.imputers import make_missforest  # noqa: E402
from bcimpute.missingness import (  # noqa: E402
    add_missing_values_mar,
    add_missing_values_mcar,
    missingness_seed,
)
from sklearn.impute import KNNImputer, SimpleImputer  # noqa: E402

# Frozen baseline seeds (legacy scheme, base=42)
EXPECTED_SEEDS = {
    ("mcar", 0.2, 0): 1062,
    ("mcar", 0.2, 1): 1063,
    ("mcar", 0.2, 2): 1064,
    ("mcar", 0.2, 3): 1065,
    ("mcar", 0.2, 4): 1066,
    ("mar", 0.2, 0): 18062,
    ("mar", 0.2, 1): 18063,
    ("mar", 0.2, 2): 18064,
    ("mar", 0.2, 3): 18065,
    ("mar", 0.2, 4): 18066,
}

METHOD_NAMES = ["SimpleMean", "KNN(k=5,dist)", "MissForest"]
DISPLAY = {
    "SimpleMean": "Mean",
    "KNN(k=5,dist)": "KNN",
    "MissForest": "MissForest",
}


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(bool)).tobytes()).hexdigest()[
        :16
    ]


def _build_imputers(*, missforest_n_jobs: int, random_state: int = 42) -> dict:
    return {
        "SimpleMean": SimpleImputer(strategy="mean"),
        "KNN(k=5,dist)": KNNImputer(n_neighbors=5, weights="distance"),
        "MissForest": make_missforest(
            n_estimators=20,
            max_iter=5,
            random_state=random_state,
            n_jobs=missforest_n_jobs,
        ),
    }


def _audit_seeds(base_seed: int = 42) -> pd.DataFrame:
    rows = []
    for (mech, rate, rep), expected in EXPECTED_SEEDS.items():
        got = missingness_seed(
            base_seed, rate, rep, mechanism=mech, scheme="legacy"
        )
        rows.append(
            {
                "mechanism": mech,
                "missing_rate": rate,
                "replicate": rep,
                "seed_expected": expected,
                "seed_computed": got,
                "match": got == expected,
            }
        )
    return pd.DataFrame(rows)


def _run_slot(payload: dict) -> dict:
    """Worker: one (mechanism, rate, replicate) slot."""
    # Limit BLAS threads inside worker when parallelizing slots
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(k, "1")

    mechanism = payload["mechanism"]
    rate = float(payload["missing_rate"])
    replicate = int(payload["replicate"])
    seed = int(payload["seed"])
    out_dir = Path(payload["out_dir"])
    n_splits = int(payload["n_splits"])
    cv_random_state = int(payload["cv_random_state"])
    missforest_n_jobs = int(payload["missforest_n_jobs"])

    t0 = time.perf_counter()
    cohort = load_cohort("metabric")
    X = cohort.X[PAM50_GENES].astype(float)
    y = cohort.y
    obs = cohort.originally_observed_mask
    if obs is not None and hasattr(obs, "reindex"):
        obs = obs.reindex(index=X.index, columns=X.columns)
        obs_arr = np.asarray(obs, dtype=bool)
    elif obs is not None:
        obs_arr = np.asarray(obs, dtype=bool)
    else:
        obs_arr = np.ones(X.shape, dtype=bool)

    adder = add_missing_values_mcar if mechanism == "mcar" else add_missing_values_mar
    X_missing, mask, meta = adder(
        X,
        rate,
        seed,
        originally_observed_mask=obs_arr,
        target_cell_policy="originally_observed_only",
    )
    mask_arr = np.asarray(mask, dtype=bool)
    mhash = _mask_hash(mask_arr)

    mask_dir = out_dir / "masks" / mechanism / f"rate_{rate:.2f}" / f"rep_{replicate}"
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_dir / "mask.npz",
        mask=mask_arr,
        seed=np.array([seed]),
        mask_hash=np.array([mhash]),
        mechanism=np.array([mechanism]),
        missing_rate=np.array([rate]),
        replicate=np.array([replicate]),
    )
    (mask_dir / "meta.json").write_text(
        json.dumps(
            {
                "mechanism": mechanism,
                "missing_rate": rate,
                "replicate": replicate,
                "seed": seed,
                "mask_hash": mhash,
                "n_masked": int(mask_arr.sum()),
                "n_eligible": int(meta["n_eligible_cells"]),
                "mar_predictor": meta.get("mar_predictor"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    imputers = _build_imputers(missforest_n_jobs=missforest_n_jobs)
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=cv_random_state
    )
    X_full_arr = X.to_numpy(dtype=float)
    genes = list(X.columns)

    gene_rows: list[dict] = []
    fold_rows: list[dict] = []

    for imputer_name, imputer in imputers.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_missing, y), start=1):
            imp = clone(imputer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp.fit(X_missing.iloc[train_idx])
                X_test_imp = imp.transform(X_missing.iloc[test_idx])
            X_test_imp = np.asarray(X_test_imp, dtype=float)

            m_test = mask_arr[test_idx]
            obs_test = obs_arr[test_idx]
            metric_mask = m_test & obs_test

            true_block = X_full_arr[test_idx]
            # Global fold metrics (sanity)
            tv = true_block[metric_mask]
            pv = X_test_imp[metric_mask]
            if tv.size == 0:
                rmse = float("nan")
                mae = float("nan")
            else:
                diff = pv - tv
                rmse = float(np.sqrt(np.mean(diff**2)))
                mae = float(np.mean(np.abs(diff)))
            fold_rows.append(
                {
                    "mechanism": mechanism,
                    "missing_rate": rate,
                    "replicate": replicate,
                    "seed": seed,
                    "mask_hash": mhash,
                    "imputer": imputer_name,
                    "method": DISPLAY[imputer_name],
                    "fold": fold,
                    "rmse": rmse,
                    "mae": mae,
                    "n_masked_test": int(metric_mask.sum()),
                }
            )

            # Per-gene on test fold masked cells
            for j, gene in enumerate(genes):
                gmask = metric_mask[:, j]
                if not np.any(gmask):
                    continue
                g_true = true_block[gmask, j]
                g_pred = X_test_imp[gmask, j]
                gdiff = g_pred - g_true
                gene_rows.append(
                    {
                        "mechanism": mechanism,
                        "missing_rate": rate,
                        "replicate": replicate,
                        "seed": seed,
                        "mask_hash": mhash,
                        "imputer": imputer_name,
                        "method": DISPLAY[imputer_name],
                        "fold": fold,
                        "gene": gene,
                        "rmse": float(np.sqrt(np.mean(gdiff**2))),
                        "mae": float(np.mean(np.abs(gdiff))),
                        "n_masked": int(gmask.sum()),
                    }
                )

    elapsed = time.perf_counter() - t0
    slot_dir = out_dir / "slots" / mechanism / f"rate_{rate:.2f}" / f"rep_{replicate}"
    slot_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gene_rows).to_csv(slot_dir / "per_gene_raw.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(slot_dir / "fold_aggregate.csv", index=False)
    done = {
        "mechanism": mechanism,
        "missing_rate": rate,
        "replicate": replicate,
        "seed": seed,
        "mask_hash": mhash,
        "wall_seconds": elapsed,
        "n_gene_rows": len(gene_rows),
        "n_fold_rows": len(fold_rows),
        "status": "ok",
    }
    (slot_dir / "DONE.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to launch the collection run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit seeds + write plan only; do not impute.",
    )
    parser.add_argument("--workers", type=int, default=5, help="Parallel slots.")
    parser.add_argument(
        "--missforest-n-jobs",
        type=int,
        default=2,
        help="Inner MissForest jobs (keep low when workers>1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default: artifacts/baseline_gene_metrics_<ts>).",
    )
    args = parser.parse_args()

    seed_df = _audit_seeds()
    if not bool(seed_df["match"].all()):
        print("SEED AUDIT FAILED:")
        print(seed_df.to_string(index=False))
        return 2

    print("Seed audit PASS (legacy scheme matches frozen metabric_full_*).")
    print(seed_df.to_string(index=False))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "artifacts" / f"baseline_gene_metrics_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out_dir / "seed_audit.csv", index=False)

    config = {
        "cohort": "metabric",
        "methods": METHOD_NAMES,
        "mechanisms": ["mcar", "mar"],
        "missing_rate": 0.2,
        "replicates": [0, 1, 2, 3, 4],
        "seed_scheme": "legacy",
        "base_seed": 42,
        "n_splits": 5,
        "cv_random_state": 42,
        "missforest_n_estimators": 20,
        "missforest_max_iter": 5,
        "missforest_n_jobs": args.missforest_n_jobs,
        "workers": args.workers,
        "classification": False,
        "expected_seeds": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in EXPECTED_SEEDS.items()},
        "source_artifacts": [
            "artifacts/metabric_full_20260724_185916",
            "artifacts/metabric_full_mar_20260725_062517",
        ],
        "protocol_note": (
            "Same CV shared-mask baseline protocol as frozen full runs; "
            "gene-level metrics newly exported."
        ),
    }
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (out_dir / "PLAN.md").write_text(
        "\n".join(
            [
                "# Baseline gene-metrics collection plan",
                "",
                f"- Output: `{out_dir}`",
                "- Methods: Mean, KNN, MissForest",
                "- MCAR 20% + MAR 20%, reps 0–4",
                "- Seeds: legacy (audited against frozen configs)",
                "- Imputation-only; StratifiedKFold(5, random_state=42)",
                f"- Workers: {args.workers}; MissForest n_jobs: {args.missforest_n_jobs}",
                "",
                "Launch:",
                f"```bash",
                f"python experiments/run_baseline_gene_metrics.py --confirm --out \"{out_dir}\" --workers {args.workers}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.dry_run or not args.confirm:
        print(f"Prepared at: {out_dir}")
        if not args.confirm:
            print("Pass --confirm to launch (or --dry-run for plan-only).")
        return 0

    payloads = []
    for mech in ["mcar", "mar"]:
        for rep in range(5):
            payloads.append(
                {
                    "mechanism": mech,
                    "missing_rate": 0.2,
                    "replicate": rep,
                    "seed": EXPECTED_SEEDS[(mech, 0.2, rep)],
                    "out_dir": str(out_dir),
                    "n_splits": 5,
                    "cv_random_state": 42,
                    "missforest_n_jobs": args.missforest_n_jobs,
                }
            )

    progress_path = out_dir / "progress.jsonl"
    t_wall0 = time.perf_counter()
    results = []
    workers = max(1, int(args.workers))

    print(f"Launching {len(payloads)} slots with workers={workers} -> {out_dir}")
    if workers == 1:
        for i, p in enumerate(payloads, start=1):
            print(f"[{i}/{len(payloads)}] {p['mechanism']} rep={p['replicate']} ...")
            done = _run_slot(p)
            results.append(done)
            with progress_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(done) + "\n")
            print(
                f"  done seed={done['seed']} hash={done['mask_hash']} "
                f"wall={done['wall_seconds']:.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_slot, p): p for p in payloads}
            for i, fut in enumerate(as_completed(futs), start=1):
                p = futs[fut]
                done = fut.result()
                results.append(done)
                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(done) + "\n")
                print(
                    f"[{i}/{len(payloads)}] {done['mechanism']} rep={done['replicate']} "
                    f"seed={done['seed']} hash={done['mask_hash']} "
                    f"wall={done['wall_seconds']:.1f}s"
                )

    # Aggregate
    gene_parts = sorted(out_dir.glob("slots/*/*/*/per_gene_raw.csv"))
    fold_parts = sorted(out_dir.glob("slots/*/*/*/fold_aggregate.csv"))
    gene_raw = pd.concat([pd.read_csv(p) for p in gene_parts], ignore_index=True)
    fold_agg = pd.concat([pd.read_csv(p) for p in fold_parts], ignore_index=True)
    gene_raw.to_csv(out_dir / "per_gene_raw.csv", index=False)
    fold_agg.to_csv(out_dir / "fold_aggregate.csv", index=False)

    # Mean over folds within rep, then mean over reps
    fold_mean = (
        gene_raw.groupby(
            ["mechanism", "missing_rate", "replicate", "method", "gene"],
            as_index=False,
        )
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), n_masked=("n_masked", "sum"))
    )
    summary = (
        fold_mean.groupby(["mechanism", "missing_rate", "method", "gene"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_reps=("replicate", "nunique"),
        )
        .sort_values(["mechanism", "method", "rmse_mean"], ascending=[True, True, False])
    )
    summary.to_csv(out_dir / "per_gene_summary.csv", index=False)

    # Wide heatmap-ready table (mean RMSE)
    wide_parts = []
    for mech in ["mcar", "mar"]:
        sub = summary[summary.mechanism == mech]
        piv = sub.pivot_table(
            index="gene", columns="method", values="rmse_mean", aggfunc="mean"
        )
        piv = piv.reindex(columns=["Mean", "KNN", "MissForest"])
        piv.insert(0, "mechanism", mech.upper())
        piv.insert(1, "missing_rate", 0.2)
        wide_parts.append(piv.reset_index())
    wide = pd.concat(wide_parts, ignore_index=True)
    wide.to_csv(out_dir / "gene_method_rmse_wide.csv", index=False)

    # Sanity: compare fold-level global RMSE means to freeze summaries
    sanity = (
        fold_agg.groupby(["mechanism", "method"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), n=("rmse", "size"))
    )
    sanity.to_csv(out_dir / "sanity_vs_protocol_means.csv", index=False)

    wall = time.perf_counter() - t_wall0
    report = {
        "status": "PASS",
        "output_dir": str(out_dir),
        "n_slots": len(results),
        "wall_seconds": wall,
        "seed_audit": "PASS",
        "results": results,
        "sanity": sanity.to_dict(orient="records"),
    }
    (out_dir / "DONE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Baseline gene-metrics report",
                "",
                f"- Status: **{report['status']}**",
                f"- Wall: **{wall:.1f}s** ({wall/3600:.2f} h)",
                f"- Slots: {len(results)}",
                f"- Seed audit: PASS",
                "",
                "## Sanity (global RMSE mean over folds×reps @ 20%)",
                "",
                sanity.to_string(index=False),
                "",
                "## Outputs",
                "- `per_gene_summary.csv`",
                "- `gene_method_rmse_wide.csv`",
                "- `per_gene_raw.csv`",
                "- `masks/`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"DONE -> {out_dir} ({wall:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
