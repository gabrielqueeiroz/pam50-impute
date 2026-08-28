#!/usr/bin/env python3
"""
Fair paired baseline gene-metrics vs OriginalRFECA freeze.

Protocol aligned for comparison (NOT the legacy CV campaign):
  - Reuse Exact masks from artifacts/original_rfeca_reduced_metabric/
  - Seeds v2 / mask hashes audited against FREEZE/mask_hashes.csv
  - Grid: MCAR+MAR x {0.05,0.10,0.20,0.30} x reps 0-4  (40 slots)
  - Methods: SimpleMean, KNN(k=5,dist), MissForest
  - Evaluation: repeated_mask_holdout (fit once on full masked matrix;
    RMSE/MAE on all artificially masked cells — same cells as OriginalRFECA)
  - Imputation-only; per-gene + slot-level metrics

After completion, regenerates:
  - artifacts/final_analysis/figures/gene_method_heatmap_rmse_fair.png
  - artifacts/final_analysis/figures/gene_method_heatmap_rmse_zscore_fair.png
  - artifacts/final_analysis/fair_gene_comparison_*.csv/md

Example:
  python experiments/run_fair_baseline_holdout_vs_rfeca.py --dry-run
  python experiments/run_fair_baseline_holdout_vs_rfeca.py --confirm --workers 8
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.base import clone
from sklearn.impute import KNNImputer, SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.config import PAM50_GENES  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.imputers import make_missforest  # noqa: E402
from bcimpute.missingness import missingness_seed  # noqa: E402

RFECA_ROOT = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
FREEZE_HASHES = RFECA_ROOT / "FREEZE" / "mask_hashes.csv"
OUT_DEFAULT = ROOT / "artifacts" / "fair_baseline_holdout_vs_rfeca"
FINAL = ROOT / "artifacts" / "final_analysis"

RATES = [0.05, 0.10, 0.20, 0.30]
MECHS = ["mcar", "mar"]
REPS = [0, 1, 2, 3, 4]
DISPLAY = {
    "SimpleMean": "Mean",
    "KNN(k=5,dist)": "KNN",
    "MissForest": "MissForest",
}
METHODS = list(DISPLAY.keys())


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(bool)).tobytes()
    ).hexdigest()[:16]


def _rate_dir(rate: float) -> str:
    return f"rate_{rate:.2f}"


def _slot_mask_path(mech: str, rate: float, rep: int) -> Path:
    return RFECA_ROOT / mech / _rate_dir(rate) / f"rep_{rep}" / "mask.npz"


def audit_masks() -> pd.DataFrame:
    freeze = pd.read_csv(FREEZE_HASHES)
    rows = []
    for _, r in freeze.iterrows():
        mech = str(r["mechanism"]).lower()
        rate = float(r["rate"])
        rep = int(r["replicate"])
        expected_seed = int(r["seed"])
        expected_hash = str(r["mask_hash"])
        v2 = missingness_seed(42, rate, rep, mechanism=mech, scheme="v2")
        path = _slot_mask_path(mech, rate, rep)
        ok_path = path.exists()
        got_hash = None
        got_seed = None
        if ok_path:
            z = np.load(path, allow_pickle=True)
            mask = np.asarray(z["mask"], dtype=bool)
            got_hash = _mask_hash(mask)
            if "seed" in z.files:
                got_seed = int(np.asarray(z["seed"]).reshape(-1)[0])
            elif "mask_hash" in z.files:
                # some files store hash only
                pass
        rows.append(
            {
                "mechanism": mech,
                "missing_rate": rate,
                "replicate": rep,
                "seed_freeze": expected_seed,
                "seed_v2_formula": v2,
                "seed_match_v2": v2 == expected_seed,
                "mask_path_exists": ok_path,
                "mask_hash_freeze": expected_hash,
                "mask_hash_file": got_hash,
                "hash_match": got_hash == expected_hash if got_hash else False,
                "seed_in_npz": got_seed,
            }
        )
    return pd.DataFrame(rows)


def _build_imputers(missforest_n_jobs: int) -> dict:
    return {
        "SimpleMean": SimpleImputer(strategy="mean"),
        "KNN(k=5,dist)": KNNImputer(n_neighbors=5, weights="distance"),
        "MissForest": make_missforest(
            n_estimators=20,
            max_iter=5,
            random_state=42,
            n_jobs=missforest_n_jobs,
        ),
    }


def _run_slot(payload: dict) -> dict:
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(k, "1")

    mech = payload["mechanism"]
    rate = float(payload["missing_rate"])
    rep = int(payload["replicate"])
    seed = int(payload["seed"])
    expected_hash = payload["expected_hash"]
    out_dir = Path(payload["out_dir"])
    mf_jobs = int(payload["missforest_n_jobs"])

    t0 = time.perf_counter()
    cohort = load_cohort("metabric")
    X = cohort.X[PAM50_GENES].astype(float)
    X_arr = X.to_numpy(dtype=float)
    genes = list(X.columns)

    z = np.load(_slot_mask_path(mech, rate, rep), allow_pickle=True)
    mask = np.asarray(z["mask"], dtype=bool)
    mhash = _mask_hash(mask)
    if mhash != expected_hash:
        raise AssertionError(
            f"mask hash mismatch {mech} {rate} rep{rep}: {mhash} != {expected_hash}"
        )

    X_missing = X.copy()
    X_missing_arr = X_missing.to_numpy(dtype=float)
    X_missing_arr[mask] = np.nan
    X_missing = pd.DataFrame(X_missing_arr, index=X.index, columns=X.columns)

    imputers = _build_imputers(mf_jobs)
    gene_rows = []
    slot_rows = []

    for name, imputer in imputers.items():
        imp = clone(imputer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp.fit(X_missing)
            X_imp = np.asarray(imp.transform(X_missing), dtype=float)

        # global on all masked cells
        tv = X_arr[mask]
        pv = X_imp[mask]
        diff = pv - tv
        slot_rows.append(
            {
                "mechanism": mech,
                "missing_rate": rate,
                "replicate": rep,
                "seed": seed,
                "mask_hash": mhash,
                "imputer": name,
                "method": DISPLAY[name],
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "mae": float(np.mean(np.abs(diff))),
                "n_masked": int(mask.sum()),
                "evaluation_protocol": "repeated_mask_holdout",
                "seed_scheme": "v2",
                "paired_with": "OriginalRFECA_FREEZE",
            }
        )

        for j, gene in enumerate(genes):
            gmask = mask[:, j]
            if not np.any(gmask):
                continue
            gdiff = X_imp[gmask, j] - X_arr[gmask, j]
            gene_rows.append(
                {
                    "mechanism": mech,
                    "missing_rate": rate,
                    "replicate": rep,
                    "seed": seed,
                    "mask_hash": mhash,
                    "imputer": name,
                    "method": DISPLAY[name],
                    "gene": gene,
                    "rmse": float(np.sqrt(np.mean(gdiff**2))),
                    "mae": float(np.mean(np.abs(gdiff))),
                    "n_masked": int(gmask.sum()),
                }
            )

    slot_dir = out_dir / "slots" / mech / _rate_dir(rate) / f"rep_{rep}"
    slot_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gene_rows).to_csv(slot_dir / "per_gene_raw.csv", index=False)
    pd.DataFrame(slot_rows).to_csv(slot_dir / "slot_aggregate.csv", index=False)
    elapsed = time.perf_counter() - t0
    done = {
        "mechanism": mech,
        "missing_rate": rate,
        "replicate": rep,
        "seed": seed,
        "mask_hash": mhash,
        "wall_seconds": elapsed,
        "n_gene_rows": len(gene_rows),
        "status": "ok",
    }
    (slot_dir / "DONE.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    return done


def _load_rfeca_gene_means() -> pd.DataFrame:
    frames = []
    for p in sorted(RFECA_ROOT.glob("per_gene_all_*.csv")):
        frames.append(pd.read_csv(p))
    g = pd.concat(frames, ignore_index=True)
    g["mechanism"] = g["mechanism"].astype(str).str.lower()
    g["method"] = "OriginalRFECA"
    return (
        g.groupby(["mechanism", "missing_rate", "gene"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_reps=("replicate", "nunique"),
        )
    )


def build_comparison_outputs(out_dir: Path) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    fig_dir = FINAL / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    gene_parts = sorted(out_dir.glob("slots/*/*/*/per_gene_raw.csv"))
    slot_parts = sorted(out_dir.glob("slots/*/*/*/slot_aggregate.csv"))
    gene_raw = pd.concat([pd.read_csv(p) for p in gene_parts], ignore_index=True)
    slot_agg = pd.concat([pd.read_csv(p) for p in slot_parts], ignore_index=True)
    gene_raw.to_csv(out_dir / "per_gene_raw.csv", index=False)
    slot_agg.to_csv(out_dir / "slot_aggregate.csv", index=False)

    base_gene = (
        gene_raw.groupby(["mechanism", "missing_rate", "method", "gene"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_reps=("replicate", "nunique"),
        )
    )
    rfeca_gene = _load_rfeca_gene_means()
    rfeca_gene["method"] = "OriginalRFECA"
    all_gene = pd.concat([base_gene, rfeca_gene], ignore_index=True)
    all_gene.to_csv(FINAL / "fair_gene_comparison_long.csv", index=False)
    all_gene.to_csv(out_dir / "fair_gene_comparison_long.csv", index=False)

    # Slot-level paired comparison (global RMSE)
    base_slot = (
        slot_agg.groupby(["mechanism", "missing_rate", "replicate", "method"], as_index=False)
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
    )
    # OriginalRFECA slot means from freeze / slot_level
    rfeca_slots = pd.read_csv(FINAL / "original_rfeca_slot_level.csv")
    rfeca_slots["mechanism"] = rfeca_slots["mechanism"].astype(str).str.lower()
    rfeca_slots["method"] = "OriginalRFECA"
    rfeca_s = rfeca_slots[
        ["mechanism", "missing_rate", "replicate", "method", "rmse", "mae"]
    ]
    slot_all = pd.concat([base_slot, rfeca_s], ignore_index=True)
    slot_all.to_csv(FINAL / "fair_slot_comparison_long.csv", index=False)

    slot_summary = (
        slot_all.groupby(["mechanism", "missing_rate", "method"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n=("rmse", "size"),
        )
    )
    slot_summary.to_csv(FINAL / "fair_imputation_comparison_summary.csv", index=False)
    slot_summary.to_csv(out_dir / "fair_imputation_comparison_summary.csv", index=False)

    # Wide display table
    def fmt(m, s):
        if pd.isna(m):
            return "n/a"
        if pd.isna(s):
            return f"{m:.3f}"
        return f"{m:.3f} +/- {s:.3f}"

    wide_rows = []
    for mech in MECHS:
        for rate in RATES:
            row = {"mechanism": mech.upper(), "rate_pct": int(rate * 100)}
            means = {}
            for method in ["Mean", "KNN", "MissForest", "OriginalRFECA"]:
                sub = slot_summary[
                    (slot_summary.mechanism == mech)
                    & (np.isclose(slot_summary.missing_rate, rate))
                    & (slot_summary.method == method)
                ]
                if len(sub) == 0:
                    row[f"{method}_RMSE"] = "n/a"
                    continue
                s = sub.iloc[0]
                row[f"{method}_RMSE"] = fmt(s.rmse_mean, s.rmse_std)
                row[f"{method}_MAE"] = fmt(s.mae_mean, s.mae_std)
                means[method] = float(s.rmse_mean)
            order = sorted(means, key=means.get)
            row["rmse_best"] = order[0]
            row["rmse_ranking"] = " < ".join(f"{m}({means[m]:.3f})" for m in order)
            if "OriginalRFECA" in means and "MissForest" in means:
                row["delta_RFECA_minus_MF"] = means["OriginalRFECA"] - means["MissForest"]
            wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(FINAL / "fair_imputation_comparison_display.csv", index=False)

    # Heatmap: mean RMSE across all 8 cells per gene, or pool all mech×rate
    # Use overall mean across mechanism×rate×reps already in all_gene aggregated further
    overall = (
        all_gene.groupby(["gene", "method"], as_index=False)
        .agg(rmse_mean=("rmse_mean", "mean"))
    )
    piv = overall.pivot(index="gene", columns="method", values="rmse_mean")
    for c in ["Mean", "KNN", "MissForest", "OriginalRFECA"]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[["Mean", "KNN", "MissForest", "OriginalRFECA"]]
    piv["rmse_global"] = piv.mean(axis=1, skipna=True)
    piv = piv.sort_values("rmse_global", ascending=False)
    piv_out = piv.reset_index()
    piv_out.to_csv(FINAL / "gene_method_heatmap_fair.csv", index=False)

    _plot_heatmap(piv, zscore=False, path=fig_dir / "gene_method_heatmap_rmse_fair.png")
    _plot_heatmap(piv, zscore=True, path=fig_dir / "gene_method_heatmap_rmse_zscore_fair.png")

    # Gene winners
    win = piv[["Mean", "KNN", "MissForest", "OriginalRFECA"]].copy()
    win["best"] = win.idxmin(axis=1)
    win_counts = win["best"].value_counts().rename_axis("method").reset_index(name="n_genes")
    win_counts.to_csv(FINAL / "fair_per_gene_winners.csv", index=False)

    # Markdown report
    lines = [
        "# Fair imputation comparison (holdout + freeze masks)",
        "",
        "Protocol: `repeated_mask_holdout`, seeds/masks from OriginalRFECA FREEZE (v2).",
        "Methods: Mean, KNN, MissForest, OriginalRFECA.",
        "",
        "## Slot-level RMSE (mean +/- SD over 5 reps)",
        "",
        wide.to_string(index=False),
        "",
        "## Per-gene winners (lowest mean RMSE across grid)",
        "",
        win_counts.to_string(index=False),
        "",
        "## Figures",
        "- `figures/gene_method_heatmap_rmse_fair.png`",
        "- `figures/gene_method_heatmap_rmse_zscore_fair.png`",
        "",
        "## Tables",
        "- `fair_imputation_comparison_display.csv`",
        "- `fair_gene_comparison_long.csv`",
        "- `fair_per_gene_winners.csv`",
        "",
    ]
    (FINAL / "fair_imputation_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "fair_imputation_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_heatmap(piv: pd.DataFrame, zscore: bool, path: Path) -> None:
    cols = ["Mean", "KNN", "MissForest", "OriginalRFECA"]
    mat = piv[cols].to_numpy(dtype=float)
    genes = list(piv.index)
    best_idx = np.nanargmin(mat, axis=1)

    if zscore:
        data = np.full_like(mat, np.nan)
        for i in range(mat.shape[0]):
            row = mat[i]
            obs = np.isfinite(row)
            if obs.sum() >= 2:
                mu, sd = row[obs].mean(), row[obs].std(ddof=0)
                data[i, obs] = (row[obs] - mu) / sd if sd > 0 else 0.0
        display = data
        cmap = plt.get_cmap("RdBu_r").copy()
        title = "Per-gene RMSE z-score (★ indicates the lowest RMSE)"
        cbar = "z-score"
        finite = display[np.isfinite(display)]
        lim = float(np.nanmax(np.abs(finite))) if len(finite) else 1.0
        vmin, vmax = -max(lim, 1.0), max(lim, 1.0)
    else:
        display = mat
        cmap = plt.get_cmap("YlOrRd").copy()
        title = "Per-gene mean RMSE across imputers (★ indicates the lowest RMSE)"
        cbar = "RMSE"
        finite = display[np.isfinite(display)]
        vmin, vmax = float(finite.min()), float(finite.max())

    fig, ax = plt.subplots(figsize=(8.5, max(10, 0.22 * len(genes) + 2.2)))
    im = ax.imshow(
        np.ma.masked_invalid(display),
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        extent=(-0.5, len(cols) - 0.5, len(genes) - 0.5, -0.5),
    )

    # Star on the lowest-RMSE method per gene.
    ax.scatter(
        best_idx,
        np.arange(len(genes)),
        marker="*",
        s=55,
        c="white",
        edgecolors="black",
        linewidths=0.45,
        zorder=3,
    )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--missforest-n-jobs", type=int, default=2)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip imputation; rebuild comparison from existing out dir.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_masks()
    audit.to_csv(out_dir / "mask_audit.csv", index=False)
    n_ok = int(audit["hash_match"].sum())
    print(f"Mask audit: {n_ok}/{len(audit)} hash matches")
    if n_ok != len(audit):
        print(audit[~audit["hash_match"]].to_string(index=False))
        return 2
    if not bool(audit["seed_match_v2"].all()):
        print("WARNING: some freeze seeds != v2 formula (still using freeze masks)")

    freeze = pd.read_csv(FREEZE_HASHES)
    config = {
        "protocol": "repeated_mask_holdout",
        "seed_scheme": "v2",
        "paired_masks": "original_rfeca_reduced_metabric FREEZE",
        "methods": METHODS,
        "mechanisms": MECHS,
        "rates": RATES,
        "replicates": REPS,
        "n_slots": 40,
        "workers": args.workers,
        "missforest_n_jobs": args.missforest_n_jobs,
        "classification": False,
    }
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    if args.figures_only:
        build_comparison_outputs(out_dir)
        print(f"Figures rebuilt from {out_dir}")
        return 0

    if args.dry_run or not args.confirm:
        print(f"Prepared at {out_dir} (mask audit PASS). Pass --confirm to launch.")
        return 0

    payloads = []
    for _, r in freeze.iterrows():
        payloads.append(
            {
                "mechanism": str(r["mechanism"]).lower(),
                "missing_rate": float(r["rate"]),
                "replicate": int(r["replicate"]),
                "seed": int(r["seed"]),
                "expected_hash": str(r["mask_hash"]),
                "out_dir": str(out_dir),
                "missforest_n_jobs": args.missforest_n_jobs,
            }
        )

    # Skip completed slots
    pending = []
    for p in payloads:
        done = (
            out_dir
            / "slots"
            / p["mechanism"]
            / _rate_dir(p["missing_rate"])
            / f"rep_{p['replicate']}"
            / "DONE.json"
        )
        if done.exists():
            continue
        pending.append(p)
    print(f"Launching {len(pending)}/{len(payloads)} slots, workers={args.workers}")

    t0 = time.perf_counter()
    progress = out_dir / "progress.jsonl"
    results = []
    workers = max(1, args.workers)

    if workers == 1:
        for i, p in enumerate(pending, 1):
            print(
                f"[{i}/{len(pending)}] {p['mechanism']} rate={p['missing_rate']} "
                f"rep={p['replicate']} ..."
            )
            done = _run_slot(p)
            results.append(done)
            with progress.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(done) + "\n")
            print(f"  wall={done['wall_seconds']:.1f}s hash={done['mask_hash']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_slot, p): p for p in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                done = fut.result()
                results.append(done)
                with progress.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(done) + "\n")
                print(
                    f"[{i}/{len(pending)}] {done['mechanism']} "
                    f"rate={done['missing_rate']} rep={done['replicate']} "
                    f"wall={done['wall_seconds']:.1f}s"
                )

    wall = time.perf_counter() - t0
    build_comparison_outputs(out_dir)
    report = {
        "status": "PASS",
        "n_slots_run": len(results),
        "wall_seconds": wall,
        "output_dir": str(out_dir),
    }
    (out_dir / "DONE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"DONE -> {out_dir} ({wall:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
