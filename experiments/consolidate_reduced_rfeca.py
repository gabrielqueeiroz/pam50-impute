#!/usr/bin/env python3
"""Consolidate reduced OriginalRFECA (MCAR+MAR @20%) and recommend scope expansion."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REDUCED = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
OUT = ROOT / "artifacts" / "original_rfeca_reduced"
BASELINE_FULL = ROOT / "artifacts" / "metabric_full_20260719_234633"


def _slot_done(mech: str, rep: int) -> Path:
    return REDUCED / mech / "rate_0.20" / f"rep_{rep}" / "DONE.json"


def _slot_dir(mech: str, rep: int) -> Path:
    return REDUCED / mech / "rate_0.20" / f"rep_{rep}"


def _validate_replica(mech: str, rep: int) -> dict:
    d = _slot_dir(mech, rep)
    done = d / "DONE.json"
    pg = d / "per_gene_metrics.csv"
    status = {
        "mechanism": mech,
        "replicate": rep,
        "path": str(d),
        "status": "missing",
        "valid": False,
    }
    if not done.exists() or not pg.exists():
        status["status"] = "partial" if (d / "checkpoint").exists() else "missing"
        return status
    summary = json.loads(done.read_text(encoding="utf-8"))
    df = pd.read_csv(pg)
    n_genes = len(df)
    n_ok = int((df["status"].astype(str) == "ok").sum()) if "status" in df else n_genes
    n_fail = n_genes - n_ok
    finite = bool(np.isfinite(df["rmse"]).all() and np.isfinite(df["mae"]).all())
    fb = int(df["fallback_count"].sum()) if "fallback_count" in df else 0
    pred_nan = int(summary.get("n_predictor_nans_at_impute", 0))
    cov = float(summary.get("svr_coverage", 0))
    valid = (
        n_genes == 50
        and n_fail == 0
        and finite
        and fb == 0
        and pred_nan == 0
        and cov >= 1.0 - 1e-12
        and summary.get("classification") in {"A", "B"}
        and not summary.get("leakage_or_protocol_fail", False)
    )
    if n_fail > 0:
        st = "failed"
    elif not valid:
        st = "invalid"
    else:
        st = "completed"
    status.update(
        {
            "status": st if valid else ("invalid" if st == "completed" else st),
            "valid": valid,
            "n_genes": n_genes,
            "n_ok": n_ok,
            "n_fail": n_fail,
            "rmse": float(summary.get("rmse", float("nan"))),
            "mae": float(summary.get("mae", float("nan"))),
            "wall_seconds": float(summary.get("wall_seconds", float("nan"))),
            "svr_coverage": cov,
            "fallback_count": fb,
            "classification": summary.get("classification"),
            "seed": summary.get("seed"),
            "mask_hash": summary.get("mask_hash"),
        }
    )
    # Normalize: completed+valid only if valid
    if valid:
        status["status"] = "completed"
    return status


def _load_per_gene(mech: str, reps: list[int]) -> pd.DataFrame:
    frames = []
    for r in reps:
        p = _slot_dir(mech, r) / "per_gene_metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["mechanism"] = mech
        df["replicate"] = r
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _agg_mech(df: pd.DataFrame, mech: str, wall_total: float) -> dict:
    if df.empty:
        return {"mechanism": mech, "n_rows": 0}
    return {
        "mechanism": mech,
        "n_gene_rows": int(len(df)),
        "n_unique_genes": int(df["gene"].nunique()),
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_median": float(df["rmse"].median()),
        "rmse_std": float(df["rmse"].std(ddof=1)) if len(df) > 1 else 0.0,
        "rmse_q25": float(df["rmse"].quantile(0.25)),
        "rmse_q75": float(df["rmse"].quantile(0.75)),
        "mae_mean": float(df["mae"].mean()),
        "mae_median": float(df["mae"].median()),
        "prefix_mean": float(df["winning_prefix_len"].mean()),
        "n_features_mean": float(df["n_predictors_selected"].mean()),
        "wall_seconds_total": wall_total,
    }


def _recommend(mcar: dict, mar: dict, vs_baseline: dict | None) -> dict:
    """Heuristic dissertation-oriented recommendation."""
    reasons = []
    score_expand = 0

    if mcar.get("n_gene_rows", 0) and mar.get("n_gene_rows", 0):
        delta = abs(mcar["rmse_mean"] - mar["rmse_mean"]) / max(mcar["rmse_mean"], 1e-9)
        if delta >= 0.05:
            score_expand += 2
            reasons.append(
                f"MCAR vs MAR RMSE differs by {delta*100:.1f}% "
                f"({mcar['rmse_mean']:.4f} vs {mar['rmse_mean']:.4f})"
            )
        else:
            reasons.append(
                f"MCAR vs MAR nearly redundant (RMSE delta {delta*100:.1f}%)"
            )

    if vs_baseline:
        # competitive if within 10% of best non-RFECA or better than mean
        ratio = vs_baseline.get("rfeca_vs_best_ratio")
        if ratio is not None:
            if ratio <= 1.05:
                score_expand += 2
                reasons.append(
                    f"OriginalRFECA competitive vs best baseline "
                    f"(ratio={ratio:.3f})"
                )
            elif ratio <= 1.15:
                score_expand += 1
                reasons.append(
                    f"OriginalRFECA within 15% of best baseline (ratio={ratio:.3f})"
                )
            else:
                reasons.append(
                    f"OriginalRFECA clearly behind best baseline (ratio={ratio:.3f})"
                )
        reasons.append(
            "Baseline methods already cover 10/20/30 — asymmetry favors "
            "at least documenting RFECA degradation curve"
        )
        score_expand += 1  # methodological symmetry argument

    # Always note dissertation value of severity curve
    reasons.append(
        "Degradation across missingness severity is scientifically useful "
        "if RFECA is retained in the dissertation comparison"
    )

    if score_expand >= 3:
        decision = "ampliar"
        detail = (
            "Execute 10% and 30% for MCAR and MAR with 5 replicates "
            "(full reduced-severity grid)."
        )
    elif score_expand == 2:
        decision = "ampliar_parcialmente"
        detail = (
            "Minimum informative add-on: MCAR+MAR at 30% only (5 reps), "
            "keeping 20% as the primary table; add 10% later if needed."
        )
    else:
        decision = "nao_ampliar"
        detail = (
            "Keep reduced 20%-only scope for OriginalRFECA; discuss cost/"
            "performance trade-off and asymmetry as a limitation."
        )

    return {
        "decision": decision,
        "detail": detail,
        "score_expand": score_expand,
        "reasons": reasons,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _baseline_compare(rate: float = 0.20) -> dict | None:
    """Compare to full-benchmark imputers at same rate if available."""
    summary = BASELINE_FULL / "exp1_imputation_summary.csv"
    if not summary.exists():
        return None
    df = pd.read_csv(summary)
    # expect columns like imputer, missing_rate, rmse
    rate_col = "missing_rate" if "missing_rate" in df.columns else None
    if rate_col:
        df = df[np.isclose(df[rate_col].astype(float), rate)]
    if df.empty or "imputer" not in df.columns:
        return None
    # Prefer mean across reps if present
    if "rmse" not in df.columns:
        return None
    g = df.groupby("imputer", as_index=False)["rmse"].mean()
    best = g.loc[g["rmse"].idxmin()]
    return {
        "source": str(summary),
        "by_imputer": {str(r.imputer): float(r.rmse) for r in g.itertuples()},
        "best_imputer": str(best.imputer),
        "best_rmse": float(best.rmse),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--elapsed-hours", type=float, default=None)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # Version previous report if present
    for name in ("reduced_benchmark_report.md", "validation_report.json"):
        old = OUT / name
        if old.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            shutil.copy2(old, OUT / f"{old.stem}_backup_{stamp}{old.suffix}")

    reps = [0, 1, 2, 3, 4]
    validations = []
    for mech in ("mcar", "mar"):
        for r in reps:
            validations.append(_validate_replica(mech, r))

    valid_reps = {
        mech: [
            v
            for v in validations
            if v["mechanism"] == mech and v.get("valid") and v["status"] == "completed"
        ]
        for mech in ("mcar", "mar")
    }

    # Per-replica metrics
    rep_rows = []
    for v in validations:
        if v.get("rmse") is not None:
            rep_rows.append(v)
    rep_df = pd.DataFrame(rep_rows)
    if not rep_df.empty:
        rep_df.to_csv(OUT / "per_replica_metrics.csv", index=False)

    gene_mcar = _load_per_gene("mcar", [v["replicate"] for v in valid_reps["mcar"]])
    gene_mar = _load_per_gene("mar", [v["replicate"] for v in valid_reps["mar"]])
    gene_all = pd.concat([gene_mcar, gene_mar], ignore_index=True)
    if not gene_all.empty:
        gene_all.to_csv(OUT / "per_gene_metrics.csv", index=False)
        gene_all.to_csv(OUT / "reduced_results_combined.csv", index=False)

    # Mechanism summaries
    mcar_wall = float(
        sum(v.get("wall_seconds", 0) or 0 for v in valid_reps["mcar"])
    )
    mar_wall = float(sum(v.get("wall_seconds", 0) or 0 for v in valid_reps["mar"]))
    mcar_agg = _agg_mech(gene_mcar, "mcar", mcar_wall)
    mar_agg = _agg_mech(gene_mar, "mar", mar_wall)
    pd.DataFrame([mcar_agg, mar_agg]).to_csv(OUT / "mcar_20_results.csv", index=False)
    # also write mar file explicitly
    pd.DataFrame([mar_agg]).to_csv(OUT / "mar_20_results.csv", index=False)

    # Selected features summary
    if not gene_all.empty:
        feat = (
            gene_all.groupby(["mechanism", "gene"], as_index=False)
            .agg(
                mean_prefix=("winning_prefix_len", "mean"),
                mean_n_features=("n_predictors_selected", "mean"),
                mean_rmse=("rmse", "mean"),
            )
        )
        feat.to_csv(OUT / "selected_features_summary.csv", index=False)

    # Runtime from observed walls
    runtime_rows = []
    for v in validations:
        if v.get("wall_seconds") and np.isfinite(v["wall_seconds"]):
            runtime_rows.append(
                {
                    "mechanism": v["mechanism"],
                    "replicate": v["replicate"],
                    "wall_seconds": v["wall_seconds"],
                    "wall_hours": v["wall_seconds"] / 3600,
                    "status": v["status"],
                    "valid": v.get("valid"),
                }
            )
    runtime_df = pd.DataFrame(runtime_rows)
    if not runtime_df.empty:
        runtime_df.to_csv(OUT / "runtime_metrics.csv", index=False)

    # ETA update from observed
    if not runtime_df.empty:
        sec_per_rep = float(runtime_df["wall_seconds"].mean())
        # effective s/gene amortized
        sec_per_gene_eff = sec_per_rep / 50.0
    else:
        sec_per_rep = float("nan")
        sec_per_gene_eff = 109.4  # from autotune 16w

    def band(hours: float) -> dict:
        return {
            "optimistic_hours": hours * 0.85,
            "central_hours": hours,
            "conservative_hours": hours * 1.25,
        }

    eta = {
        "seconds_per_gene_effective_observed": sec_per_gene_eff,
        "seconds_per_replica_observed": sec_per_rep,
        "mcar_20_5reps": band(5 * sec_per_rep / 3600 if np.isfinite(sec_per_rep) else 7.5),
        "mar_20_5reps": band(5 * sec_per_rep / 3600 if np.isfinite(sec_per_rep) else 7.5),
        "reduced_total": band(
            10 * sec_per_rep / 3600 if np.isfinite(sec_per_rep) else 15.0
        ),
        "extra_10pct_mcar_mar_5reps": band(
            10 * sec_per_rep / 3600 if np.isfinite(sec_per_rep) else 15.0
        ),
        "extra_30pct_mcar_mar_5reps": band(
            10 * sec_per_rep / 3600 if np.isfinite(sec_per_rep) else 15.0
        ),
        "expanded_full_10_20_30": band(
            30 * sec_per_rep / 3600 if np.isfinite(sec_per_rep) else 45.0
        ),
        "note": (
            "Bands: optimistic 0.85x, central 1.0x, conservative 1.25x of "
            "observed mean wall/replica. 10%/30% assumed similar cost to 20%."
        ),
        "elapsed_hours_orchestrator": args.elapsed_hours,
    }

    baseline = _baseline_compare(0.20)
    vs = None
    if baseline and mcar_agg.get("rmse_mean"):
        vs = {
            **baseline,
            "rfeca_mcar_rmse": mcar_agg["rmse_mean"],
            "rfeca_vs_best_ratio": mcar_agg["rmse_mean"] / baseline["best_rmse"],
        }

    recommendation = _recommend(mcar_agg, mar_agg, vs)

    validation_report = {
        "replicas": validations,
        "n_mcar_valid": len(valid_reps["mcar"]),
        "n_mar_valid": len(valid_reps["mar"]),
        "mcar_complete": len(valid_reps["mcar"]) == 5,
        "mar_complete": len(valid_reps["mar"]) == 5,
        "methodological_validity": (
            "A"
            if len(valid_reps["mcar"]) == 5 and len(valid_reps["mar"]) == 5
            else "B"
        ),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "scope_expansion_recommendation.json").write_text(
        json.dumps(
            {**recommendation, "eta": eta, "vs_baseline": vs},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Markdown report
    md = [
        "# Reduced OriginalRFECA benchmark (20% MCAR + MAR)",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- MCAR valid replicas: {len(valid_reps['mcar'])}/5",
        f"- MAR valid replicas: {len(valid_reps['mar'])}/5",
        f"- Methodological validity: {validation_report['methodological_validity']}",
        "",
        "## Aggregates",
        "",
        "| Mechanism | RMSE mean | RMSE median | RMSE std | MAE mean | wall (h) |",
        "|-----------|-----------|-------------|----------|----------|----------|",
        f"| MCAR | {mcar_agg.get('rmse_mean', float('nan')):.4f} | "
        f"{mcar_agg.get('rmse_median', float('nan')):.4f} | "
        f"{mcar_agg.get('rmse_std', float('nan')):.4f} | "
        f"{mcar_agg.get('mae_mean', float('nan')):.4f} | "
        f"{mcar_wall/3600:.2f} |",
        f"| MAR | {mar_agg.get('rmse_mean', float('nan')):.4f} | "
        f"{mar_agg.get('rmse_median', float('nan')):.4f} | "
        f"{mar_agg.get('rmse_std', float('nan')):.4f} | "
        f"{mar_agg.get('mae_mean', float('nan')):.4f} | "
        f"{mar_wall/3600:.2f} |",
        "",
        "## Scope expansion recommendation",
        "",
        f"**Decision: `{recommendation['decision']}`**",
        "",
        recommendation["detail"],
        "",
        "Reasons:",
    ]
    for r in recommendation["reasons"]:
        md.append(f"- {r}")
    md += [
        "",
        "## ETA (observed-based)",
        "",
        f"- s/gene effective: {sec_per_gene_eff:.1f}",
        f"- s/replica: {sec_per_rep:.1f}",
        f"- Reduced total central: {eta['reduced_total']['central_hours']:.1f} h "
        f"(opt {eta['reduced_total']['optimistic_hours']:.1f}, "
        f"cons {eta['reduced_total']['conservative_hours']:.1f})",
        f"- Extra for 10% grid: ~{eta['extra_10pct_mcar_mar_5reps']['central_hours']:.1f} h",
        f"- Extra for 30% grid: ~{eta['extra_30pct_mcar_mar_5reps']['central_hours']:.1f} h",
        f"- Full expanded 10/20/30: ~{eta['expanded_full_10_20_30']['central_hours']:.1f} h",
        "",
        "## Next step",
        "",
        "Do **not** auto-start 10%/30%. Await explicit decision.",
        "",
    ]
    (OUT / "reduced_benchmark_report.md").write_text("\n".join(md), encoding="utf-8")

    # Simple plots if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not gene_all.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            data = [
                gene_mcar["rmse"].dropna().to_numpy() if not gene_mcar.empty else [],
                gene_mar["rmse"].dropna().to_numpy() if not gene_mar.empty else [],
            ]
            ax.boxplot(data, tick_labels=["MCAR", "MAR"])
            ax.set_ylabel("RMSE")
            ax.set_title("OriginalRFECA RMSE @ 20%")
            fig.tight_layout()
            fig.savefig(OUT / "rmse_by_mechanism.png", dpi=140)
            plt.close(fig)
        if not runtime_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for mech, sub in runtime_df.groupby("mechanism"):
                ax.plot(sub["replicate"], sub["wall_hours"], "o-", label=mech)
            ax.set_xlabel("replicate")
            ax.set_ylabel("wall hours")
            ax.legend()
            ax.set_title("Wall time per replicate")
            fig.tight_layout()
            fig.savefig(OUT / "wall_by_replica.png", dpi=140)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"plot warning: {exc}", flush=True)

    print(json.dumps(recommendation, indent=2))
    print(f"Wrote {OUT}")
    mcar_ok = len(valid_reps["mcar"]) == 5
    mar_ok = len(valid_reps["mar"]) == 5
    return 0 if mcar_ok and mar_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
