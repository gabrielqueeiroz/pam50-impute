#!/usr/bin/env python3
"""
Regenerate OriginalRFECA TARGET-WISE paper tables/figures from frozen artifacts only.

Inputs (no re-fit):
  artifacts/original_rfeca_reduced_metabric/{mcar,mar}/rate_*/rep_*/DONE.json
  artifacts/original_rfeca_reduced_metabric/per_gene_all_*.csv
  artifacts/original_rfeca_reduced_metabric/FREEZE/manifest.json  (cross-check)

Outputs:
  artifacts/paper_results_original_rfeca/

Usage (repo root):
  python scripts/generate_original_rfeca_paper_results.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
OUT = ROOT / "artifacts" / "paper_results_original_rfeca"
FREEZE = ART / "FREEZE"

MECHS = ("mcar", "mar")
RATES = (0.10, 0.20, 0.30)
REPS = range(5)

# Paper display rounding (full precision kept in *_full.csv)
ROUND_RMSE = 3
ROUND_MAE = 3
ROUND_PCT = 1

# Absolute tolerance: recomputed mean vs REPORT mean
TOL_REPORT = 1e-12
# Absolute tolerance: rounded display vs figure annotation source
TOL_ROUND = 5e-4  # half-up edge at 3 decimals


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _round(x: float, nd: int) -> float:
    """Half-up rounding for display (matches typical paper tables)."""
    if not np.isfinite(x):
        return float("nan")
    p = 10**nd
    return math.floor(float(x) * p + 0.5) / p


def _fmt(x: float, nd: int) -> str:
    return f"{_round(x, nd):.{nd}f}"


def _sha16(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_slots() -> pd.DataFrame:
    rows = []
    for mech in MECHS:
        for rate in RATES:
            for rep in REPS:
                d = ART / mech / f"rate_{rate:.2f}" / f"rep_{rep}"
                done = d / "DONE.json"
                if not done.exists():
                    raise FileNotFoundError(done)
                s = json.loads(done.read_text(encoding="utf-8"))
                mask = d / "mask.npz"
                rows.append(
                    {
                        "mechanism": mech,
                        "rate": float(rate),
                        "rate_pct": int(round(rate * 100)),
                        "replicate": int(rep),
                        "seed": int(s["seed"]),
                        "mask_hash": s["mask_hash"],
                        "mask_npz_sha16": _sha16(mask),
                        "rmse": float(s["rmse"]),
                        "mae": float(s["mae"]),
                        "classification": s.get("classification"),
                        "svr_coverage": float(s.get("svr_coverage", float("nan"))),
                        "fallback_rate": float(s.get("fallback_rate", float("nan"))),
                        "n_predictor_nans_at_impute": int(
                            s.get("n_predictor_nans_at_impute", -1)
                        ),
                        "n_genes_completed": int(s.get("n_genes_completed", 0)),
                        "n_failures": int(s.get("n_failures", 0)),
                        "wall_seconds": float(s.get("wall_seconds", float("nan"))),
                        "evaluation_protocol": s.get("evaluation_protocol"),
                        "input_protocol": s.get("input_protocol"),
                        "predictor_values": s.get("predictor_values"),
                        "selection_protocol": s.get("selection_protocol"),
                        "use_scaler": s.get("use_scaler"),
                        "max_candidates": s.get("max_candidates"),
                    }
                )
    df = pd.DataFrame(rows)
    assert len(df) == 30, f"expected 30 slots, got {len(df)}"
    return df


def load_per_gene() -> pd.DataFrame:
    frames = []
    for p in sorted(ART.glob("per_gene_all_*.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("no per_gene_all_*.csv")
    g = pd.concat(frames, ignore_index=True)
    g["rate"] = g["missing_rate"].astype(float)
    g["rate_pct"] = (g["rate"] * 100).round().astype(int)
    return g


def summarize_slots(slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mech, rate), g in slots.groupby(["mechanism", "rate"], sort=True):
        rows.append(
            {
                "mechanism": mech,
                "rate": float(rate),
                "rate_pct": int(round(float(rate) * 100)),
                "n_reps": int(len(g)),
                "rmse_mean": float(g["rmse"].mean()),
                "rmse_std": float(g["rmse"].std(ddof=1)),
                "rmse_median": float(g["rmse"].median()),
                "rmse_min": float(g["rmse"].min()),
                "rmse_max": float(g["rmse"].max()),
                "mae_mean": float(g["mae"].mean()),
                "mae_std": float(g["mae"].std(ddof=1)),
                "mae_median": float(g["mae"].median()),
                "wall_seconds_mean": float(g["wall_seconds"].mean()),
                "svr_coverage_min": float(g["svr_coverage"].min()),
                "fallback_rate_max": float(g["fallback_rate"].max()),
                "all_class_A": bool((g["classification"] == "A").all()),
                "n_predictor_nans_total": int(g["n_predictor_nans_at_impute"].sum()),
            }
        )
    return pd.DataFrame(rows)


def load_reports() -> pd.DataFrame:
    rows = []
    for mech in MECHS:
        for rate in RATES:
            pct = int(round(rate * 100))
            p = ART / f"REPORT_{mech.upper()}_{pct}_5REPS.json"
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "mechanism": mech,
                    "rate": float(rate),
                    "rate_pct": pct,
                    "report_rmse_mean": float(r["rmse_mean"]),
                    "report_rmse_std": float(r["rmse_std"]),
                    "report_mae_mean": float(r["mae_mean"]),
                    "report_classification": r.get("classification"),
                }
            )
    return pd.DataFrame(rows)


def summarize_genes(genes: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE/MAE per gene across reps, for each mech×rate."""
    agg = (
        genes.groupby(["mechanism", "rate", "rate_pct", "gene"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            n_reps=("replicate", "nunique"),
            n_predictors_selected_mean=("n_predictors_selected", "mean"),
        )
        .sort_values(["mechanism", "rate", "rmse_mean"])
    )
    return agg


def audit(
    slots: pd.DataFrame,
    summary: pd.DataFrame,
    reports: pd.DataFrame,
    genes: pd.DataFrame,
) -> dict:
    issues: list[str] = []
    checks: list[dict] = []

    # Protocol uniformity
    for col, expected in [
        ("evaluation_protocol", "repeated_mask_holdout"),
        ("input_protocol", "target_wise_complete_predictors"),
        ("predictor_values", "original_complete_matrix"),
        ("selection_protocol", "leakage_safe"),
    ]:
        ok = (slots[col] == expected).all()
        checks.append({"check": f"protocol_{col}", "pass": bool(ok), "expected": expected})
        if not ok:
            issues.append(f"protocol mismatch: {col}")

    if not (slots["use_scaler"] == False).all():  # noqa: E712
        issues.append("use_scaler not all False")
        checks.append({"check": "use_scaler_false", "pass": False})
    else:
        checks.append({"check": "use_scaler_false", "pass": True})

    if not (slots["max_candidates"] == 49).all():
        issues.append("max_candidates != 49")
        checks.append({"check": "max_candidates_49", "pass": False})
    else:
        checks.append({"check": "max_candidates_49", "pass": True})

    # All A, coverage, fallbacks
    all_a = (slots["classification"] == "A").all()
    checks.append({"check": "all_slots_class_A", "pass": bool(all_a)})
    if not all_a:
        issues.append("not all slots classification A")

    cov_ok = (slots["svr_coverage"] >= 1.0 - 1e-12).all()
    checks.append({"check": "svr_coverage_1", "pass": bool(cov_ok)})
    fb_ok = (slots["fallback_rate"] == 0).all() and (
        slots["n_predictor_nans_at_impute"] == 0
    ).all()
    checks.append({"check": "zero_fallbacks_and_pred_nans", "pass": bool(fb_ok)})

    # Recomputed summary vs REPORT_*
    merged = summary.merge(reports, on=["mechanism", "rate", "rate_pct"])
    for _, row in merged.iterrows():
        d_rmse = abs(row["rmse_mean"] - row["report_rmse_mean"])
        d_mae = abs(row["mae_mean"] - row["report_mae_mean"])
        ok = d_rmse <= TOL_REPORT and d_mae <= TOL_REPORT
        checks.append(
            {
                "check": f"report_match_{row['mechanism']}_{row['rate_pct']}",
                "pass": bool(ok),
                "delta_rmse": d_rmse,
                "delta_mae": d_mae,
            }
        )
        if not ok:
            issues.append(
                f"REPORT mismatch {row['mechanism']} {row['rate_pct']}% "
                f"d_rmse={d_rmse} d_mae={d_mae}"
            )

    # FREEZE mask hashes
    freeze_csv = FREEZE / "mask_hashes.csv"
    if freeze_csv.exists():
        fz = pd.read_csv(freeze_csv)
        m = slots.merge(
            fz,
            on=["mechanism", "rate", "replicate"],
            suffixes=("", "_fz"),
        )
        hash_ok = (m["mask_hash"] == m["mask_hash_fz"]).all()
        seed_ok = (m["seed"] == m["seed_fz"]).all()
        checks.append({"check": "freeze_mask_hash_match", "pass": bool(hash_ok)})
        checks.append({"check": "freeze_seed_match", "pass": bool(seed_ok)})
        if not hash_ok:
            issues.append("FREEZE mask_hash mismatch")
        if not seed_ok:
            issues.append("FREEZE seed mismatch")
    else:
        issues.append("FREEZE/mask_hashes.csv missing")
        checks.append({"check": "freeze_present", "pass": False})

    # Per-gene files: 50 genes × 5 reps × 6 mech×rate = 1500
    expected_gene_rows = 50 * 5 * 6
    gene_ok = len(genes) == expected_gene_rows
    checks.append(
        {
            "check": "per_gene_row_count",
            "pass": bool(gene_ok),
            "n": int(len(genes)),
            "expected": expected_gene_rows,
        }
    )
    if not gene_ok:
        issues.append(f"per_gene rows {len(genes)} != {expected_gene_rows}")

    status_ok = (genes["status"].astype(str) == "ok").all()
    checks.append({"check": "per_gene_all_ok", "pass": bool(status_ok)})

    # Slot RMSE equals cell-weighted? DONE.json rmse is the official slot metric;
    # cross-check finite and matches gene file presence.
    for _, s in slots.iterrows():
        sub = genes[
            (genes["mechanism"] == s["mechanism"])
            & (np.isclose(genes["rate"], s["rate"]))
            & (genes["replicate"] == s["replicate"])
        ]
        if len(sub) != 50:
            issues.append(
                f"gene count {len(sub)} for {s['mechanism']} {s['rate']} r{s['replicate']}"
            )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c.get("pass")),
        "n_fail": sum(1 for c in checks if not c.get("pass")),
        "issues": issues,
        "checks": checks,
        "rounding_policy": {
            "rmse_decimals": ROUND_RMSE,
            "mae_decimals": ROUND_MAE,
            "method": "half-up",
            "full_precision_files": ["*_full.csv", "slot_level_full.csv"],
            "display_files": ["*_display.csv", "table_*.tex", "figure annotations"],
        },
    }


def write_display_tables(summary: pd.DataFrame, slots: pd.DataFrame) -> None:
    # Full precision
    slots.to_csv(OUT / "slot_level_full.csv", index=False)
    summary.to_csv(OUT / "summary_by_mech_rate_full.csv", index=False)

    # Display rounding
    disp = summary.copy()
    for col, nd in [
        ("rmse_mean", ROUND_RMSE),
        ("rmse_std", ROUND_RMSE),
        ("rmse_median", ROUND_RMSE),
        ("rmse_min", ROUND_RMSE),
        ("rmse_max", ROUND_RMSE),
        ("mae_mean", ROUND_MAE),
        ("mae_std", ROUND_MAE),
        ("mae_median", ROUND_MAE),
    ]:
        disp[col] = disp[col].map(lambda x, n=nd: _round(float(x), n))
    # Force fixed decimals in CSV (avoid 0.64 vs 0.640)
    disp_out = disp.copy()
    for col, nd in [
        ("rmse_mean", ROUND_RMSE),
        ("rmse_std", ROUND_RMSE),
        ("rmse_median", ROUND_RMSE),
        ("rmse_min", ROUND_RMSE),
        ("rmse_max", ROUND_RMSE),
        ("mae_mean", ROUND_MAE),
        ("mae_std", ROUND_MAE),
        ("mae_median", ROUND_MAE),
    ]:
        disp_out[col] = disp_out[col].map(lambda x, n=nd: f"{float(x):.{n}f}")
    disp_out.to_csv(OUT / "summary_by_mech_rate_display.csv", index=False)

    # Wide paper table: mechanism × rate RMSE mean±sd
    wide_rows = []
    for mech in MECHS:
        row = {"mechanism": mech.upper()}
        for rate in RATES:
            pct = int(round(rate * 100))
            r = summary[(summary.mechanism == mech) & (summary.rate == rate)].iloc[0]
            row[f"rmse_{pct}"] = (
                f"{_fmt(r['rmse_mean'], ROUND_RMSE)} +/- {_fmt(r['rmse_std'], ROUND_RMSE)}"
            )
            row[f"mae_{pct}"] = (
                f"{_fmt(r['mae_mean'], ROUND_MAE)} +/- {_fmt(r['mae_std'], ROUND_MAE)}"
            )
            row[f"rmse_{pct}_mean"] = _fmt(r["rmse_mean"], ROUND_RMSE)
            row[f"rmse_{pct}_std"] = _fmt(r["rmse_std"], ROUND_RMSE)
            row[f"mae_{pct}_mean"] = _fmt(r["mae_mean"], ROUND_MAE)
            row[f"mae_{pct}_std"] = _fmt(r["mae_std"], ROUND_MAE)
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(OUT / "table_paper_rmse_mae_display.csv", index=False)

    # Compact numeric-only for plotting annotations
    numeric = summary[
        ["mechanism", "rate_pct", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    ].copy()
    numeric["rmse_mean_r"] = numeric["rmse_mean"].map(lambda x: _round(x, ROUND_RMSE))
    numeric["rmse_std_r"] = numeric["rmse_std"].map(lambda x: _round(x, ROUND_RMSE))
    numeric["mae_mean_r"] = numeric["mae_mean"].map(lambda x: _round(x, ROUND_MAE))
    numeric["mae_std_r"] = numeric["mae_std"].map(lambda x: _round(x, ROUND_MAE))
    numeric.to_csv(OUT / "figure_source_means.csv", index=False)

    # LaTeX
    lines = [
        "% Auto-generated from OriginalRFECA TARGET-WISE freeze artifacts",
        "% Do not edit by hand — regenerate via scripts/generate_original_rfeca_paper_results.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{OriginalRFECA TARGET-WISE reconstruction on METABRIC PAM50 "
        "(mean $\\pm$ SD across 5 missingness replications; "
        "repeated mask-holdout; leakage-safe selection).}",
        "\\label{tab:original_rfeca_metabric}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Mechanism & 10\\% & 20\\% & 30\\% \\\\",
        "\\hline",
        "\\multicolumn{4}{l}{\\textit{RMSE}} \\\\",
    ]
    for mech in MECHS:
        cells = []
        for rate in RATES:
            r = summary[(summary.mechanism == mech) & (summary.rate == rate)].iloc[0]
            cells.append(
                f"{_fmt(r['rmse_mean'], ROUND_RMSE)} $\\pm$ {_fmt(r['rmse_std'], ROUND_RMSE)}"
            )
        lines.append(f"{mech.upper()} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\multicolumn{4}{l}{\\textit{MAE}} \\\\"]
    for mech in MECHS:
        cells = []
        for rate in RATES:
            r = summary[(summary.mechanism == mech) & (summary.rate == rate)].iloc[0]
            cells.append(
                f"{_fmt(r['mae_mean'], ROUND_MAE)} $\\pm$ {_fmt(r['mae_std'], ROUND_MAE)}"
            )
        lines.append(f"{mech.upper()} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "table_original_rfeca_metabric.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_gene_tables(gene_sum: pd.DataFrame) -> None:
    gene_sum.to_csv(OUT / "per_gene_rmse_by_mech_rate_full.csv", index=False)
    # Top/bottom genes at 20% MCAR (paper highlight)
    sub = gene_sum[
        (gene_sum.mechanism == "mcar") & (np.isclose(gene_sum.rate, 0.20))
    ].sort_values("rmse_mean")
    top = sub.head(10).copy()
    bot = sub.tail(10).sort_values("rmse_mean", ascending=False).copy()
    for df in (top, bot):
        df["rmse_mean_r"] = df["rmse_mean"].map(lambda x: _round(x, ROUND_RMSE))
        df["mae_mean_r"] = df["mae_mean"].map(lambda x: _round(x, ROUND_MAE))
    top.to_csv(OUT / "table_top10_genes_mcar20_lowest_rmse.csv", index=False)
    bot.to_csv(OUT / "table_bottom10_genes_mcar20_highest_rmse.csv", index=False)


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rmse_by_rate(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    colors = {"mcar": "#0072B2", "mar": "#D55E00"}
    markers = {"mcar": "o", "mar": "s"}
    for mech in MECHS:
        sub = summary[summary.mechanism == mech].sort_values("rate_pct")
        ax.errorbar(
            sub["rate_pct"],
            sub["rmse_mean"],
            yerr=sub["rmse_std"],
            label=mech.upper(),
            color=colors[mech],
            marker=markers[mech],
            capsize=3,
            linewidth=1.5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                _fmt(r["rmse_mean"], ROUND_RMSE),
                (r["rate_pct"], r["rmse_mean"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color=colors[mech],
            )
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("RMSE")
    ax.set_xticks([10, 20, 30])
    ax.set_title("OriginalRFECA TARGET-WISE — METABRIC")
    ax.legend(frameon=False)
    ax.set_ylim(0.58, 0.68)
    save_fig(fig, "fig01_rmse_by_missingness")


def fig_mae_by_rate(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    colors = {"mcar": "#0072B2", "mar": "#D55E00"}
    markers = {"mcar": "o", "mar": "s"}
    for mech in MECHS:
        sub = summary[summary.mechanism == mech].sort_values("rate_pct")
        ax.errorbar(
            sub["rate_pct"],
            sub["mae_mean"],
            yerr=sub["mae_std"],
            label=mech.upper(),
            color=colors[mech],
            marker=markers[mech],
            capsize=3,
            linewidth=1.5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                _fmt(r["mae_mean"], ROUND_MAE),
                (r["rate_pct"], r["mae_mean"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color=colors[mech],
            )
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("MAE")
    ax.set_xticks([10, 20, 30])
    ax.set_title("OriginalRFECA TARGET-WISE — METABRIC")
    ax.legend(frameon=False)
    save_fig(fig, "fig02_mae_by_missingness")


def fig_boxplot_slots(slots: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    colors = {"mcar": "#0072B2", "mar": "#D55E00"}
    for ax, mech in zip(axes, MECHS):
        data = [
            slots[(slots.mechanism == mech) & (slots.rate == rate)]["rmse"].to_numpy()
            for rate in RATES
        ]
        bp = ax.boxplot(
            data,
            tick_labels=[f"{int(r*100)}%" for r in RATES],
            patch_artist=True,
            widths=0.55,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[mech])
            patch.set_alpha(0.35)
        ax.set_title(mech.upper())
        ax.set_xlabel("Missingness")
        if mech == "mcar":
            ax.set_ylabel("RMSE (per replicate)")
    fig.suptitle("Replicate spread — OriginalRFECA TARGET-WISE", y=1.02)
    save_fig(fig, "fig03_rmse_boxplot_by_rate")


def fig_gene_bars(gene_sum: pd.DataFrame) -> None:
    sub = gene_sum[
        (gene_sum.mechanism == "mcar") & (np.isclose(gene_sum.rate, 0.20))
    ].sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    y = np.arange(len(sub))
    ax.barh(y, sub["rmse_mean"], color="#0072B2", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["gene"], fontsize=6)
    ax.set_xlabel("Mean RMSE (5 reps)")
    ax.set_title("Per-gene RMSE — MCAR 20%")
    ax.axvline(sub["rmse_mean"].mean(), color="#D55E00", linestyle="--", linewidth=1)
    save_fig(fig, "fig04_per_gene_rmse_mcar20")


def fig_mcar_mar_delta(summary: pd.DataFrame) -> None:
    mcar = summary[summary.mechanism == "mcar"].set_index("rate_pct")
    mar = summary[summary.mechanism == "mar"].set_index("rate_pct")
    delta = (mar["rmse_mean"] - mcar["rmse_mean"]).reindex([10, 20, 30])
    pct = 100.0 * delta / mcar["rmse_mean"].reindex([10, 20, 30])
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = pct.index.to_numpy()
    ax.bar(x, delta, width=4, color="#999999", edgecolor="black", linewidth=0.5)
    for xi, d, p in zip(x, delta, pct):
        ax.annotate(
            f"{_fmt(d, ROUND_RMSE)}\n({_fmt(p, ROUND_PCT)}%)",
            (xi, d),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("ΔRMSE (MAR − MCAR)")
    ax.set_xticks([10, 20, 30])
    ax.set_title("MAR − MCAR gap")
    ax.axhline(0, color="black", linewidth=0.6)
    save_fig(fig, "fig05_delta_mar_minus_mcar")


def verify_figure_table_consistency(summary: pd.DataFrame) -> list[dict]:
    """Ensure figure annotation sources match display table rounding."""
    disp = pd.read_csv(OUT / "summary_by_mech_rate_display.csv")
    src = pd.read_csv(OUT / "figure_source_means.csv")
    results = []
    for _, row in summary.iterrows():
        d = disp[
            (disp.mechanism == row.mechanism) & (disp.rate_pct == row.rate_pct)
        ].iloc[0]
        s = src[
            (src.mechanism == row.mechanism) & (src.rate_pct == row.rate_pct)
        ].iloc[0]
        d_rmse = float(d["rmse_mean"])
        d_mae = float(d["mae_mean"])
        s_rmse = float(s["rmse_mean_r"])
        s_mae = float(s["mae_mean_r"])
        ok_rmse = abs(d_rmse - s_rmse) < 1e-12
        ok_mae = abs(d_mae - s_mae) < 1e-12
        ok_half = abs(s_rmse - row["rmse_mean"]) <= TOL_ROUND + 1e-12
        results.append(
            {
                "mechanism": row.mechanism,
                "rate_pct": int(row.rate_pct),
                "full_rmse": float(row.rmse_mean),
                "display_rmse": d_rmse,
                "figure_rmse": s_rmse,
                "table_figure_match": bool(ok_rmse and ok_mae),
                "rounding_consistent": bool(ok_half),
            }
        )
    return results


def write_captions() -> None:
    text = """# OriginalRFECA TARGET-WISE — figure/table captions

Protocol: METABRIC PAM50; TARGET-WISE complete predictors; `repeated_mask_holdout`;
leakage-safe Pearson prefixes + RFE + linear SVR; `max_candidates=49`; `use_scaler=false`;
`seed_scheme=v2`; `base_seed=42`; 5 replications; rates 10/20/30%; freeze `v0.3.0-original-rfeca-targetwise`.

Rounding: RMSE/MAE displayed to **3 decimals** (half-up). Full precision in `*_full.csv`.

## Figures
- **Fig. 1** `fig01_rmse_by_missingness` — mean±SD RMSE vs missingness (MCAR/MAR).
- **Fig. 2** `fig02_mae_by_missingness` — mean±SD MAE vs missingness.
- **Fig. 3** `fig03_rmse_boxplot_by_rate` — replicate RMSE spread.
- **Fig. 4** `fig04_per_gene_rmse_mcar20` — per-gene mean RMSE at MCAR 20%.
- **Fig. 5** `fig05_delta_mar_minus_mcar` — MAR−MCAR RMSE gap.

## Tables
- `table_original_rfeca_metabric.tex` / `table_paper_rmse_mae_display.csv` — main RMSE/MAE grid.
- `slot_level_full.csv` — 30 slots (seeds, mask hashes, metrics).
- `summary_by_mech_rate_{full,display}.csv` — aggregated means.
- `per_gene_rmse_by_mech_rate_full.csv` — gene-level means.
- `table_top10_genes_mcar20_lowest_rmse.csv` / `table_bottom10_genes_mcar20_highest_rmse.csv`.
"""
    (OUT / "figure_table_captions.md").write_text(text, encoding="utf-8")


def write_audit_md(audit: dict, consistency: list[dict]) -> None:
    lines = [
        "# Audit — OriginalRFECA paper package",
        "",
        f"Generated: `{audit['generated_at_utc']}`",
        f"Checks: **{audit['n_pass']}/{audit['n_checks']} PASS**"
        + (f" ({audit['n_fail']} FAIL)" if audit["n_fail"] else ""),
        "",
        "## Rounding",
        f"- Display decimals: RMSE={ROUND_RMSE}, MAE={ROUND_MAE} (half-up)",
        "- Figure annotations use the same rounded means as `summary_by_mech_rate_display.csv`",
        "",
        "## Table ↔ figure consistency",
    ]
    for c in consistency:
        flag = "PASS" if c["table_figure_match"] and c["rounding_consistent"] else "FAIL"
        lines.append(
            f"- {c['mechanism'].upper()} {c['rate_pct']}%: {flag} "
            f"(full={c['full_rmse']:.6f} → display/fig={c['display_rmse']:.3f})"
        )
    lines += ["", "## Automated checks"]
    for ch in audit["checks"]:
        st = "PASS" if ch.get("pass") else "FAIL"
        extra = ""
        if "delta_rmse" in ch:
            extra = f" Δrmse={ch['delta_rmse']:.2e}"
        lines.append(f"- [{st}] {ch['check']}{extra}")
    if audit["issues"]:
        lines += ["", "## Issues"]
        lines += [f"- {i}" for i in audit["issues"]]
    else:
        lines += ["", "No issues."]
    (OUT / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _setup_style()

    print("Loading slots from DONE.json ...")
    slots = load_slots()
    print("Loading per-gene CSVs ...")
    genes = load_per_gene()
    summary = summarize_slots(slots)
    reports = load_reports()
    gene_sum = summarize_genes(genes)

    print("Auditing consistency ...")
    aud = audit(slots, summary, reports, genes)

    print("Writing tables ...")
    write_display_tables(summary, slots)
    write_gene_tables(gene_sum)
    genes.to_csv(OUT / "per_gene_all_reps_full.csv", index=False)

    print("Generating figures ...")
    fig_rmse_by_rate(summary)
    fig_mae_by_rate(summary)
    fig_boxplot_slots(slots)
    fig_gene_bars(gene_sum)
    fig_mcar_mar_delta(summary)

    consistency = verify_figure_table_consistency(summary)
    pd.DataFrame(consistency).to_csv(OUT / "consistency_table_figure.csv", index=False)
    write_captions()
    write_audit_md(aud, consistency)

    (OUT / "audit.json").write_text(json.dumps(aud, indent=2), encoding="utf-8")

    # Console summary
    print("\n" + "=" * 64)
    print("OriginalRFECA paper package regenerated")
    print(f"Output: {OUT}")
    print(f"Audit: {aud['n_pass']}/{aud['n_checks']} PASS")
    if aud["issues"]:
        print("ISSUES:")
        for i in aud["issues"]:
            print(f"  - {i}")
    print("\nDisplay RMSE (3 d.p.):")
    for _, r in summary.iterrows():
        print(
            f"  {r['mechanism'].upper():4s} {int(r['rate_pct']):2d}%  "
            f"{_fmt(r['rmse_mean'], ROUND_RMSE)} ± {_fmt(r['rmse_std'], ROUND_RMSE)}"
        )
    fig_ok = all(c["table_figure_match"] and c["rounding_consistent"] for c in consistency)
    print(f"\nTable-figure consistency: {'PASS' if fig_ok else 'FAIL'}")
    print("=" * 64)
    return 0 if aud["n_fail"] == 0 and fig_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
