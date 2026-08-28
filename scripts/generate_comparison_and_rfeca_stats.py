#!/usr/bin/env python3
"""
Multi-imputer comparison figures + RFECA post-hoc stats.

Comparison figures (``comparison/``) use:
  Mean, KNN, MissForest, and RFECA (= OriginalRFECA TARGET-WISE).
Legacy RFECA-k5/k10/k20 are excluded from comparison figures.

RFECA RMSE comes from the TARGET-WISE freeze (mask-holdout, 5 reps, seed v2)
at rates 5/10/20/30%. Mean/KNN/MissForest come from the six-imputer METABRIC
campaign (CV, 10 reps).

PAM50 Macro-F1: EnsembleSoft for Mean/KNN/MissForest (imputer-within-CV) and
RFECA (post–TARGET-WISE identity CV). Footnotes document the nesting difference.

Stats (``stats/``) still document Wilcoxon/Holm on the six-imputer campaign
(including legacy RFECA_SVR(k=*)), separate from the comparison figure set.

Usage (repo root):
  python scripts/generate_comparison_and_rfeca_stats.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "artifacts" / "paper_results_original_rfeca"
OUT_CMP = OUT_ROOT / "comparison"
OUT_STATS = OUT_ROOT / "stats"
STATS_DIR = ROOT / "artifacts" / "stats_mcar_mar_20260727_160425"
ORIG_ART = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
CLF_ART = ORIG_ART / "classification"
PAPER = ROOT / "artifacts" / "paper_results"

# Display names in comparison figures
BASELINE_METHODS = ["Mean", "KNN", "MissForest"]
RFECA_LABEL = "RFECA"  # OriginalRFECA TARGET-WISE, displayed as RFECA
RMSE_METHODS = ["Mean", "KNN", "MissForest", "RFECA"]
F1_METHODS = ["Mean", "KNN", "MissForest", "RFECA"]
COMPARE_RATES = [0.05, 0.10, 0.20, 0.30]

METHOD_COLORS = {
    "Mean": "#000000",
    "KNN": "#E69F00",
    "MissForest": "#D55E00",
    "RFECA": "#0072B2",
}
METHOD_MARKERS = {
    "Mean": "o",
    "KNN": "s",
    "MissForest": "P",
    "RFECA": "D",
}

FOOTNOTE_CLF = (
    "RFECA = post–TARGET-WISE classification (identity imputer in CV). "
    "Mean/KNN/MissForest = imputer-within-CV. EnsembleSoft. RFECA-k* excluded."
)
FOOTNOTE_RMSE = (
    "RFECA = OriginalRFECA TARGET-WISE (mask-holdout, 5 reps, seed v2). "
    "Mean/KNN/MissForest: shared-mask CV (10 reps). RFECA-k5/10/20 excluded."
)


def _load_paper_module():
    path = ROOT / "scripts" / "generate_paper_figures_tables.py"
    spec = importlib.util.spec_from_file_location("paper_gen", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _round3(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{math.floor(float(x) * 1000 + 0.5) / 1000:.3f}"


def load_original_rfeca_summary() -> pd.DataFrame:
    """OriginalRFECA TARGET-WISE slot summaries, labeled as RFECA for display."""
    rows = []
    for mech in ("mcar", "mar"):
        for rate in COMPARE_RATES:
            pct = int(round(rate * 100))
            p = ORIG_ART / f"REPORT_{mech.upper()}_{pct}_5REPS.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            n = 5
            mean = float(r["rmse_mean"])
            std = float(r["rmse_std"])
            se = std / math.sqrt(n)
            tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
            rows.append(
                {
                    "method": RFECA_LABEL,
                    "source_method": "OriginalRFECA",
                    "mechanism": mech.upper(),
                    "missing_rate": rate,
                    "rmse_mean": mean,
                    "rmse_std": std,
                    "rmse_ci95_low": mean - tcrit * se,
                    "rmse_ci95_high": mean + tcrit * se,
                    "mae_mean": float(r["mae_mean"]),
                    "mae_std": float(r["mae_std"]),
                    "n_reps": n,
                    "protocol": "target_wise_mask_holdout_v2",
                }
            )
    return pd.DataFrame(rows)


def load_original_rfeca_classification(
    model: str = "EnsembleSoft",
) -> pd.DataFrame:
    """
    RFECA PAM50 Macro-F1 (post–TARGET-WISE).

    Aggregates fold means within each replicate, then mean/CI across 5 reps.
    """
    raw_path = CLF_ART / "exp2_classification_raw.csv"
    if not raw_path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(raw_path)
    raw = raw[raw["model"] == model].copy()
    if raw.empty:
        return pd.DataFrame()

    # fold → replication mean
    rep = (
        raw.groupby(["mechanism", "missing_rate", "replicate"], as_index=False)[
            ["f1_macro", "bal_acc"]
        ]
        .mean()
    )
    rows = []
    for (mech, rate), g in rep.groupby(["mechanism", "missing_rate"]):
        n = len(g)
        f1_mean = float(g["f1_macro"].mean())
        f1_std = float(g["f1_macro"].std(ddof=1)) if n > 1 else 0.0
        bal_mean = float(g["bal_acc"].mean())
        bal_std = float(g["bal_acc"].std(ddof=1)) if n > 1 else 0.0
        se = f1_std / math.sqrt(n) if n > 0 else 0.0
        tcrit = float(scipy_stats.t.ppf(0.975, df=max(n - 1, 1))) if n > 1 else 0.0
        rows.append(
            {
                "method": RFECA_LABEL,
                "source_method": "OriginalRFECA",
                "mechanism": str(mech).upper(),
                "missing_rate": float(rate),
                "model": model,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_ci95_low": f1_mean - tcrit * se,
                "f1_ci95_high": f1_mean + tcrit * se,
                "bal_mean": bal_mean,
                "bal_std": bal_std,
                "n_reps": n,
                "protocol": "post_target_wise_impute",
            }
        )
    return pd.DataFrame(rows)


def anchor_rfeca_clf_at_complete_data(
    rfeca_clf: pd.DataFrame,
    agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepend Macro-F1 at 0% missingness for RFECA.

    At rate 0 there is nothing to impute, so all imputers share the complete-data
    EnsembleSoft F1 from the six-imputer campaign (identical across Mean/KNN/MF).
    """
    if rfeca_clf.empty:
        return rfeca_clf
    complete = agg[
        (agg.dataset == "METABRIC")
        & (agg.metric == "f1_macro")
        & np.isclose(agg.missing_rate, 0.0)
        & (agg.method == "Mean")
    ]
    if complete.empty:
        return rfeca_clf

    rows = []
    for _, r in complete.iterrows():
        rows.append(
            {
                "method": RFECA_LABEL,
                "source_method": "OriginalRFECA",
                "mechanism": str(r["mechanism"]).upper(),
                "missing_rate": 0.0,
                "model": "EnsembleSoft",
                "f1_mean": float(r["mean"]),
                "f1_std": float(r["std"]),
                "f1_ci95_low": float(r["ci95_low"]),
                "f1_ci95_high": float(r["ci95_high"]),
                "bal_mean": np.nan,
                "bal_std": np.nan,
                "n_reps": int(r.get("n_replications", 10)),
                "protocol": "complete_data_shared_baseline",
            }
        )
    out = pd.concat([pd.DataFrame(rows), rfeca_clf], ignore_index=True)
    return out.sort_values(["mechanism", "missing_rate"]).reset_index(drop=True)


def copy_non_figure_paper_artifacts() -> list[str]:
    """Copy tables/captions from paper_results (not figures — those are regenerated)."""
    copied = []
    for name in [
        "table01_compact_metabric.tex",
        "table01_metabric_mcar.tex",
        "table02_metabric_mar.tex",
        "table03_statistical_summary.tex",
        "table06_metabric_classification.tex",
        "table06_metabric_classification.csv",
        "full_pairwise_statistics.csv",
        "primary_contrasts_statistics.csv",
        "figure_table_captions.md",
    ]:
        src = PAPER / name
        if src.exists():
            shutil.copy2(src, OUT_CMP / name)
            copied.append(name)
    return copied


def _line_metric_panels(
    agg: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    stem: str,
    methods: list[str],
    rfeca: pd.DataFrame | None = None,
    rfeca_clf: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
    footnote: str | None = None,
) -> None:
    """METABRIC MCAR/MAR line panels; optional RFECA RMSE / F1 overlay series."""
    sub = agg[(agg.dataset == "METABRIC") & (agg.metric == metric)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        sm = sub[sub.mechanism == mech]
        for method in methods:
            if method == RFECA_LABEL:
                continue  # drawn from rfeca / rfeca_clf frames
            m = sm[sm.method == method]
            if m.empty:
                continue
            x = m["missing_rate"].to_numpy() * 100
            y = m["mean"].to_numpy()
            lo = m["ci95_low"].to_numpy()
            hi = m["ci95_high"].to_numpy()
            ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4,
                linewidth=1.2,
                label=method,
            )
            ax.fill_between(x, lo, hi, color=METHOD_COLORS[method], alpha=0.18, linewidth=0)
        if rfeca is not None and metric == "rmse" and not rfeca.empty:
            o = rfeca[rfeca.mechanism == mech].sort_values("missing_rate")
            if not o.empty:
                ax.plot(
                    o["missing_rate"] * 100,
                    o["rmse_mean"],
                    color=METHOD_COLORS[RFECA_LABEL],
                    marker=METHOD_MARKERS[RFECA_LABEL],
                    markersize=5,
                    linewidth=1.4,
                    label=RFECA_LABEL,
                )
                ax.fill_between(
                    o["missing_rate"] * 100,
                    o["rmse_ci95_low"],
                    o["rmse_ci95_high"],
                    color=METHOD_COLORS[RFECA_LABEL],
                    alpha=0.18,
                    linewidth=0,
                )
        if rfeca_clf is not None and metric == "f1_macro" and not rfeca_clf.empty:
            o = rfeca_clf[rfeca_clf.mechanism == mech].sort_values("missing_rate")
            if not o.empty:
                ax.plot(
                    o["missing_rate"] * 100,
                    o["f1_mean"],
                    color=METHOD_COLORS[RFECA_LABEL],
                    marker=METHOD_MARKERS[RFECA_LABEL],
                    markersize=5,
                    linewidth=1.4,
                    label=RFECA_LABEL,
                )
                ax.fill_between(
                    o["missing_rate"] * 100,
                    o["f1_ci95_low"],
                    o["f1_ci95_high"],
                    color=METHOD_COLORS[RFECA_LABEL],
                    alpha=0.18,
                    linewidth=0,
                )
        ax.set_title(mech)
        ax.set_xlabel("Missingness (%)")
        ax.set_xticks([0, 5, 10, 20, 30] if metric == "f1_macro" else [5, 10, 20, 30])
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.10),
    )
    if footnote:
        fig.text(0.5, -0.04, footnote, ha="center", fontsize=6.5, style="italic", color="#444")
    _save(fig, OUT_CMP, stem)


def fig_rmse_vs_f1(agg: pd.DataFrame, rfeca: pd.DataFrame, rfeca_clf: pd.DataFrame) -> None:
    """RMSE vs Macro-F1 for Mean/KNN/MissForest + RFECA when classification exists."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        for method in BASELINE_METHODS:
            rmse = agg[
                (agg.dataset == "METABRIC")
                & (agg.mechanism == mech)
                & (agg.metric == "rmse")
                & (agg.method == method)
                & (agg.missing_rate > 0)
            ]
            f1 = agg[
                (agg.dataset == "METABRIC")
                & (agg.mechanism == mech)
                & (agg.metric == "f1_macro")
                & (agg.method == method)
                & (agg.missing_rate > 0)
            ]
            m = rmse.merge(
                f1,
                on=["missing_rate"],
                suffixes=("_rmse", "_f1"),
            )
            if m.empty:
                continue
            ax.scatter(
                m["mean_rmse"],
                m["mean_f1"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                s=36,
                label=method,
                zorder=3,
            )
            for _, r in m.iterrows():
                ax.annotate(
                    f"{int(round(r['missing_rate'] * 100))}",
                    (r["mean_rmse"], r["mean_f1"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                    color=METHOD_COLORS[method],
                )
        if not rfeca.empty and not rfeca_clf.empty:
            o = rfeca[rfeca.mechanism == mech].merge(
                rfeca_clf[rfeca_clf.mechanism == mech],
                on=["missing_rate"],
                suffixes=("_rmse", "_f1"),
            )
            if not o.empty:
                ax.scatter(
                    o["rmse_mean"],
                    o["f1_mean"],
                    color=METHOD_COLORS[RFECA_LABEL],
                    marker=METHOD_MARKERS[RFECA_LABEL],
                    s=42,
                    label=RFECA_LABEL,
                    zorder=4,
                )
                for _, r in o.iterrows():
                    ax.annotate(
                        f"{int(round(r['missing_rate'] * 100))}",
                        (r["rmse_mean"], r["f1_mean"]),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=6,
                        color=METHOD_COLORS[RFECA_LABEL],
                    )
        ax.set_title(mech)
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.text(
        0.5,
        -0.02,
        FOOTNOTE_CLF,
        ha="center",
        fontsize=6.5,
        style="italic",
        color="#444",
    )
    _save(fig, OUT_CMP, "fig04_rmse_vs_macrof1")


def fig_classification_bars(agg: pd.DataFrame, rfeca_clf: pd.DataFrame) -> None:
    """Grouped Macro-F1 at 5/10/20/30% — Mean / KNN / MissForest / RFECA."""
    rates = list(COMPARE_RATES)
    methods = list(F1_METHODS)
    sub = agg[
        (agg.dataset == "METABRIC")
        & (agg.metric == "f1_macro")
        & (agg.missing_rate.isin(rates))
        & (agg.method.isin(BASELINE_METHODS))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    x0 = np.arange(len(rates))
    width = 0.18
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        sm = sub[sub.mechanism == mech]
        for i, method in enumerate(methods):
            vals, errs = [], []
            for rate in rates:
                if method == RFECA_LABEL:
                    row = rfeca_clf[
                        (rfeca_clf.mechanism == mech)
                        & (np.isclose(rfeca_clf.missing_rate, rate))
                    ]
                    if row.empty:
                        vals.append(np.nan)
                        errs.append(0.0)
                    else:
                        vals.append(float(row["f1_mean"].iloc[0]))
                        errs.append(
                            0.5
                            * (
                                float(row["f1_ci95_high"].iloc[0])
                                - float(row["f1_ci95_low"].iloc[0])
                            )
                        )
                else:
                    row = sm[(sm.method == method) & (np.isclose(sm.missing_rate, rate))]
                    if row.empty:
                        vals.append(np.nan)
                        errs.append(0.0)
                    else:
                        vals.append(float(row["mean"].iloc[0]))
                        lo = float(row["ci95_low"].iloc[0])
                        hi = float(row["ci95_high"].iloc[0])
                        errs.append(0.5 * (hi - lo))
            ax.bar(
                x0 + (i - 1.5) * width,
                vals,
                width=width,
                color=METHOD_COLORS[method],
                yerr=errs,
                capsize=1.5,
                label=method,
                error_kw={"linewidth": 0.6},
            )
        ax.set_xticks(x0)
        ax.set_xticklabels([f"{int(r * 100)}%" for r in rates])
        ax.set_title(mech)
        ax.set_xlabel("Missingness")
    axes[0].set_ylabel("Macro-F1")
    # ylim from available values
    lows, highs = [], []
    if not sub.empty:
        lows.append(float(sub["ci95_low"].min()))
        highs.append(float(sub["ci95_high"].max()))
    if not rfeca_clf.empty:
        lows.append(float(rfeca_clf["f1_ci95_low"].min()))
        highs.append(float(rfeca_clf["f1_ci95_high"].max()))
    if lows:
        axes[0].set_ylim(max(0.8, min(lows) - 0.005), min(1.0, max(highs) + 0.005))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.text(
        0.5,
        -0.02,
        FOOTNOTE_CLF,
        ha="center",
        fontsize=6.5,
        style="italic",
        color="#444",
    )
    _save(fig, OUT_CMP, "fig06_metabric_macrof1_bars_5_10_20_30")
    for ext in (".png", ".pdf"):
        src = OUT_CMP / f"fig06_metabric_macrof1_bars_5_10_20_30{ext}"
        dst = OUT_CMP / f"fig06_metabric_macrof1_bars_10_20_30{ext}"
        if src.exists():
            shutil.copy2(src, dst)


def fig_rmse_bars(agg: pd.DataFrame, rfeca: pd.DataFrame) -> None:
    """RMSE bars at 5/10/20/30 including RFECA (= OriginalRFECA)."""
    rates = list(COMPARE_RATES)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    x0 = np.arange(len(rates))
    width = 0.18
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        for i, method in enumerate(RMSE_METHODS):
            vals, errs = [], []
            for rate in rates:
                if method == RFECA_LABEL:
                    row = rfeca[
                        (rfeca.mechanism == mech) & (np.isclose(rfeca.missing_rate, rate))
                    ]
                    if row.empty:
                        vals.append(np.nan)
                        errs.append(0.0)
                    else:
                        vals.append(float(row["rmse_mean"].iloc[0]))
                        errs.append(
                            0.5
                            * (
                                float(row["rmse_ci95_high"].iloc[0])
                                - float(row["rmse_ci95_low"].iloc[0])
                            )
                        )
                else:
                    row = agg[
                        (agg.dataset == "METABRIC")
                        & (agg.mechanism == mech)
                        & (agg.metric == "rmse")
                        & (agg.method == method)
                        & (np.isclose(agg.missing_rate, rate))
                    ]
                    if row.empty:
                        vals.append(np.nan)
                        errs.append(0.0)
                    else:
                        vals.append(float(row["mean"].iloc[0]))
                        errs.append(
                            0.5
                            * (
                                float(row["ci95_high"].iloc[0])
                                - float(row["ci95_low"].iloc[0])
                            )
                        )
            ax.bar(
                x0 + (i - 1.5) * width,
                vals,
                width=width,
                color=METHOD_COLORS[method],
                yerr=errs,
                capsize=1.5,
                label=method,
                error_kw={"linewidth": 0.6},
            )
        ax.set_xticks(x0)
        ax.set_xticklabels([f"{int(r * 100)}%" for r in rates])
        ax.set_title(mech)
        ax.set_xlabel("Missingness")
    axes[0].set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.text(
        0.5,
        -0.02,
        FOOTNOTE_RMSE,
        ha="center",
        fontsize=6.5,
        style="italic",
        color="#444",
    )
    _save(fig, OUT_CMP, "fig05_metabric_rmse_bars_5_10_20_30")
    for ext in (".png", ".pdf"):
        src = OUT_CMP / f"fig05_metabric_rmse_bars_5_10_20_30{ext}"
        dst = OUT_CMP / f"fig05_metabric_rmse_bars_10_20_30{ext}"
        if src.exists():
            shutil.copy2(src, dst)

def regenerate_comparison_figures(
    agg: pd.DataFrame,
    rfeca: pd.DataFrame,
    rfeca_clf: pd.DataFrame,
) -> None:
    """Replace RFECA-k* with RFECA (OriginalRFECA) in comparison figures."""
    _line_metric_panels(
        agg,
        metric="rmse",
        ylabel="RMSE",
        stem="fig01_metabric_rmse_by_missingness",
        methods=RMSE_METHODS,
        rfeca=rfeca,
        footnote=FOOTNOTE_RMSE,
    )
    _line_metric_panels(
        agg,
        metric="corr_rv",
        ylabel="RV coefficient",
        stem="fig02_metabric_rv_by_missingness",
        methods=BASELINE_METHODS,
        footnote="RFECA-k5/10/20 excluded. RFECA (TARGET-WISE) has no RV metric in this freeze.",
    )
    f1 = agg[
        (agg.dataset == "METABRIC")
        & (agg.metric == "f1_macro")
        & (agg.missing_rate > 0)
        & (agg.method.isin(BASELINE_METHODS))
    ]
    lows = [float(f1["ci95_low"].min())] if not f1.empty else []
    highs = [float(f1["ci95_high"].max())] if not f1.empty else []
    if not rfeca_clf.empty:
        lows.append(float(rfeca_clf["f1_ci95_low"].min()))
        highs.append(float(rfeca_clf["f1_ci95_high"].max()))
    ymin = max(0.80, min(lows) - 0.005) if lows else 0.80
    ymax = min(1.0, max(highs) + 0.005) if highs else 1.0
    _line_metric_panels(
        agg,
        metric="f1_macro",
        ylabel="Macro-F1",
        stem="fig03_metabric_macrof1_by_missingness",
        methods=F1_METHODS,
        rfeca_clf=rfeca_clf,
        ylim=(ymin, ymax),
        footnote=FOOTNOTE_CLF,
    )
    fig_rmse_vs_f1(agg, rfeca, rfeca_clf)
    fig_rmse_bars(agg, rfeca)
    fig_classification_bars(agg, rfeca_clf)


def build_comparison_tables(
    agg: pd.DataFrame,
    rfeca: pd.DataFrame,
    rfeca_clf: pd.DataFrame,
) -> None:
    """CSV tables: baselines + RFECA (OriginalRFECA), no RFECA-k*."""
    rates = list(COMPARE_RATES)
    rows = []
    for mech in ("MCAR", "MAR"):
        for rate in rates:
            row: dict = {
                "mechanism": mech,
                "missing_rate": rate,
                "rate_pct": int(rate * 100),
            }
            for method in BASELINE_METHODS:
                for metric, key in [("rmse", "rmse"), ("f1_macro", "f1")]:
                    m = agg[
                        (agg.dataset == "METABRIC")
                        & (agg.mechanism == mech)
                        & (agg.metric == metric)
                        & (agg.method == method)
                        & (np.isclose(agg.missing_rate, rate))
                    ]
                    if m.empty:
                        row[f"{method}_{key}_mean"] = np.nan
                        row[f"{method}_{key}_std"] = np.nan
                    else:
                        row[f"{method}_{key}_mean"] = float(m["mean"].iloc[0])
                        row[f"{method}_{key}_std"] = float(m["std"].iloc[0])
            o = rfeca[(rfeca.mechanism == mech) & (np.isclose(rfeca.missing_rate, rate))]
            if o.empty:
                row["RFECA_rmse_mean"] = np.nan
                row["RFECA_rmse_std"] = np.nan
            else:
                row["RFECA_rmse_mean"] = float(o["rmse_mean"].iloc[0])
                row["RFECA_rmse_std"] = float(o["rmse_std"].iloc[0])
            c = rfeca_clf[
                (rfeca_clf.mechanism == mech) & (np.isclose(rfeca_clf.missing_rate, rate))
            ] if not rfeca_clf.empty else pd.DataFrame()
            if c.empty:
                row["RFECA_f1_mean"] = np.nan
                row["RFECA_f1_std"] = np.nan
            else:
                row["RFECA_f1_mean"] = float(c["f1_mean"].iloc[0])
                row["RFECA_f1_std"] = float(c["f1_std"].iloc[0])
            row["RFECA_source"] = "OriginalRFECA_TARGET_WISE"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CMP / "table_methods_rmse_f1_full.csv", index=False)

    disp_rows = []
    for _, r in df.iterrows():
        d = {"mechanism": r.mechanism, "rate_pct": int(r.rate_pct)}
        for method in BASELINE_METHODS:
            d[f"{method}_RMSE"] = (
                f"{_round3(r[f'{method}_rmse_mean'])} +/- {_round3(r[f'{method}_rmse_std'])}"
            )
            d[f"{method}_F1"] = (
                f"{_round3(r[f'{method}_f1_mean'])} +/- {_round3(r[f'{method}_f1_std'])}"
            )
        d["RFECA_RMSE"] = (
            f"{_round3(r['RFECA_rmse_mean'])} +/- {_round3(r['RFECA_rmse_std'])}"
        )
        d["RFECA_F1"] = (
            f"{_round3(r['RFECA_f1_mean'])} +/- {_round3(r['RFECA_f1_std'])}"
        )
        disp_rows.append(d)
    pd.DataFrame(disp_rows).to_csv(OUT_CMP / "table_methods_rmse_f1_display.csv", index=False)

    compact = []
    for mech in ("MCAR", "MAR"):
        for rate in rates:
            r = df[(df.mechanism == mech) & (np.isclose(df.missing_rate, rate))]
            if r.empty:
                continue
            r = r.iloc[0]
            compact.append(
                {
                    "mechanism": mech,
                    "rate_pct": int(rate * 100),
                    "Mean": _round3(r["Mean_rmse_mean"]),
                    "KNN": _round3(r["KNN_rmse_mean"]),
                    "MissForest": _round3(r["MissForest_rmse_mean"]),
                    "RFECA": _round3(r["RFECA_rmse_mean"]),
                }
            )
    pd.DataFrame(compact).to_csv(OUT_CMP / "table_rmse_compact_with_rfeca.csv", index=False)
    pd.DataFrame(compact).to_csv(OUT_CMP / "table_rmse_compact_with_original.csv", index=False)

    compact_f1 = []
    for mech in ("MCAR", "MAR"):
        for rate in rates:
            r = df[(df.mechanism == mech) & (np.isclose(df.missing_rate, rate))]
            if r.empty:
                continue
            r = r.iloc[0]
            compact_f1.append(
                {
                    "mechanism": mech,
                    "rate_pct": int(rate * 100),
                    "Mean": _round3(r["Mean_f1_mean"]),
                    "KNN": _round3(r["KNN_f1_mean"]),
                    "MissForest": _round3(r["MissForest_f1_mean"]),
                    "RFECA": _round3(r["RFECA_f1_mean"]),
                }
            )
    pd.DataFrame(compact_f1).to_csv(OUT_CMP / "table_f1_compact_with_rfeca.csv", index=False)


def export_rfeca_stats() -> dict:
    """Filter / regenerate RFECA-focused Wilcoxon+Holm+Friedman tables."""
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    pairwise = pd.read_csv(STATS_DIR / "pairwise_all.csv")
    primary = pd.read_csv(STATS_DIR / "primary_contrasts.csv")
    friedman_all = pd.read_csv(STATS_DIR / "friedman_all_imputers.csv")
    friedman_red = pd.read_csv(STATS_DIR / "friedman_reduced_imputers.csv")

    pair_m = pairwise[pairwise["cohort"] == "metabric"].copy()
    prim_m = primary[primary["cohort"] == "metabric"].copy()

    rfeca_names = ["RFECA_SVR(k=5)", "RFECA_SVR(k=10)", "RFECA_SVR(k=20)"]
    mask = pair_m["method_a"].isin(rfeca_names) | pair_m["method_b"].isin(rfeca_names)
    rfeca_pair = pair_m[mask].copy()
    rfeca_pair.to_csv(OUT_STATS / "rfeca_pairwise_wilcoxon_holm.csv", index=False)

    prim_rfeca = prim_m[
        prim_m["method_a"].isin(rfeca_names) | prim_m["method_b"].isin(rfeca_names)
    ].copy()
    prim_rfeca.to_csv(OUT_STATS / "rfeca_primary_contrasts.csv", index=False)

    vs_mean = prim_m[
        (
            (prim_m["method_a"].isin(rfeca_names) & (prim_m["method_b"] == "SimpleMean"))
            | (prim_m["method_b"].isin(rfeca_names) & (prim_m["method_a"] == "SimpleMean"))
        )
    ].copy()
    vs_mean.to_csv(OUT_STATS / "rfeca_vs_mean_primary.csv", index=False)

    pcol = "p_holm_primary_family"
    sig = vs_mean[vs_mean[pcol] < 0.05].copy()
    sig.to_csv(OUT_STATS / "rfeca_vs_mean_holm05.csv", index=False)

    friedman_all[friedman_all.cohort == "metabric"].to_csv(
        OUT_STATS / "friedman_metabric_all.csv", index=False
    )
    friedman_red[friedman_red.cohort == "metabric"].to_csv(
        OUT_STATS / "friedman_metabric_reduced.csv", index=False
    )

    display_rows = []
    for _, r in vs_mean.iterrows():
        if "RFECA_SVR(k=20)" not in (r["method_a"], r["method_b"]):
            continue
        raw_delta = float(r["delta_a_minus_b"])
        if r["method_a"] == "RFECA_SVR(k=20)" and r["method_b"] == "SimpleMean":
            dlt = raw_delta
        elif r["method_a"] == "SimpleMean" and r["method_b"] == "RFECA_SVR(k=20)":
            dlt = -raw_delta
        else:
            continue
        p_h = float(r[pcol])
        display_rows.append(
            {
                "mechanism": str(r["mechanism"]).upper(),
                "metric": r["metric"],
                "missing_rate": float(r["missing_rate"]),
                "rate_pct": int(round(float(r["missing_rate"]) * 100)),
                "contrast": "RFECA-k20 - Mean",
                "mean_delta": dlt,
                "delta_ci95_low": float(r["delta_ci95_low"])
                if r["method_a"] == "RFECA_SVR(k=20)"
                else -float(r["delta_ci95_high"]),
                "delta_ci95_high": float(r["delta_ci95_high"])
                if r["method_a"] == "RFECA_SVR(k=20)"
                else -float(r["delta_ci95_low"]),
                "p_value": float(r["p_value"]),
                "p_holm": p_h,
                "significant_holm05": bool(np.isfinite(p_h) and p_h < 0.05),
                "r_rb": float(r["rank_biserial_r"]),
            }
        )
    disp = pd.DataFrame(display_rows)
    disp.to_csv(OUT_STATS / "rfeca_k20_vs_mean_display.csv", index=False)

    summary = {
        "n_rfeca_pairwise": int(len(rfeca_pair)),
        "n_rfeca_primary": int(len(prim_rfeca)),
        "n_rfeca_vs_mean": int(len(vs_mean)),
        "n_rfeca_vs_mean_holm05": int(len(sig)),
        "n_k20_vs_mean_rows": int(len(disp)),
    }
    return {
        "summary": summary,
        "k20_vs_mean": disp,
        "pairwise": rfeca_pair,
        "primary": prim_rfeca,
    }


def fig_rfeca_holm_heatmap(pairwise: pd.DataFrame) -> None:
    """Heatmap of Holm p-values for RFECA-k20 vs each other method (RMSE & F1)."""
    target = "RFECA_SVR(k=20)"
    others = [
        "SimpleMean",
        "KNN(k=5,dist)",
        "RFECA_SVR(k=5)",
        "RFECA_SVR(k=10)",
        "MissForest",
    ]
    label = {
        "SimpleMean": "Mean",
        "KNN(k=5,dist)": "KNN",
        "RFECA_SVR(k=5)": "RFECA-k5",
        "RFECA_SVR(k=10)": "RFECA-k10",
        "MissForest": "MissForest",
    }
    rates = [0.05, 0.10, 0.20, 0.30]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    im = None
    for ax, (mech, metric) in zip(
        axes.ravel(),
        [
            ("mcar", "rmse"),
            ("mar", "rmse"),
            ("mcar", "f1_macro"),
            ("mar", "f1_macro"),
        ],
    ):
        mat = np.full((len(others), len(rates)), np.nan)
        for i, other in enumerate(others):
            for j, rate in enumerate(rates):
                sub = pairwise[
                    (pairwise.mechanism.str.lower() == mech)
                    & (pairwise.metric == metric)
                    & (np.isclose(pairwise.missing_rate, rate))
                    & (
                        ((pairwise.method_a == target) & (pairwise.method_b == other))
                        | ((pairwise.method_b == target) & (pairwise.method_a == other))
                    )
                ]
                if not sub.empty:
                    mat[i, j] = float(sub.iloc[0]["p_holm"])
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=0, vmax=0.1)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{int(r*100)}%" for r in rates])
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels([label[o] for o in others], fontsize=7)
        ax.set_title(f"{mech.upper()} · {metric}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v:.3f}*" if v < 0.05 else f"{v:.3f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="p Holm")
    fig.suptitle("RFECA-k20 vs others — Holm-adjusted Wilcoxon p", y=1.02)
    _save(fig, OUT_STATS, "fig_rfeca_k20_holm_heatmap")


def fig_rfeca_delta_forest(disp: pd.DataFrame) -> None:
    """Forest-style Δ for RFECA-k20 vs Mean (RMSE and F1)."""
    if disp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, metric, title in zip(
        axes,
        ["rmse", "f1_macro"],
        ["ΔRMSE (RFECA-k20 − Mean)", "ΔMacro-F1 (RFECA-k20 − Mean)"],
    ):
        sub = disp[disp.metric == metric].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values(["mechanism", "rate_pct"])
        y = np.arange(len(sub))
        colors = ["#009E73" if s else "#999999" for s in sub["significant_holm05"]]
        ax.barh(y, sub["mean_delta"], color=colors, height=0.7)
        ax.axvline(0, color="black", linewidth=0.7)
        labels = [f"{m} {p}%" for m, p in zip(sub.mechanism, sub.rate_pct)]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(title)
        for yi, (dlt, sig) in enumerate(zip(sub.mean_delta, sub.significant_holm05)):
            star = "*" if sig else ""
            ax.text(
                dlt,
                yi,
                f" {dlt:+.4f}{star}",
                va="center",
                fontsize=6,
                ha="left" if dlt >= 0 else "right",
            )
    fig.suptitle("Primary contrasts vs Mean (Holm * p<0.05)", y=1.02)
    _save(fig, OUT_STATS, "fig_rfeca_k20_vs_mean_deltas")


def regenerate_paper_package(mod) -> None:
    """Refresh artifacts/paper_results via the locked generator (source for baselines)."""
    print("Refreshing artifacts/paper_results (baseline package)...")
    rc = mod.main()
    if rc != 0:
        raise RuntimeError(f"generate_paper_figures_tables failed with {rc}")


def write_readme(copied: list[str], stats_summary: dict) -> None:
    text = f"""# Comparison + RFECA stats package

Generated: `{datetime.now(timezone.utc).isoformat()}`

## Layout
- `comparison/` — METABRIC figures with **Mean, KNN, MissForest, RFECA**
  (RFECA = OriginalRFECA TARGET-WISE; **RFECA-k5/k10/k20 excluded**)
- `stats/` — Wilcoxon/Holm/Friedman on the six-imputer campaign (legacy RFECA_SVR(k=*))

## Comparison methods
| Display name | Source |
|---|---|
| Mean, KNN, MissForest | Shared-mask CV METABRIC (10 reps); F1 = EnsembleSoft imputer-within-CV |
| **RFECA** | OriginalRFECA TARGET-WISE (mask-holdout, 5 reps); F1 = post-impute identity CV |

## Key figures (`comparison/`)
- `fig01_metabric_rmse_by_missingness` — RMSE lines including **RFECA** (5/10/20/30%)
- `fig02_metabric_rv_by_missingness` — RV (baselines only; no RFECA-k*)
- `fig03_metabric_macrof1_by_missingness` — Macro-F1 including **RFECA**
- `fig04_rmse_vs_macrof1` — baselines + RFECA
- `fig05_metabric_rmse_bars_5_10_20_30` — RMSE bars with **RFECA**
- `fig06_metabric_macrof1_bars_5_10_20_30` — F1 bars with **RFECA**

## Protocol note
RFECA classification uses already-imputed matrices (identity imputer in StratifiedKFold).
Baselines keep imputer-within-CV from the METABRIC full campaign.

## Stats summary
```json
{json.dumps(stats_summary, indent=2)}
```

Copied non-figure paper artifacts: {len(copied)} files.
"""
    (OUT_ROOT / "COMPARISON_AND_STATS_README.md").write_text(text, encoding="utf-8")


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    _setup_style()
    OUT_CMP.mkdir(parents=True, exist_ok=True)
    OUT_STATS.mkdir(parents=True, exist_ok=True)

    # Remove obsolete overlay figure from earlier revision
    for stem in ("fig05_rmse_with_original_rfeca_overlay",):
        for ext in (".png", ".pdf"):
            p = OUT_CMP / f"{stem}{ext}"
            if p.exists():
                p.unlink()

    mod = _load_paper_module()
    regenerate_paper_package(mod)
    copied = copy_non_figure_paper_artifacts()
    print(f"Copied {len(copied)} non-figure paper artifacts into comparison/")

    print("Building replication/aggregate metrics...")
    rep = mod.build_replication_level(mod.MAIN_RUNS)
    agg = mod.aggregate_metrics(rep)
    # Drop legacy RFECA-k* from any comparison aggregates we write
    agg_cmp = agg[~agg["method"].astype(str).str.startswith("RFECA-k")].copy()
    agg_cmp.to_csv(OUT_CMP / "aggregated_metrics.csv", index=False)
    rep_cmp = rep[~rep["method"].astype(str).str.startswith("RFECA-k")].copy()
    rep_cmp.to_csv(OUT_CMP / "replication_level_metrics.csv", index=False)

    rfeca = load_original_rfeca_summary()
    rfeca.to_csv(OUT_CMP / "original_rfeca_summary.csv", index=False)
    rfeca_clf = load_original_rfeca_classification()
    if not rfeca_clf.empty:
        rfeca_clf = anchor_rfeca_clf_at_complete_data(rfeca_clf, agg_cmp)
        rfeca_clf.to_csv(OUT_CMP / "original_rfeca_classification_summary.csv", index=False)
    else:
        print("WARN: RFECA classification summary empty — F1 panels omit RFECA")

    print("Generating comparison figures (Mean/KNN/MissForest + RFECA)...")
    regenerate_comparison_figures(agg_cmp, rfeca, rfeca_clf)
    build_comparison_tables(agg_cmp, rfeca, rfeca_clf)

    print("Exporting RFECA Wilcoxon/Holm/Friedman (six-imputer campaign)...")
    stats_pack = export_rfeca_stats()
    fig_rfeca_holm_heatmap(stats_pack["pairwise"])
    fig_rfeca_delta_forest(stats_pack["k20_vs_mean"])

    (OUT_STATS / "SOURCE_STATS_DIR.txt").write_text(
        str(STATS_DIR.relative_to(ROOT)).replace("\\", "/") + "\n",
        encoding="utf-8",
    )
    shutil.copy2(STATS_DIR / "stats_report.md", OUT_STATS / "stats_report_full.md")

    write_readme(copied, stats_pack["summary"])

    print("\n" + "=" * 64)
    print("Done.")
    print(f"  comparison -> {OUT_CMP}")
    print("  methods in RMSE/F1 figures: Mean, KNN, MissForest, RFECA")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
