#!/usr/bin/env python3
"""
Generate publication figures and tables for the Results section.

Main six-imputer benchmark (legacy seed scheme, shared masks within each run):
  - discovery_full_20260724_172348          (CPTAC 2C MCAR)
  - metabric_full_20260724_185916           (METABRIC MCAR)
  - discovery_full_mar_20260725_045052      (CPTAC 2C MAR)
  - metabric_full_mar_20260725_062517       (METABRIC MAR)
  - stats_mcar_mar_20260727_160425          (consolidated tests)

Inductive RFECA-only runs are handled separately as sensitivity analysis.

Usage (repo root):
  python scripts/generate_paper_figures_tables.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "paper_results"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Final main-benchmark artifacts (do not mix with inductive RFECA-only)
# ---------------------------------------------------------------------------
MAIN_RUNS = [
    {
        "dataset": "CPTAC 2C",
        "mechanism": "MCAR",
        "dir": ROOT / "artifacts" / "discovery_full_20260724_172348",
        "seed_scheme": "legacy",
    },
    {
        "dataset": "METABRIC",
        "mechanism": "MCAR",
        "dir": ROOT / "artifacts" / "metabric_full_20260724_185916",
        "seed_scheme": "legacy",
    },
    {
        "dataset": "CPTAC 2C",
        "mechanism": "MAR",
        "dir": ROOT / "artifacts" / "discovery_full_mar_20260725_045052",
        "seed_scheme": "legacy",
    },
    {
        "dataset": "METABRIC",
        "mechanism": "MAR",
        "dir": ROOT / "artifacts" / "metabric_full_mar_20260725_062517",
        "seed_scheme": "legacy",
    },
]

STATS_DIR = ROOT / "artifacts" / "stats_mcar_mar_20260727_160425"

INDUCTIVE_RUNS = [
    {
        "dataset": "CPTAC 2C",
        "mechanism": "MCAR",
        "dir": ROOT / "artifacts" / "discovery_full_rfeca_20260727_161656",
        "seed_scheme": "v2",
        "protocol": "inductive_rfeca_only",
    },
    {
        "dataset": "METABRIC",
        "mechanism": "MCAR",
        "dir": ROOT / "artifacts" / "metabric_full_rfeca_20260727_163213",
        "seed_scheme": "v2",
        "protocol": "inductive_rfeca_only",
    },
    {
        "dataset": "CPTAC 2C",
        "mechanism": "MAR",
        "dir": ROOT / "artifacts" / "discovery_full_mar_rfeca_20260727_201003",
        "seed_scheme": "v2",
        "protocol": "inductive_rfeca_only",
    },
    {
        "dataset": "METABRIC",
        "mechanism": "MAR",
        "dir": ROOT / "artifacts" / "metabric_full_mar_rfeca_20260727_202556",
        "seed_scheme": "v2",
        "protocol": "inductive_rfeca_only",
    },
]

METHOD_MAP = {
    "SimpleMean": "Mean",
    "KNN(k=5,dist)": "KNN",
    "MissForest": "MissForest",
    "RFECA_SVR(k=5)": "RFECA-k5",
    "RFECA_SVR(k=10)": "RFECA-k10",
    "RFECA_SVR(k=20)": "RFECA-k20",
}
METHOD_ORDER = ["Mean", "KNN", "RFECA-k5", "RFECA-k10", "RFECA-k20", "MissForest"]

# Okabe–Ito colorblind-safe
METHOD_COLORS = {
    "Mean": "#000000",
    "KNN": "#E69F00",
    "RFECA-k5": "#56B4E9",
    "RFECA-k10": "#0072B2",
    "RFECA-k20": "#009E73",
    "MissForest": "#D55E00",
}
METHOD_MARKERS = {
    "Mean": "o",
    "KNN": "s",
    "RFECA-k5": "^",
    "RFECA-k10": "D",
    "RFECA-k20": "v",
    "MissForest": "P",
}

HIGHER_BETTER = {
    "rmse": False,
    "mae": False,
    "corr_rv": True,
    "corr_frobenius_rel": False,
    "f1_macro": True,
    "bal_acc": True,
    "precision_macro": True,
    "recall_macro": True,
}


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


def _load_config_seed_scheme(run_dir: Path) -> str:
    snap = run_dir / "config_snapshot.json"
    if not snap.exists():
        return "unknown"
    cfg = json.loads(snap.read_text(encoding="utf-8"))
    return str(cfg.get("missingness_seed_scheme", "legacy (pre-field)"))


def audit_runs(runs: list[dict]) -> list[dict]:
    print("=" * 72)
    print("AUDIT SUMMARY — main six-imputer benchmark")
    print("=" * 72)
    flags: list[str] = []
    audited = []
    for run in runs:
        d = run["dir"]
        if not d.exists():
            flags.append(f"MISSING DIR: {d}")
            continue
        imp = pd.read_csv(d / "exp1_imputation_raw.csv")
        cls = pd.read_csv(d / "exp2_classification_raw.csv")
        if "model" in cls.columns:
            cls = cls[cls["model"] == "EnsembleSoft"].copy()
        scheme = _load_config_seed_scheme(d)
        info = {
            **run,
            "seed_scheme_recorded": scheme,
            "imp_columns": list(imp.columns),
            "cls_columns": list(cls.columns),
            "methods_raw": sorted(cls["imputer"].unique()),
            "rates": sorted(cls["missing_rate"].unique()),
            "n_reps": int(cls["replicate"].nunique()),
            "n_folds": int(cls["fold"].nunique()),
            "grain": "per fold within replication (raw); aggregate to replication mean for CI",
            "n_cls_rows": int(len(cls)),
            "n_imp_rows": int(len(imp)),
        }
        audited.append(info)
        print(f"\n[{info['dataset']} | {info['mechanism']}]")
        print(f"  path: {d}")
        print(f"  seed scheme: {scheme} (declared legacy for main package)")
        print(f"  methods: {[METHOD_MAP.get(m, m) for m in info['methods_raw']]}")
        print(f"  rates: {info['rates']}")
        print(f"  reps={info['n_reps']} folds={info['n_folds']}")
        print(f"  cls rows={info['n_cls_rows']} imp rows={info['n_imp_rows']}")
        expected_methods = set(METHOD_MAP)
        if set(info["methods_raw"]) != expected_methods:
            flags.append(f"{d.name}: unexpected methods {info['methods_raw']}")
        if info["n_reps"] != 10 or info["n_folds"] != 5:
            flags.append(f"{d.name}: unexpected reps/folds")
        # shared-mask fairness: same seed column per rate×rep across imputers
        g = cls.groupby(["missing_rate", "replicate", "imputer"])["seed"].nunique()
        if (g > 1).any():
            flags.append(f"{d.name}: multiple seeds within imputer×rate×rep")
        seeds = (
            cls.groupby(["missing_rate", "replicate"])["seed"].nunique()
        )
        if (seeds > 1).any():
            flags.append(
                f"{d.name}: methods do not share the same seed within rate×rep "
                "(masks may differ across methods)"
            )

    print("\nStats dir:", STATS_DIR)
    print("  files:", sorted(p.name for p in STATS_DIR.glob("*")))
    if flags:
        print("\nFLAGS / INCONSISTENCIES:")
        for f in flags:
            print("  !", f)
    else:
        print("\nNo critical inconsistencies detected for main six-imputer package.")
    print(
        "\nNOTE: Inductive RFECA-only artifacts use seed scheme v2 and must NOT be "
        "merged into main figures."
    )
    return audited


def _macro_from_per_class(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[cols].mean(axis=1)


def build_replication_level(runs: list[dict]) -> pd.DataFrame:
    """Mean over folds within each replication → one value per experimental unit."""
    rows = []
    for run in runs:
        d = run["dir"]
        imp = pd.read_csv(d / "exp1_imputation_raw.csv")
        cls = pd.read_csv(d / "exp2_classification_raw.csv")
        if "model" in cls.columns:
            cls = cls[cls["model"] == "EnsembleSoft"].copy()
        cls["method"] = cls["imputer"].map(METHOD_MAP)
        imp["method"] = imp["imputer"].map(METHOD_MAP)
        cls["precision_macro"] = _macro_from_per_class(cls, "precision_")
        cls["recall_macro"] = _macro_from_per_class(cls, "recall_")

        cls_m = (
            cls.groupby(
                ["method", "missing_rate", "replicate"], as_index=False
            )[["f1_macro", "bal_acc", "precision_macro", "recall_macro"]]
            .mean()
        )
        for _, r in cls_m.iterrows():
            for metric in ["f1_macro", "bal_acc", "precision_macro", "recall_macro"]:
                rows.append(
                    {
                        "dataset": run["dataset"],
                        "mechanism": run["mechanism"],
                        "missing_rate": float(r["missing_rate"]),
                        "method": r["method"],
                        "replication": int(r["replicate"]),
                        "metric": metric,
                        "value": float(r[metric]),
                        "seed_scheme": run["seed_scheme"],
                        "artifact": d.name,
                    }
                )

        imp_pos = imp[imp["missing_rate"] > 0].copy()
        imp_m = (
            imp_pos.groupby(
                ["method", "missing_rate", "replicate"], as_index=False
            )[["rmse", "mae", "corr_rv", "corr_frobenius_rel", "corr_mae_offdiag"]]
            .mean()
        )
        for _, r in imp_m.iterrows():
            for metric in [
                "rmse",
                "mae",
                "corr_rv",
                "corr_frobenius_rel",
                "corr_mae_offdiag",
            ]:
                if metric not in r or pd.isna(r[metric]):
                    continue
                rows.append(
                    {
                        "dataset": run["dataset"],
                        "mechanism": run["mechanism"],
                        "missing_rate": float(r["missing_rate"]),
                        "method": r["method"],
                        "replication": int(r["replicate"]),
                        "metric": metric,
                        "value": float(r[metric]),
                        "seed_scheme": run["seed_scheme"],
                        "artifact": d.name,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_metrics(rep: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD/median/IQR and t-based 95% CI across replications."""
    rows = []
    keys = ["dataset", "mechanism", "missing_rate", "method", "metric"]
    for key, g in rep.groupby(keys, sort=False):
        vals = g["value"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        mean = float(np.mean(vals)) if n else float("nan")
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        med = float(np.median(vals)) if n else float("nan")
        q1 = float(np.percentile(vals, 25)) if n else float("nan")
        q3 = float(np.percentile(vals, 75)) if n else float("nan")
        if n > 1:
            se = std / math.sqrt(n)
            tcrit = float(stats.t.ppf(0.975, df=n - 1))
            ci_lo = mean - tcrit * se
            ci_hi = mean + tcrit * se
        else:
            ci_lo = ci_hi = float("nan")
        rows.append(
            {
                "dataset": key[0],
                "mechanism": key[1],
                "missing_rate": key[2],
                "method": key[3],
                "metric": key[4],
                "n_replications": n,
                "mean": mean,
                "std": std,
                "median": med,
                "iqr": q3 - q1 if n else float("nan"),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            }
        )
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(
        ["dataset", "mechanism", "metric", "missing_rate", "method"]
    ).reset_index(drop=True)


def _save_fig(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _line_panels(
    agg: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    ylabel: str,
    stem: str,
    ylim: tuple[float, float] | None = None,
    restrict_note: str | None = None,
) -> None:
    sub = agg[(agg["dataset"] == dataset) & (agg["metric"] == metric)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        sm = sub[sub["mechanism"] == mech]
        for method in METHOD_ORDER:
            m = sm[sm["method"] == method]
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
        ax.set_xlabel("Missingness (%)")
        ax.set_title(mech)
        ax.set_xticks([5, 10, 20, 30])
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Method",
    )
    if restrict_note:
        fig.text(0.01, -0.02, restrict_note, fontsize=6, style="italic")
    fig.tight_layout()
    _save_fig(fig, stem)


def fig_rmse_vs_f1(agg: pd.DataFrame) -> dict:
    sub = agg[
        (agg["dataset"] == "METABRIC")
        & (agg["metric"].isin(["rmse", "f1_macro"]))
        & (agg["missing_rate"] > 0)
    ].copy()
    wide = sub.pivot_table(
        index=["mechanism", "method", "missing_rate"],
        columns="metric",
        values="mean",
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, sharex=True)
    rate_annot = {0.05: "5", 0.10: "10", 0.20: "20", 0.30: "30"}
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        sm = wide[wide["mechanism"] == mech]
        for method in METHOD_ORDER:
            m = sm[sm["method"] == method]
            if m.empty:
                continue
            ax.scatter(
                m["rmse"],
                m["f1_macro"],
                c=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                s=28,
                label=method,
                zorder=3,
            )
            for _, r in m.iterrows():
                ax.annotate(
                    rate_annot.get(float(r["missing_rate"]), ""),
                    (r["rmse"], r["f1_macro"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=5.5,
                    color=METHOD_COLORS[method],
                )
        ax.set_title(mech)
        ax.set_xlabel("RMSE")
    axes[0].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Method",
    )
    fig.tight_layout()
    _save_fig(fig, "fig04_rmse_vs_macrof1")

    # Descriptive Spearman on method×rate means (not independent)
    corr = {}
    for label, part in [
        ("overall", wide),
        ("MCAR", wide[wide.mechanism == "MCAR"]),
        ("MAR", wide[wide.mechanism == "MAR"]),
    ]:
        if len(part) >= 3:
            rho, p = stats.spearmanr(part["rmse"], part["f1_macro"])
            corr[label] = {"spearman_rho": float(rho), "p_value": float(p), "n_points": int(len(part))}
    return corr


def _fmt_mean_sd(mean: float, std: float, precision: int = 3) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def _best_mask(series: pd.Series, higher_is_better: bool) -> pd.Series:
    if higher_is_better:
        return series == series.max()
    return series == series.min()


def latex_performance_table(
    agg: pd.DataFrame,
    *,
    dataset: str,
    mechanism: str | None,
    rates: list[float] | None,
    path: Path,
    caption: str,
    label: str,
) -> None:
    sub = agg[
        (agg["dataset"] == dataset)
        & (agg["metric"].isin(["rmse", "corr_rv", "f1_macro"]))
    ].copy()
    if mechanism:
        sub = sub[sub["mechanism"] == mechanism]
    if rates is not None:
        sub = sub[sub["missing_rate"].isin(rates)]

    metrics = ["rmse", "corr_rv", "f1_macro"]
    metric_names = {"rmse": "RMSE", "corr_rv": "RV", "f1_macro": "Macro-F1"}
    methods = [m for m in METHOD_ORDER if m in set(sub["method"].astype(str))]

    lines = [
        "% Auto-generated — do not edit by hand",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{caption} Best value per row in \\textbf{{bold}} "
        f"(lowest RMSE; highest RV and Macro-F1). "
        f"Entries are mean $\\pm$ SD across 10 missingness replications "
        f"(fold-averaged within replication).}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{ll" + "c" * len(methods) + "}",
        "\\hline",
        "Rate & Metric & " + " & ".join(methods) + " \\\\",
        "\\hline",
    ]

    mechs = ["MCAR", "MAR"] if mechanism is None else [mechanism]
    for mech in mechs:
        sm = sub[sub["mechanism"] == mech]
        rate_list = sorted(sm["missing_rate"].unique())
        if mechanism is None:
            lines.append(
                f"\\multicolumn{{{2 + len(methods)}}}{{l}}{{\\textit{{{mech}}}}} \\\\"
            )
        for rate in rate_list:
            rate_printed = False
            for metric in metrics:
                g = sm[(sm["missing_rate"] == rate) & (sm["metric"] == metric)]
                if g.empty:
                    continue
                vals = {
                    str(r["method"]): (float(r["mean"]), float(r["std"]))
                    for _, r in g.iterrows()
                }
                means = pd.Series({m: vals[m][0] for m in methods if m in vals})
                best = _best_mask(means, HIGHER_BETTER[metric])
                cells = []
                for m in methods:
                    if m not in vals:
                        cells.append("---")
                        continue
                    txt = _fmt_mean_sd(*vals[m], precision=3)
                    if bool(best.get(m, False)):
                        txt = f"\\textbf{{{txt}}}"
                    cells.append(txt)
                rate_cell = ""
                if not rate_printed:
                    rate_cell = f"{int(round(rate * 100))}\\%"
                    rate_printed = True
                lines.append(
                    f"{rate_cell} & {metric_names[metric]} & "
                    + " & ".join(cells)
                    + " \\\\"
                )
            if rate_printed:
                lines.append("\\hline")
    # remove trailing hline duplicate before end
    if lines[-1] == "\\hline":
        pass
    lines += ["\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_classification_metrics_table(
    agg: pd.DataFrame,
    *,
    path_tex: Path,
    path_csv: Path,
    rates: list[float] | None = None,
) -> None:
    """METABRIC supplementary table: Macro-F1, BalAcc, Precision, Recall."""
    metrics = ["f1_macro", "bal_acc", "precision_macro", "recall_macro"]
    metric_names = {
        "f1_macro": "Macro-F1",
        "bal_acc": "Bal.\\ Acc.",
        "precision_macro": "Precision",
        "recall_macro": "Recall",
    }
    sub = agg[
        (agg["dataset"] == "METABRIC") & (agg["metric"].isin(metrics))
    ].copy()
    if rates is not None:
        sub = sub[sub["missing_rate"].isin(rates)]

    export = sub[
        [
            "mechanism",
            "missing_rate",
            "method",
            "metric",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "n_replications",
        ]
    ].sort_values(["mechanism", "missing_rate", "metric", "method"])
    export.to_csv(path_csv, index=False)

    methods = [m for m in METHOD_ORDER if m in set(sub["method"].astype(str))]
    caption = (
        "METABRIC downstream PAM50 classification metrics (EnsembleSoft) under MCAR and MAR, "
        "including the 0\\% missingness baseline. Best value per row in \\textbf{bold} "
        "(all metrics higher is better). Entries are mean $\\pm$ SD across 10 missingness "
        "replications (fold-averaged within replication). Macro-F1 is the primary endpoint; "
        "balanced accuracy, macro-precision, and macro-recall are reported for completeness."
    )
    lines = [
        "% Auto-generated — do not edit by hand",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{caption}}}",
        "\\label{tab:metabric_classification}",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\begin{tabular}{ll" + "c" * len(methods) + "}",
        "\\hline",
        "Rate & Metric & " + " & ".join(methods) + " \\\\",
        "\\hline",
    ]
    for mech in ["MCAR", "MAR"]:
        sm = sub[sub["mechanism"] == mech]
        lines.append(
            f"\\multicolumn{{{2 + len(methods)}}}{{l}}{{\\textit{{{mech}}}}} \\\\"
        )
        for rate in sorted(sm["missing_rate"].unique()):
            rate_printed = False
            for metric in metrics:
                g = sm[(sm["missing_rate"] == rate) & (sm["metric"] == metric)]
                if g.empty:
                    continue
                vals = {
                    str(r["method"]): (float(r["mean"]), float(r["std"]))
                    for _, r in g.iterrows()
                }
                means = pd.Series({m: vals[m][0] for m in methods if m in vals})
                best = _best_mask(means, HIGHER_BETTER[metric])
                cells = []
                for m in methods:
                    if m not in vals:
                        cells.append("---")
                        continue
                    txt = _fmt_mean_sd(*vals[m], precision=3)
                    if bool(best.get(m, False)):
                        txt = f"\\textbf{{{txt}}}"
                    cells.append(txt)
                rate_cell = ""
                if not rate_printed:
                    rate_cell = f"{int(round(float(rate) * 100))}\\%"
                    rate_printed = True
                lines.append(
                    f"{rate_cell} & {metric_names[metric]} & "
                    + " & ".join(cells)
                    + " \\\\"
                )
            if rate_printed:
                lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    path_tex.write_text("\n".join(lines), encoding="utf-8")


def latex_stats_summary(path: Path) -> None:
    fr = pd.read_csv(STATS_DIR / "friedman_reduced_imputers.csv")
    pr = pd.read_csv(STATS_DIR / "primary_contrasts.csv")
    fr = fr[fr["cohort"] == "metabric"].copy()
    pr = pr[pr["cohort"] == "metabric"].copy()

    def short(m: str) -> str:
        return METHOD_MAP.get(m, m)

    metric_order = {"rmse": 0, "corr_rv": 1, "f1_macro": 2}
    metric_display = {
        "rmse": "RMSE",
        "corr_rv": "RV",
        "f1_macro": "Macro-F1",
    }
    fr = fr.copy()
    fr["_m"] = fr["metric"].map(metric_order)
    fr["_mech"] = fr["mechanism"].str.upper().map({"MCAR": 0, "MAR": 1})
    rows_tex = []
    for _, r in fr.sort_values(["_m", "_mech", "missing_rate"]).iterrows():
        metric = metric_display.get(str(r["metric"]), str(r["metric"]).replace("_", "\\_"))
        mech = r["mechanism"].upper()
        rate = f"{int(round(r['missing_rate'] * 100))}\\%"
        chi2 = f"{r['friedman_chi2']:.2f}"
        p = float(r["p_value"])
        sig = "Yes" if p < 0.05 else "No"
        p_s = f"{p:.2g}"
        sub = pr[
            (pr["metric"] == r["metric"])
            & (pr["mechanism"] == r["mechanism"])
            & (np.isclose(pr["missing_rate"], r["missing_rate"]))
            & (pr["method_b"] == "SimpleMean")
            & (
                pr["method_a"].isin(
                    ["MissForest", "RFECA_SVR(k=20)", "KNN(k=5,dist)"]
                )
            )
        ]
        bits = []
        for _, c in sub.iterrows():
            if float(c["p_holm_primary_family"]) < 0.05:
                bits.append(
                    f"{short(c['method_a'])} ($r={c['rank_biserial_r']:.2f}$)"
                )
        contrasts = "; ".join(bits) if bits else "none"
        if r["metric"] == "rmse":
            interp = "recon.\\ differs" if sig == "Yes" else "tied"
        elif r["metric"] == "f1_macro":
            interp = "modest F1 sep." if sig == "Yes" else "F1 tied"
        else:
            interp = "structure differs" if sig == "Yes" else "tied"
        rows_tex.append(
            f"{mech} & {metric} & {rate} & {chi2} & {p_s} & {sig} & {contrasts} & {interp} \\\\"
        )

    caption = (
        "METABRIC statistical summary (Friedman on Mean/KNN/RFECA-k20/MissForest). "
        "Pairwise column: Holm-significant primary contrasts versus Mean "
        "(matched-pairs rank-biserial $r$). Unit: replication means ($n=10$)."
    )
    body = "\n".join(rows_tex)
    tex = f"""% Auto-generated
\\begin{{table}}[t]
\\centering
\\scriptsize
\\caption{{{caption}}}
\\label{{tab:stats_summary}}
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{llp{{0.9cm}}p{{0.9cm}}p{{0.9cm}}ccp{{3.2cm}}p{{2.0cm}}}}
\\hline
Mech. & Metric & Rate & $\\chi^2$ & $p$ & Sig. & Sig.\\ vs Mean & Takeaway \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    path.write_text(tex, encoding="utf-8")


def latex_cptac_table(agg: pd.DataFrame, path_tex: Path, path_csv: Path) -> None:
    sub = agg[
        (agg["dataset"] == "CPTAC 2C")
        & (agg["metric"].isin(["rmse", "corr_rv", "f1_macro"]))
        & (agg["missing_rate"] > 0)
    ].copy()
    export = sub[
        [
            "mechanism",
            "missing_rate",
            "method",
            "metric",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "n_replications",
        ]
    ]
    export.to_csv(path_csv, index=False)
    latex_performance_table(
        agg,
        dataset="CPTAC 2C",
        mechanism=None,
        rates=[0.05, 0.10, 0.20, 0.30],
        path=path_tex,
        caption=(
            "CPTAC 2C exploratory summary (mean $\\pm$ SD). "
            "Small $n=117$ yields high variance; rankings are not used as primary evidence."
        ),
        label="tab:cptac2c",
    )


def inductive_sensitivity_table(path_tex: Path, path_csv: Path) -> None:
    """Compare legacy vs inductive RFECA means; masks differ (legacy vs v2)."""
    rows = []
    legacy_dirs = {
        ("CPTAC 2C", "MCAR"): ROOT / "artifacts" / "discovery_full_20260724_172348",
        ("METABRIC", "MCAR"): ROOT / "artifacts" / "metabric_full_20260724_185916",
        ("CPTAC 2C", "MAR"): ROOT / "artifacts" / "discovery_full_mar_20260725_045052",
        ("METABRIC", "MAR"): ROOT / "artifacts" / "metabric_full_mar_20260725_062517",
    }
    for run in INDUCTIVE_RUNS:
        if not run["dir"].exists():
            continue
        key = (run["dataset"], run["mechanism"])
        legacy_dir = legacy_dirs.get(key)
        if legacy_dir is None or not legacy_dir.exists():
            continue
        for kind, path, scheme in [
            ("legacy", legacy_dir, "legacy"),
            ("inductive", run["dir"], "v2"),
        ]:
            imp = pd.read_csv(path / "exp1_imputation_summary.csv")
            cls = pd.read_csv(path / "exp2_classification_summary.csv")
            if "model" in cls.columns:
                cls = cls[cls["model"] == "EnsembleSoft"]
            for k in [5, 10, 20]:
                name = f"RFECA_SVR(k={k})"
                for rate in [0.05, 0.1, 0.2, 0.3]:
                    ri = imp[(imp.imputer == name) & np.isclose(imp.missing_rate, rate)]
                    rc = cls[(cls.imputer == name) & np.isclose(cls.missing_rate, rate)]
                    if ri.empty or rc.empty:
                        continue
                    rows.append(
                        {
                            "dataset": run["dataset"],
                            "mechanism": run["mechanism"],
                            "missing_rate": rate,
                            "method": f"RFECA-k{k}",
                            "protocol": kind,
                            "seed_scheme": scheme,
                            "masks_identical_to_legacy": False,
                            "rmse_mean": float(ri["rmse_mean"].iloc[0]),
                            "f1_mean": float(rc["f1_mean"].iloc[0]),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        path_csv.write_text("no inductive artifacts found\n", encoding="utf-8")
        path_tex.write_text("% no inductive artifacts\n", encoding="utf-8")
        return

    wide_r = df.pivot_table(
        index=["dataset", "mechanism", "missing_rate", "method"],
        columns="protocol",
        values="rmse_mean",
    ).reset_index()
    wide_f = df.pivot_table(
        index=["dataset", "mechanism", "missing_rate", "method"],
        columns="protocol",
        values="f1_mean",
    ).reset_index()
    merged = wide_r.merge(
        wide_f,
        on=["dataset", "mechanism", "missing_rate", "method"],
        suffixes=("_rmse", "_f1"),
    )
    # columns may be legacy/inductive
    for col in ["legacy", "inductive"]:
        if f"{col}_rmse" not in merged.columns and col in wide_r.columns:
            pass
    # normalize column names after merge
    cols = list(merged.columns)
    # pivot with suffixes: legacy_x style — handle both
    def pick(frame, base):
        if base in frame.columns:
            return frame[base]
        for c in frame.columns:
            if c.startswith("legacy") and base == "legacy":
                return frame[c]
        return None

    out_rows = []
    for _, r in merged.iterrows():
        # After merge_suffixes _rmse _f1, columns are legacy_rmse, inductive_rmse, etc. if both exist
        leg_rmse = r.get("legacy_rmse", r.get("legacy"))
        ind_rmse = r.get("inductive_rmse", r.get("inductive"))
        leg_f1 = r.get("legacy_f1", np.nan)
        ind_f1 = r.get("inductive_f1", np.nan)
        # wide_r merge: columns legacy, inductive then rename — rebuild simpler
    # Simpler rebuild:
    out = []
    for (dataset, mech, rate, method), g in df.groupby(
        ["dataset", "mechanism", "missing_rate", "method"]
    ):
        leg = g[g.protocol == "legacy"]
        ind = g[g.protocol == "inductive"]
        if leg.empty or ind.empty:
            continue
        out.append(
            {
                "dataset": dataset,
                "mechanism": mech,
                "missing_rate": rate,
                "method": method,
                "legacy_rmse": float(leg.rmse_mean.iloc[0]),
                "inductive_rmse": float(ind.rmse_mean.iloc[0]),
                "delta_rmse_ind_minus_leg": float(ind.rmse_mean.iloc[0] - leg.rmse_mean.iloc[0]),
                "legacy_f1": float(leg.f1_mean.iloc[0]),
                "inductive_f1": float(ind.f1_mean.iloc[0]),
                "delta_f1_ind_minus_leg": float(ind.f1_mean.iloc[0] - leg.f1_mean.iloc[0]),
                "masks_identical": False,
                "note": "legacy seed scheme vs v2 — not a pure ablation",
            }
        )
    out_df = pd.DataFrame(out)
    out_df.to_csv(path_csv, index=False)

    # Compact LaTeX: METABRIC MCAR+MAR (CSV retains all cohorts)
    focus = out_df[out_df.dataset == "METABRIC"].copy()
    lines = [
        "% Auto-generated sensitivity table",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Inductive RFECA sensitivity on METABRIC (MCAR and MAR). "
        "Legacy six-imputer campaign versus inductive RFECA-only re-run (seed scheme v2). "
        "Masks/seeds differ; differences are \\emph{not} a pure ablation of inductiveness. "
        "Full CPTAC 2C comparisons are in the accompanying CSV.}",
        "\\label{tab:rfeca_inductive}",
        "\\begin{tabular}{llcccccc}",
        "\\hline",
        "Mech. & Method & Rate & Legacy RMSE & Ind.\\ RMSE & "
        "$\\Delta$RMSE & Legacy F1 & Ind.\\ F1 \\\\",
        "\\hline",
    ]
    for mech in ["MCAR", "MAR"]:
        sub = focus[focus.mechanism == mech].sort_values(["method", "missing_rate"])
        if sub.empty:
            continue
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{mech}}}}} \\\\")
        for _, r in sub.iterrows():
            lines.append(
                f" & {r['method']} & {int(r['missing_rate']*100)}\\% & "
                f"{r['legacy_rmse']:.3f} & {r['inductive_rmse']:.3f} & "
                f"{r['delta_rmse_ind_minus_leg']:+.3f} & "
                f"{r['legacy_f1']:.3f} & {r['inductive_f1']:.3f} \\\\"
            )
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    path_tex.write_text("\n".join(lines), encoding="utf-8")


def write_captions(spearman: dict) -> None:
    rho_o = spearman.get("overall", {}).get("spearman_rho", float("nan"))
    rho_mcar = spearman.get("MCAR", {}).get("spearman_rho", float("nan"))
    rho_mar = spearman.get("MAR", {}).get("spearman_rho", float("nan"))
    text = f"""# Definitive figure/table captions (provisional numbering)

Use these numbers when writing Results. File names under `artifacts/paper_results/` are locked to this scheme.

**Protocol (all main items):** METABRIC / CPTAC 2C six-imputer shared-mask campaigns; 10 missingness replications × 5 stratified CV folds; fold means averaged within replication; EnsembleSoft for Macro-F1; seed scheme **legacy**. Inductive RFECA (v2) is Supplementary only.

---

## Main text

### Figure 1 — Reconstruction (RMSE)
**File:** `fig01_metabric_rmse_by_missingness.{{pdf,png}}`  
**LaTeX label:** `fig:metabric_rmse`

**Caption.** Reconstruction error (RMSE) on METABRIC under MCAR (**A**) and MAR (**B**) as a function of artificial missingness (5–30\\%). Lines show means across ten missingness replications; shaded bands are 95\\% Student $t$ intervals. Metrics are first averaged over five stratified CV folds within each replication.

**In-text cue.** MissForest lowest RMSE; Mean highest; KNN/RFECA intermediate; error rises with missingness.

### Figure 2 — Structure (RV)
**File:** `fig02_metabric_rv_by_missingness.{{pdf,png}}`  
**LaTeX label:** `fig:metabric_rv`

**Caption.** Gene–gene correlation preservation (RV coefficient) on METABRIC under MCAR (**A**) and MAR (**B**). Higher values indicate better structural preservation. Means and 95\\% CIs across replications as in Figure~1.

**In-text cue.** Structure largely tracks RMSE: MissForest best, Mean worst.

### Figure 3 — Downstream PAM50 (Macro-F1)
**File:** `fig03_metabric_macrof1_by_missingness.{{pdf,png}}`  
**LaTeX label:** `fig:metabric_f1`

**Caption.** Downstream PAM50 Macro-F1 (EnsembleSoft) on METABRIC under MCAR (**A**) and MAR (**B**). The $y$-axis is restricted to highlight small differences; absolute performance remains high for all methods. Means and 95\\% CIs across replications as in Figure~1.

**In-text cue.** Near-ties at low missingness; modest MissForest/RFECA separation mainly at $\\ge$20\\%; gaps much smaller than for RMSE. Do not read restricted axis as large absolute effects.

### Figure 4 — RMSE versus Macro-F1
**File:** `fig04_rmse_vs_macrof1.{{pdf,png}}`  
**LaTeX label:** `fig:rmse_vs_f1`

**Caption.** Mean RMSE versus mean Macro-F1 for each method $\\times$ missingness rate on METABRIC (MCAR left; MAR right). Point labels indicate missingness percentage. Descriptive Spearman correlations on these aggregated points: overall $\\rho={rho_o:.2f}$; MCAR $\\rho={rho_mcar:.2f}$; MAR $\\rho={rho_mar:.2f}$ (points are not independent).

**In-text cue.** Large reconstruction gains do not map proportionally onto PAM50 gains (descriptive only).

### Table 1 — Compact METABRIC performance
**File:** `table01_compact_metabric.tex`  
**LaTeX label:** `tab:metabric_compact`

**Caption.** Compact METABRIC summary at 5\\%, 20\\%, and 30\\% missingness under MCAR and MAR. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\\pm$ SD across 10 missingness replications (fold-averaged within replication).

**In-text cue.** Primary numeric table for the two-regime message (clear RMSE/RV separation; compressed F1).

### Table 2 — Statistical summary (METABRIC)
**File:** `table03_statistical_summary.tex`  
**LaTeX label:** `tab:stats_summary`

**Caption.** METABRIC statistical summary. Friedman omnibus on the reduced method set (Mean, KNN, RFECA-$k$20, MissForest). Pairwise column: Holm-significant primary contrasts versus Mean (matched-pairs rank-biserial $r$). Unit of analysis: replication means ($n=10$).

**In-text cue.** Reconstruction/structure differences significant at all rates; F1 omnibus significance mainly at higher missingness.

---

## Supplementary

### Table S1 — METABRIC MCAR (full rates)
**File:** `table01_metabric_mcar.tex`  
**LaTeX label:** `tab:metabric_mcar`

**Caption.** METABRIC performance under MCAR (mean $\\pm$ SD across replications), including the 0\\% missingness Macro-F1 baseline. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\\pm$ SD across 10 missingness replications (fold-averaged within replication).

### Table S2 — METABRIC MAR (full rates)
**File:** `table02_metabric_mar.tex`  
**LaTeX label:** `tab:metabric_mar`

**Caption.** METABRIC performance under MAR (mean $\\pm$ SD across replications), including the 0\\% missingness Macro-F1 baseline. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\\pm$ SD across 10 missingness replications (fold-averaged within replication).

### Table S3 — CPTAC 2C exploratory
**File:** `table04_cptac2c_summary.tex` (+ `table04_cptac2c_summary.csv`)  
**LaTeX label:** `tab:cptac2c`

**Caption.** CPTAC 2C exploratory external cohort ($n=117$; mean $\\pm$ SD). Small $n$ yields high variance; rankings are not used as primary evidence of method superiority. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\\pm$ SD across 10 missingness replications (fold-averaged within replication).

### Table S4 — Inductive RFECA sensitivity
**File:** `table05_rfeca_inductive_sensitivity.tex` (+ full CSV for all cohorts)  
**LaTeX label:** `tab:rfeca_inductive`

**Caption.** Inductive RFECA sensitivity on METABRIC (MCAR and MAR). Legacy six-imputer campaign versus inductive RFECA-only re-run (seed scheme v2). Masks/seeds differ; differences are *not* a pure ablation of inductiveness. Full CPTAC 2C comparisons are in the accompanying CSV.

**In-text cue.** METABRIC Macro-F1 essentially unchanged; do not treat $\\Delta$RMSE as causal effect of the inductive fix alone.

### Table S5 — Classification metrics (Macro-F1 / BalAcc / Precision / Recall)
**File:** `table06_metabric_classification.tex` (+ `table06_metabric_classification.csv`)  
**LaTeX label:** `tab:metabric_classification`

**Caption.** METABRIC downstream PAM50 classification metrics (EnsembleSoft) under MCAR and MAR, including the 0\\% missingness baseline. Best value per row in **bold** (all metrics higher is better). Entries are mean $\\pm$ SD across 10 missingness replications (fold-averaged within replication). Macro-F1 is the primary endpoint; balanced accuracy, macro-precision, and macro-recall are reported for completeness.

**In-text cue.** Companion metrics closely track Macro-F1; use for completeness, not as a second primary claim.

---

## Not numbered in the paper (data only)
- `aggregated_metrics.csv`, `replication_level_metrics.csv`
- `primary_contrasts_statistics.csv`, `full_pairwise_statistics.csv`
- `spearman_rmse_vs_f1.json`, `AUDIT.md`, `results_findings.md`
"""
    (OUT / "figure_table_captions.md").write_text(text, encoding="utf-8")


def write_findings(spearman: dict, agg: pd.DataFrame) -> None:
    text = f"""# Results findings (tied to generated outputs)

## Reconstruction accuracy
- **Statistically supported (METABRIC):** MissForest has the lowest RMSE/MAE at all rates under MCAR and MAR; Mean has the highest.
- **Descriptively consistent:** KNN and RFECA($k$) occupy an intermediate tier; larger RFECA $k$ often helps RMSE but remains above MissForest.
- Ranking of best/worst methods is stable as missingness increases; mid-tier swaps are secondary.

## Structural preservation
- **Statistically supported:** MissForest best, Mean worst on RV; Friedman significant at all rates.
- Structure largely agrees with RMSE for headline methods; mild mid-tier RMSE–RV disagreements are descriptively observed.

## Downstream classification
- Absolute Macro-F1 remains high for all methods.
- **Statistically supported:** near-ties at ~5% (Friedman often n.s.); modest MissForest/RFECA advantages vs Mean at ~20–30% after Holm in primary contrasts.
- **Unsupported:** RFECA superiority over MissForest on F1.
- Reconstruction improvements do **not** translate proportionally into F1 gains (Figure 4; descriptive Spearman overall $\\rho={spearman.get('overall',{}).get('spearman_rho', float('nan')):.2f}$).

## MCAR versus MAR
- **Directionally and statistically consistent:** same qualitative ordering and two-regime message under both mechanisms.
- MAR can make Mean look somewhat worse on reconstruction/structure; it does not reorder methods.

## Statistical evidence
- Source: `artifacts/stats_mcar_mar_20260727_160425`.
- Friedman (reduced set) + Wilcoxon–Holm primary family + rank-biserial + bootstrap CIs on paired deltas.
- Unit of analysis: fold-averaged replication means ($n=10$).

## CPTAC 2C exploratory validation
- Reconstruction direction matches METABRIC (MissForest best / Mean worst) **descriptively**.
- F1 rankings are high-variance / unstable — **exploratory only**.

## Inductive RFECA sensitivity
- Separate artifacts (`*_full_rfeca_*`, seed scheme v2); **not mixed** into main figures.
- METABRIC MCAR F1 changes are negligible relative to main conclusions; RMSE comparisons are confounded by mask changes.
- Label: **non-causal / not a pure ablation**.

## Main vs supplementary recommendation
**Main paper:** Figures 1–4; Table 1 (compact METABRIC); Table 2 (stats summary).  
**Supplementary:** Table S1–S2 (full METABRIC MCAR/MAR); Table S3 (CPTAC 2C); Table S4 (inductive sensitivity); Table S5 (classification metrics); pairwise CSVs if needed.

See `figure_table_captions.md` for locked provisional numbering and final captions.
"""
    (OUT / "results_findings.md").write_text(text, encoding="utf-8")


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    _setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    audited = audit_runs(MAIN_RUNS)
    if len(audited) != 4:
        print("ERROR: expected 4 main runs; aborting figure generation.")
        return 1
    if not STATS_DIR.exists():
        print("ERROR: stats dir missing:", STATS_DIR)
        return 1

    print("\nBuilding replication-level and aggregated datasets...")
    rep = build_replication_level(MAIN_RUNS)
    agg = aggregate_metrics(rep)
    rep.to_csv(OUT / "replication_level_metrics.csv", index=False)
    agg.to_csv(OUT / "aggregated_metrics.csv", index=False)
    print(f"  replication rows: {len(rep)}")
    print(f"  aggregated rows: {len(agg)}")

    print("\nGenerating figures...")
    _line_panels(
        agg,
        dataset="METABRIC",
        metric="rmse",
        ylabel="RMSE",
        stem="fig01_metabric_rmse_by_missingness",
    )
    _line_panels(
        agg,
        dataset="METABRIC",
        metric="corr_rv",
        ylabel="RV coefficient",
        stem="fig02_metabric_rv_by_missingness",
    )
    # F1 with restricted axis to reveal small gaps
    f1 = agg[(agg.dataset == "METABRIC") & (agg.metric == "f1_macro") & (agg.missing_rate > 0)]
    ymin = max(0.80, float(f1["ci95_low"].min()) - 0.005)
    ymax = min(1.0, float(f1["ci95_high"].max()) + 0.005)
    _line_panels(
        agg,
        dataset="METABRIC",
        metric="f1_macro",
        ylabel="Macro-F1",
        stem="fig03_metabric_macrof1_by_missingness",
        ylim=(ymin, ymax),
        restrict_note=f"Note: y-axis restricted to [{ymin:.3f}, {ymax:.3f}] to visualize small differences.",
    )
    spearman = fig_rmse_vs_f1(agg)
    (OUT / "spearman_rmse_vs_f1.json").write_text(
        json.dumps(spearman, indent=2), encoding="utf-8"
    )

    print("Generating tables...")
    latex_performance_table(
        agg,
        dataset="METABRIC",
        mechanism="MCAR",
        rates=None,
        path=OUT / "table01_metabric_mcar.tex",
        caption=(
            "METABRIC performance under MCAR (mean $\\pm$ SD across replications), "
            "including the 0\\% missingness Macro-F1 baseline."
        ),
        label="tab:metabric_mcar",
    )
    latex_performance_table(
        agg,
        dataset="METABRIC",
        mechanism="MAR",
        rates=None,
        path=OUT / "table02_metabric_mar.tex",
        caption=(
            "METABRIC performance under MAR (mean $\\pm$ SD across replications), "
            "including the 0\\% missingness Macro-F1 baseline."
        ),
        label="tab:metabric_mar",
    )
    latex_performance_table(
        agg,
        dataset="METABRIC",
        mechanism=None,
        rates=[0.05, 0.20, 0.30],
        path=OUT / "table01_compact_metabric.tex",
        caption="Compact METABRIC summary at 5\\%, 20\\%, and 30\\% missingness (MCAR and MAR).",
        label="tab:metabric_compact",
    )
    latex_stats_summary(OUT / "table03_statistical_summary.tex")
    # full pairwise export
    pd.read_csv(STATS_DIR / "pairwise_all.csv").to_csv(
        OUT / "full_pairwise_statistics.csv", index=False
    )
    pd.read_csv(STATS_DIR / "primary_contrasts.csv").to_csv(
        OUT / "primary_contrasts_statistics.csv", index=False
    )
    latex_cptac_table(agg, OUT / "table04_cptac2c_summary.tex", OUT / "table04_cptac2c_summary.csv")
    inductive_sensitivity_table(
        OUT / "table05_rfeca_inductive_sensitivity.tex",
        OUT / "table05_rfeca_inductive_sensitivity.csv",
    )
    latex_classification_metrics_table(
        agg,
        path_tex=OUT / "table06_metabric_classification.tex",
        path_csv=OUT / "table06_metabric_classification.csv",
        rates=None,
    )

    write_captions(spearman)
    write_findings(spearman, agg)

    # Quality checklist printout
    print("\n" + "=" * 72)
    print("QUALITY CHECKS")
    print("=" * 72)
    print("Methods in agg:", list(agg["method"].astype(str).unique()))
    print("Rates:", sorted(agg["missing_rate"].unique()))
    print("CI unit: replications (fold-averaged) — confirmed in aggregate_metrics()")
    print("Main vs inductive: separated (table05 only)")
    print("Spearman RMSE vs F1:", spearman)

    outputs = sorted(p.name for p in OUT.iterdir())
    print("\n" + "=" * 72)
    print("INPUTS USED")
    print("=" * 72)
    for run in MAIN_RUNS:
        print(" ", run["dir"])
    print(" ", STATS_DIR)
    print("\nOUTPUTS IN", OUT)
    for name in outputs:
        print(" ", name)

    print("\nUNRESOLVED / NOTES")
    print(" - Main RFECA in six-imputer runs is pre-inductive-fix protocol.")
    print(" - Inductive RFECA (all four axes) complete; Table S4 shows METABRIC MCAR+MAR.")
    print(" - Spearman on Fig4 points is descriptive (non-independent points).")

    print("\nMAIN PAPER PACKAGE")
    print("  Figures: fig01, fig02, fig03, fig04")
    print("  Tables: table01_compact_metabric.tex, table03_statistical_summary.tex")
    print("SUPPLEMENTARY")
    print("  table01/02 full METABRIC; table04 CPTAC 2C; full_pairwise_statistics.csv; table05 inductive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
