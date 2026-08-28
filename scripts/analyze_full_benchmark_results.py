#!/usr/bin/env python3
"""
Post-hoc analysis of completed full Discovery + METABRIC benchmarks.

Produces:
  - new vs legacy (paper / Colab) EnsembleSoft F1 alignment on Discovery
  - cross-cohort Discovery vs METABRIC tables
  - Wilcoxon pairwise imputer comparisons on new raw results
  - imputation→classification ranking consistency
  - fold-level stability summaries

Does not re-run CV. Per-class F1 requires a new run with updated evaluation.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_DIR = ROOT / "artifacts" / "discovery_full_20260719_230651"
METABRIC_DIR = ROOT / "artifacts" / "metabric_full_20260719_234633"
LEGACY_PAPER = ROOT / "archive" / "metricas_original" / "paper_results_table.csv"
LEGACY_RAW = ROOT / "archive" / "legacy_root" / "exp2_classification_raw_all.csv"
OUT_DIR = ROOT / "artifacts" / f"post_full_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

IMPUTERS = [
    "SimpleMean",
    "KNN(k=5,dist)",
    "RFECA_SVR(k=5)",
    "RFECA_SVR(k=10)",
    "RFECA_SVR(k=20)",
]


def _parse_pm(s: str) -> tuple[float, float]:
    m = re.match(r"([0-9.]+)\s*±\s*([0-9.]+)", str(s).strip())
    if not m:
        raise ValueError(f"Cannot parse mean±std: {s!r}")
    return float(m.group(1)), float(m.group(2))


def load_legacy_paper() -> pd.DataFrame:
    df = pd.read_csv(LEGACY_PAPER)
    rows = []
    for _, r in df.iterrows():
        f1_m, f1_s = _parse_pm(r["f1_pm_std"])
        bal_m, bal_s = _parse_pm(r["bal_pm_std"])
        rows.append(
            {
                "source": "legacy_paper",
                "imputer": r["imputer"],
                "missing_rate": float(r["missing_rate"]),
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "bal_mean": bal_m,
                "bal_std": bal_s,
            }
        )
    return pd.DataFrame(rows)


def load_legacy_raw_summary() -> pd.DataFrame:
    raw = pd.read_csv(LEGACY_RAW)
    soft = raw[raw["model"] == "EnsembleSoft"].copy()
    return (
        soft.groupby(["imputer", "missing_rate"], as_index=False)
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            bal_mean=("bal_acc", "mean"),
            bal_std=("bal_acc", "std"),
            n_rows=("f1_macro", "count"),
        )
        .assign(source="legacy_raw")
    )


def compare_new_vs_legacy(new_cls: pd.DataFrame) -> pd.DataFrame:
    paper = load_legacy_paper()
    legacy_raw = load_legacy_raw_summary()
    new = new_cls[new_cls["model"] == "EnsembleSoft"].copy()
    new = new.assign(source="new_discovery_full")

    keys = ["imputer", "missing_rate"]
    merged = new[keys + ["f1_mean", "f1_std", "bal_mean", "bal_std"]].merge(
        paper[keys + ["f1_mean", "f1_std", "bal_mean", "bal_std"]],
        on=keys,
        suffixes=("_new", "_paper"),
        how="inner",
    )
    merged = merged.merge(
        legacy_raw[keys + ["f1_mean", "f1_std", "n_rows"]],
        on=keys,
        how="left",
    ).rename(
        columns={
            "f1_mean": "f1_mean_legacy_raw",
            "f1_std": "f1_std_legacy_raw",
            "n_rows": "n_rows_legacy_raw",
        }
    )
    merged["delta_f1_new_minus_paper"] = merged["f1_mean_new"] - merged["f1_mean_paper"]
    merged["delta_bal_new_minus_paper"] = merged["bal_mean_new"] - merged["bal_mean_paper"]
    return merged.sort_values(["missing_rate", "imputer"]).reset_index(drop=True)


def cross_cohort(disc: pd.DataFrame, meta: pd.DataFrame, metric: str) -> pd.DataFrame:
    """metric in {'cls','imp'}."""
    if metric == "cls":
        d = disc[disc["model"] == "EnsembleSoft"][
            ["imputer", "missing_rate", "f1_mean", "f1_std", "bal_mean", "bal_std"]
        ].copy()
        m = meta[meta["model"] == "EnsembleSoft"][
            ["imputer", "missing_rate", "f1_mean", "f1_std", "bal_mean", "bal_std"]
        ].copy()
        out = d.merge(m, on=["imputer", "missing_rate"], suffixes=("_discovery", "_metabric"))
        out["delta_f1_meta_minus_disc"] = out["f1_mean_metabric"] - out["f1_mean_discovery"]
        return out.sort_values(["missing_rate", "imputer"]).reset_index(drop=True)

    d = disc[["imputer", "missing_rate", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]].copy()
    m = meta[["imputer", "missing_rate", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]].copy()
    out = d.merge(m, on=["imputer", "missing_rate"], suffixes=("_discovery", "_metabric"))
    out["delta_rmse_meta_minus_disc"] = out["rmse_mean_metabric"] - out["rmse_mean_discovery"]
    return out.sort_values(["missing_rate", "imputer"]).reset_index(drop=True)


def replicate_means(raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean metric per (imputer, missing_rate, replicate) — for paired Wilcoxon."""
    return (
        raw.groupby(["imputer", "missing_rate", "replicate"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )


def wilcoxon_pairwise(raw: pd.DataFrame, value_col: str, cohort: str) -> pd.DataFrame:
    means = replicate_means(raw, value_col)
    rows = []
    for rate, g_rate in means.groupby("missing_rate"):
        for a, b in combinations(IMPUTERS, 2):
            va = g_rate.loc[g_rate["imputer"] == a].set_index("replicate")["value"]
            vb = g_rate.loc[g_rate["imputer"] == b].set_index("replicate")["value"]
            common = va.index.intersection(vb.index)
            if len(common) < 2:
                continue
            x, y = va.loc[common].to_numpy(), vb.loc[common].to_numpy()
            note = ""
            if np.allclose(x, y):
                p = 1.0
                note = "identical vectors"
            else:
                try:
                    p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
                except ValueError as exc:
                    p = float("nan")
                    note = str(exc)
            rows.append(
                {
                    "cohort": cohort,
                    "missing_rate": float(rate),
                    "metric": value_col,
                    "method_a": a,
                    "method_b": b,
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "delta_a_minus_b": float(np.mean(x) - np.mean(y)),
                    "p_value": p,
                    "n_pairs": int(len(common)),
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def ranking_consistency(imp: pd.DataFrame, cls: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Spearman-like rank correlation between RMSE (lower better) and F1 (higher better)."""
    rows = []
    cls = cls[cls["model"] == "EnsembleSoft"]
    for rate in sorted(imp["missing_rate"].unique()):
        if float(rate) == 0.0:
            continue
        i = imp[np.isclose(imp["missing_rate"], rate)].set_index("imputer")
        c = cls[np.isclose(cls["missing_rate"], rate)].set_index("imputer")
        common = [x for x in IMPUTERS if x in i.index and x in c.index]
        if len(common) < 3:
            continue
        rmse = i.loc[common, "rmse_mean"].to_numpy(float)
        f1 = c.loc[common, "f1_mean"].to_numpy(float)
        # Rank: low RMSE = rank 1; high F1 = rank 1
        r_rmse = pd.Series(rmse).rank(method="average").to_numpy()
        r_f1 = pd.Series(-f1).rank(method="average").to_numpy()  # negate so high F1 → low rank
        spearman = float(np.corrcoef(r_rmse, r_f1)[0, 1])
        best_imp = common[int(np.argmin(rmse))]
        best_cls = common[int(np.argmax(f1))]
        rows.append(
            {
                "cohort": cohort,
                "missing_rate": float(rate),
                "best_imputer_by_rmse": best_imp,
                "best_imputer_by_f1": best_cls,
                "same_winner": best_imp == best_cls,
                "rank_spearman_rmse_vs_f1": spearman,
            }
        )
    return pd.DataFrame(rows)


def fold_stability(raw_cls: pd.DataFrame, cohort: str) -> pd.DataFrame:
    soft = raw_cls[raw_cls["model"] == "EnsembleSoft"].copy()
    g = (
        soft.groupby(["imputer", "missing_rate"], as_index=False)
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            bal_mean=("bal_acc", "mean"),
            bal_std=("bal_acc", "std"),
            n_degenerate=("degenerate_single_class_preds", "sum"),
            n_rows=("f1_macro", "count"),
        )
        .assign(cohort=cohort)
    )
    g["cv_f1"] = g["f1_std"] / g["f1_mean"].replace(0, np.nan)
    return g.sort_values(["missing_rate", "imputer"]).reset_index(drop=True)


def best_table(cls: pd.DataFrame, imp: pd.DataFrame, cohort: str) -> pd.DataFrame:
    soft = cls[cls["model"] == "EnsembleSoft"]
    rows = []
    for rate in sorted(soft["missing_rate"].unique()):
        c = soft[np.isclose(soft["missing_rate"], rate)]
        best_c = c.loc[c["f1_mean"].idxmax()]
        row = {
            "cohort": cohort,
            "missing_rate": float(rate),
            "best_cls_imputer": best_c["imputer"],
            "best_f1": float(best_c["f1_mean"]),
            "best_f1_std": float(best_c["f1_std"]),
            "best_bal": float(best_c["bal_mean"]),
        }
        if float(rate) > 0:
            i = imp[np.isclose(imp["missing_rate"], rate)]
            best_i = i.loc[i["rmse_mean"].idxmin()]
            row.update(
                {
                    "best_imp_imputer": best_i["imputer"],
                    "best_rmse": float(best_i["rmse_mean"]),
                    "best_rmse_std": float(best_i["rmse_std"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out: Path, findings: dict) -> None:
    lines = [
        "# Post-full-benchmark analysis",
        "",
        f"- Generated (UTC): `{findings['generated_at_utc']}`",
        f"- Discovery run: `{findings['discovery_dir']}`",
        f"- METABRIC run: `{findings['metabric_dir']}`",
        "",
        "## Executive findings",
    ]
    for bullet in findings["bullets"]:
        lines.append(f"- {bullet}")
    lines += [
        "",
        "## Protocol deltas (new vs legacy paper)",
        "- New: fold-local RFECA correlation; `originally_observed_only` mask policy; EnsembleSoft only.",
        "- Legacy paper/Colab: precomputed full-cohort correlation CSV; broader classifier suite in raw archive.",
        "- Absolute F1 levels are therefore **not** expected to match exactly; focus on relative ranking.",
        "",
        "## Artifacts",
    ]
    for name in findings["artifact_files"]:
        lines.append(f"- `{name}`")
    lines.append("")
    (out / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    disc_cls = pd.read_csv(DISCOVERY_DIR / "exp2_classification_summary.csv")
    meta_cls = pd.read_csv(METABRIC_DIR / "exp2_classification_summary.csv")
    disc_imp = pd.read_csv(DISCOVERY_DIR / "exp1_imputation_summary.csv")
    meta_imp = pd.read_csv(METABRIC_DIR / "exp1_imputation_summary.csv")
    disc_raw = pd.read_csv(DISCOVERY_DIR / "exp2_classification_raw.csv")
    meta_raw = pd.read_csv(METABRIC_DIR / "exp2_classification_raw.csv")
    disc_imp_raw = pd.read_csv(DISCOVERY_DIR / "exp1_imputation_raw.csv")
    meta_imp_raw = pd.read_csv(METABRIC_DIR / "exp1_imputation_raw.csv")

    legacy_cmp = compare_new_vs_legacy(disc_cls)
    cross_cls = cross_cohort(disc_cls, meta_cls, "cls")
    cross_imp = cross_cohort(disc_imp, meta_imp, "imp")

    wx_disc_f1 = wilcoxon_pairwise(disc_raw, "f1_macro", "discovery")
    wx_meta_f1 = wilcoxon_pairwise(meta_raw, "f1_macro", "metabric")
    wx_disc_rmse = wilcoxon_pairwise(
        disc_imp_raw[disc_imp_raw["missing_rate"] > 0], "rmse", "discovery"
    )
    wx_meta_rmse = wilcoxon_pairwise(
        meta_imp_raw[meta_imp_raw["missing_rate"] > 0], "rmse", "metabric"
    )
    wx_all = pd.concat(
        [wx_disc_f1, wx_meta_f1, wx_disc_rmse, wx_meta_rmse], ignore_index=True
    )

    rank_disc = ranking_consistency(disc_imp, disc_cls, "discovery")
    rank_meta = ranking_consistency(meta_imp, meta_cls, "metabric")
    ranks = pd.concat([rank_disc, rank_meta], ignore_index=True)

    stab = pd.concat(
        [fold_stability(disc_raw, "discovery"), fold_stability(meta_raw, "metabric")],
        ignore_index=True,
    )
    bests = pd.concat(
        [best_table(disc_cls, disc_imp, "discovery"), best_table(meta_cls, meta_imp, "metabric")],
        ignore_index=True,
    )

    # Significant RFECA vs SimpleMean / KNN highlights (METABRIC F1, alpha=0.05)
    sig_meta = wx_meta_f1[
        (wx_meta_f1["p_value"] < 0.05)
        & (
            (
                wx_meta_f1["method_a"].str.startswith("RFECA")
                & wx_meta_f1["method_b"].isin(["SimpleMean", "KNN(k=5,dist)"])
            )
            | (
                wx_meta_f1["method_b"].str.startswith("RFECA")
                & wx_meta_f1["method_a"].isin(["SimpleMean", "KNN(k=5,dist)"])
            )
        )
    ].copy()

    mean_delta_f1 = float(legacy_cmp["delta_f1_new_minus_paper"].mean())
    mean_abs_delta = float(legacy_cmp["delta_f1_new_minus_paper"].abs().mean())
    meta_vs_mean_rmse = (
        meta_imp[meta_imp["missing_rate"] > 0]
        .assign(
            uplift_vs_mean=lambda d: d.groupby("missing_rate")["rmse_mean"].transform("max")
            - d["rmse_mean"]
        )
    )
    # clearer: SimpleMean RMSE - best RFECA RMSE at each rate
    uplift_rows = []
    for rate, g in meta_imp[meta_imp["missing_rate"] > 0].groupby("missing_rate"):
        mean_rmse = float(g.loc[g["imputer"] == "SimpleMean", "rmse_mean"].iloc[0])
        rfeca = g[g["imputer"].str.startswith("RFECA")]
        best = rfeca.loc[rfeca["rmse_mean"].idxmin()]
        uplift_rows.append(
            {
                "missing_rate": float(rate),
                "simplemean_rmse": mean_rmse,
                "best_rfeca": best["imputer"],
                "best_rfeca_rmse": float(best["rmse_mean"]),
                "rmse_reduction": mean_rmse - float(best["rmse_mean"]),
                "pct_reduction": 100.0 * (mean_rmse - float(best["rmse_mean"])) / mean_rmse,
            }
        )
    uplift = pd.DataFrame(uplift_rows)

    bullets = [
        f"New Discovery EnsembleSoft F1 is on average **{mean_delta_f1:+.3f}** vs legacy paper "
        f"(mean abs delta={mean_abs_delta:.3f}); drop expected from fold-local correlation + stricter mask policy.",
        f"METABRIC: RFECA reduces RMSE vs SimpleMean by "
        f"**{uplift['pct_reduction'].mean():.1f}%** on average across rates "
        f"(best at 5%: {uplift.loc[uplift['missing_rate']==0.05,'pct_reduction'].iloc[0]:.1f}%).",
        f"Ranking consistency (RMSE vs F1): METABRIC same-winner rates = "
        f"{int(rank_meta['same_winner'].sum())}/{len(rank_meta)}; Discovery = "
        f"{int(rank_disc['same_winner'].sum())}/{len(rank_disc)} "
        f"— better imputation does not always mean better classification.",
        f"METABRIC F1 Wilcoxon (RFECA vs Mean/KNN, p<0.05): **{len(sig_meta)}** significant pairs "
        f"(see `wilcoxon_pairwise.csv`).",
        f"Stability: Discovery CV(F1) is ~"
        f"{stab.loc[stab['cohort']=='discovery','cv_f1'].median():.3f} median vs METABRIC ~"
        f"{stab.loc[stab['cohort']=='metabric','cv_f1'].median():.3f} — METABRIC estimates are far tighter.",
        "Per-subtype F1/precision/recall columns are now implemented in `evaluation.py`; "
        "re-run smoke/full to populate `exp2_classification_per_class.csv`.",
    ]

    files = {
        "new_vs_legacy_discovery.csv": legacy_cmp,
        "cross_cohort_classification.csv": cross_cls,
        "cross_cohort_imputation.csv": cross_imp,
        "wilcoxon_pairwise.csv": wx_all,
        "wilcoxon_metabric_rfeca_vs_baselines_sig.csv": sig_meta,
        "ranking_consistency.csv": ranks,
        "fold_stability.csv": stab,
        "best_by_rate.csv": bests,
        "metabric_rfeca_rmse_uplift_vs_mean.csv": uplift,
    }
    for name, df in files.items():
        df.to_csv(OUT_DIR / name, index=False)

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_dir": str(DISCOVERY_DIR),
        "metabric_dir": str(METABRIC_DIR),
        "bullets": bullets,
        "artifact_files": list(files),
        "legacy_mean_delta_f1": mean_delta_f1,
        "legacy_mean_abs_delta_f1": mean_abs_delta,
        "metabric_mean_rmse_pct_reduction_vs_simplemean": float(uplift["pct_reduction"].mean()),
        "n_sig_metabric_rfeca_vs_baseline_f1": int(len(sig_meta)),
    }
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    write_report(OUT_DIR, findings)
    print("Wrote analysis to", OUT_DIR)
    for b in bullets:
        print("  -", b.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
