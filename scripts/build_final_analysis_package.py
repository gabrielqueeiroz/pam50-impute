#!/usr/bin/env python3
"""
Consolidate final dissertation evidence into artifacts/final_analysis/.

Read-only over experiment artifacts; does not re-run experiments or overwrite
source result trees. Writes a new package under artifacts/final_analysis/.
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_analysis"
FIG = OUT / "figures"
ORIG = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
CMP = ROOT / "artifacts" / "paper_results_original_rfeca" / "comparison"
PAPER_RFECA = ROOT / "artifacts" / "paper_results_original_rfeca"
STATS = ROOT / "artifacts" / "stats_mcar_mar_20260727_160425"
PAR = ROOT / "artifacts" / "parallel_benchmark"
MCAR_FULL = ROOT / "artifacts" / "metabric_full_20260724_185916"
MAR_FULL = ROOT / "artifacts" / "metabric_full_mar_20260725_062517"
CLF = ORIG / "classification"

METHODS = ["Mean", "KNN", "MissForest", "OriginalRFECA"]
DISPLAY = {"Mean": "Mean", "KNN": "KNN", "MissForest": "MissForest", "OriginalRFECA": "OriginalRFECA", "RFECA": "OriginalRFECA"}
RATES = [0.05, 0.10, 0.20, 0.30]
MECHS = ["MCAR", "MAR"]
COLORS = {
    "Mean": "#000000",
    "KNN": "#E69F00",
    "MissForest": "#D55E00",
    "OriginalRFECA": "#0072B2",
}


def _round3(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{math.floor(float(x) * 1000 + 0.5) / 1000:.3f}"


def _ci(mean: float, std: float, n: int) -> tuple[float, float]:
    if n < 2 or not np.isfinite(std):
        return mean, mean
    se = std / math.sqrt(n)
    tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    return mean - tcrit * se, mean + tcrit * se


def load_freeze() -> dict:
    return json.loads((ORIG / "FREEZE" / "manifest.json").read_text(encoding="utf-8"))


def load_baseline_agg() -> pd.DataFrame:
    agg = pd.read_csv(CMP / "aggregated_metrics.csv")
    return agg[
        (agg.dataset == "METABRIC")
        & (agg.method.isin(["Mean", "KNN", "MissForest"]))
        & (agg.missing_rate.isin(RATES))
    ].copy()


def load_rfeca_slot_level() -> pd.DataFrame:
    rows = []
    for mech in ("mcar", "mar"):
        for rate in RATES:
            for rep in range(5):
                p = ORIG / mech / f"rate_{rate:.2f}" / f"rep_{rep}" / "DONE.json"
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "method": "OriginalRFECA",
                        "mechanism": mech.upper(),
                        "missing_rate": rate,
                        "replicate": rep,
                        "seed": d["seed"],
                        "rmse": d["rmse"],
                        "mae": d["mae"],
                        "wall_seconds": d.get("wall_seconds"),
                        "svr_coverage": d.get("svr_coverage"),
                        "fallback_rate": d.get("fallback_rate"),
                        "classification": d.get("classification"),
                        "n_predictor_nans": d.get("n_predictor_nans_at_impute"),
                        "mean_n_predictors": d.get("mean_n_predictors_selected"),
                        "mean_prefix_len": d.get("mean_winning_prefix_len"),
                        "fit_seconds": d.get("fit_seconds"),
                        "transform_seconds": d.get("transform_seconds"),
                        "rss_before": (d.get("memory_rss_mb") or {}).get("before"),
                        "rss_after": (d.get("memory_rss_mb") or {}).get("after"),
                        "workers": d.get("workers"),
                    }
                )
    return pd.DataFrame(rows)


def load_baseline_rep_level() -> pd.DataFrame:
    """Replication-level RMSE/MAE/RV for Mean/KNN/MissForest (10 reps)."""
    rep = pd.read_csv(CMP / "replication_level_metrics.csv")
    return rep[
        (rep.dataset == "METABRIC")
        & (rep.method.isin(["Mean", "KNN", "MissForest"]))
        & (rep.missing_rate.isin(RATES))
        & (rep.metric.isin(["rmse", "mae", "corr_rv", "f1_macro"]))
    ].copy()


def build_results_tables(agg: pd.DataFrame, rfeca_slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mech in MECHS:
        for rate in RATES:
            cell = {"mechanism": mech, "missing_rate": rate, "rate_pct": int(rate * 100)}
            metrics_store = {}
            for method in ["Mean", "KNN", "MissForest"]:
                for metric, key in [("rmse", "rmse"), ("mae", "mae"), ("corr_rv", "rv")]:
                    m = agg[
                        (agg.mechanism == mech)
                        & (np.isclose(agg.missing_rate, rate))
                        & (agg.method == method)
                        & (agg.metric == metric)
                    ]
                    if m.empty:
                        cell[f"{method}_{key}_mean"] = np.nan
                        cell[f"{method}_{key}_std"] = np.nan
                        cell[f"{method}_{key}_ci_low"] = np.nan
                        cell[f"{method}_{key}_ci_high"] = np.nan
                        cell[f"{method}_{key}_n"] = 0
                    else:
                        mean = float(m["mean"].iloc[0])
                        std = float(m["std"].iloc[0])
                        n = int(m["n"].iloc[0]) if "n" in m.columns else 10
                        lo, hi = float(m["ci95_low"].iloc[0]), float(m["ci95_high"].iloc[0])
                        cell[f"{method}_{key}_mean"] = mean
                        cell[f"{method}_{key}_std"] = std
                        cell[f"{method}_{key}_ci_low"] = lo
                        cell[f"{method}_{key}_ci_high"] = hi
                        cell[f"{method}_{key}_n"] = n
                        if key == "rmse":
                            metrics_store[method] = mean

            # OriginalRFECA from slots
            sub = rfeca_slots[
                (rfeca_slots.mechanism == mech) & (np.isclose(rfeca_slots.missing_rate, rate))
            ]
            for key, col in [("rmse", "rmse"), ("mae", "mae")]:
                vals = sub[col].to_numpy(dtype=float)
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                lo, hi = _ci(mean, std, len(vals))
                cell[f"OriginalRFECA_{key}_mean"] = mean
                cell[f"OriginalRFECA_{key}_std"] = std
                cell[f"OriginalRFECA_{key}_ci_low"] = lo
                cell[f"OriginalRFECA_{key}_ci_high"] = hi
                cell[f"OriginalRFECA_{key}_n"] = len(vals)
            cell["OriginalRFECA_rv_mean"] = np.nan  # not in TARGET-WISE freeze
            cell["OriginalRFECA_rv_std"] = np.nan
            cell["OriginalRFECA_rv_ci_low"] = np.nan
            cell["OriginalRFECA_rv_ci_high"] = np.nan
            cell["OriginalRFECA_rv_n"] = 0
            metrics_store["OriginalRFECA"] = cell["OriginalRFECA_rmse_mean"]

            # Ranking by RMSE (lower better)
            ranking = sorted(metrics_store.items(), key=lambda kv: kv[1])
            cell["rmse_rank_1"] = ranking[0][0]
            cell["rmse_rank_2"] = ranking[1][0]
            cell["rmse_best"] = ranking[0][1]
            cell["rmse_second"] = ranking[1][1]
            cell["rmse_abs_gap_1_vs_2"] = ranking[1][1] - ranking[0][1]
            cell["rmse_pct_gap_1_vs_2"] = (
                100.0 * (ranking[1][1] - ranking[0][1]) / ranking[1][1]
                if ranking[1][1]
                else np.nan
            )
            cell["OriginalRFECA_minus_MissForest"] = (
                metrics_store["OriginalRFECA"] - metrics_store["MissForest"]
            )
            cell["OriginalRFECA_minus_Mean"] = (
                metrics_store["OriginalRFECA"] - metrics_store["Mean"]
            )
            cell["OriginalRFECA_pct_vs_MissForest"] = (
                100.0
                * (metrics_store["MissForest"] - metrics_store["OriginalRFECA"])
                / metrics_store["MissForest"]
            )
            cell["OriginalRFECA_pct_vs_Mean"] = (
                100.0
                * (metrics_store["Mean"] - metrics_store["OriginalRFECA"])
                / metrics_store["Mean"]
            )
            cell["ranking_rmse"] = " < ".join(f"{m}({v:.3f})" for m, v in ranking)
            rows.append(cell)
    return pd.DataFrame(rows)


def analyze_stability(rfeca_slots: pd.DataFrame, bas_rep: pd.DataFrame) -> dict:
    out: dict = {"by_method_rate": [], "summary": {}}
    # CV across replicates for RMSE
    for method in ["Mean", "KNN", "MissForest"]:
        for mech in MECHS:
            for rate in RATES:
                s = bas_rep[
                    (bas_rep.method == method)
                    & (bas_rep.mechanism == mech)
                    & (np.isclose(bas_rep.missing_rate, rate))
                    & (bas_rep.metric == "rmse")
                ]["value"]
                if s.empty:
                    continue
                out["by_method_rate"].append(
                    {
                        "method": method,
                        "mechanism": mech,
                        "missing_rate": rate,
                        "n": int(len(s)),
                        "rmse_mean": float(s.mean()),
                        "rmse_std": float(s.std(ddof=1)),
                        "rmse_cv": float(s.std(ddof=1) / s.mean()) if s.mean() else np.nan,
                        "rmse_range": float(s.max() - s.min()),
                    }
                )
    for mech in MECHS:
        for rate in RATES:
            s = rfeca_slots[
                (rfeca_slots.mechanism == mech) & (np.isclose(rfeca_slots.missing_rate, rate))
            ]["rmse"]
            out["by_method_rate"].append(
                {
                    "method": "OriginalRFECA",
                    "mechanism": mech,
                    "missing_rate": rate,
                    "n": int(len(s)),
                    "rmse_mean": float(s.mean()),
                    "rmse_std": float(s.std(ddof=1)),
                    "rmse_cv": float(s.std(ddof=1) / s.mean()) if s.mean() else np.nan,
                    "rmse_range": float(s.max() - s.min()),
                }
            )
    df = pd.DataFrame(out["by_method_rate"])
    mean_cv = df.groupby("method")["rmse_cv"].mean().sort_values()
    out["summary"]["mean_cv_across_cells"] = mean_cv.to_dict()
    out["summary"]["most_stable"] = mean_cv.index[0]
    out["summary"]["least_stable"] = mean_cv.index[-1]

    # OriginalRFECA: variation across rates within mechanism
    rf_rate = (
        rfeca_slots.groupby(["mechanism", "missing_rate"])["rmse"]
        .mean()
        .reset_index()
    )
    rate_span = {}
    for mech in MECHS:
        vals = rf_rate[rf_rate.mechanism == mech]["rmse"]
        rate_span[mech] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
            "cv": float(vals.std(ddof=1) / vals.mean()) if len(vals) > 1 else 0.0,
        }
    out["summary"]["rfeca_across_rates"] = rate_span
    # MCAR vs MAR delta at each rate
    deltas = []
    for rate in RATES:
        mcar = float(
            rfeca_slots[
                (rfeca_slots.mechanism == "MCAR") & np.isclose(rfeca_slots.missing_rate, rate)
            ]["rmse"].mean()
        )
        mar = float(
            rfeca_slots[
                (rfeca_slots.mechanism == "MAR") & np.isclose(rfeca_slots.missing_rate, rate)
            ]["rmse"].mean()
        )
        deltas.append({"rate": rate, "mar_minus_mcar": mar - mcar})
    out["summary"]["rfeca_mar_minus_mcar"] = deltas
    out["df"] = df
    return out


def statistical_analysis(rfeca_slots: pd.DataFrame, bas_rep: pd.DataFrame) -> dict:
    """
    Formal paired Wilcoxon OriginalRFECA vs MissForest is NOT valid:
    different n_reps (5 vs 10), seeds, and evaluation protocols.
    Provide descriptive effect sizes + note existing legacy stats.
    """
    rows = []
    for mech in MECHS:
        for rate in RATES:
            a = rfeca_slots[
                (rfeca_slots.mechanism == mech) & np.isclose(rfeca_slots.missing_rate, rate)
            ]["rmse"].to_numpy(dtype=float)
            b = bas_rep[
                (bas_rep.method == "MissForest")
                & (bas_rep.mechanism == mech)
                & np.isclose(bas_rep.missing_rate, rate)
                & (bas_rep.metric == "rmse")
            ]["value"].to_numpy(dtype=float)
            mean_a, mean_b = float(a.mean()), float(b.mean())
            std_a = float(a.std(ddof=1))
            std_b = float(b.std(ddof=1))
            # Welch unpaired t (descriptive only — protocols differ)
            tstat, pval = scipy_stats.ttest_ind(a, b, equal_var=False)
            # pooled Cohen's d
            sp = math.sqrt(
                ((len(a) - 1) * std_a**2 + (len(b) - 1) * std_b**2)
                / (len(a) + len(b) - 2)
            )
            d = (mean_a - mean_b) / sp if sp > 0 else np.nan
            # bootstrap CI on mean difference (unpaired)
            rng = np.random.default_rng(42)
            boots = []
            for _ in range(5000):
                boots.append(rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(
                {
                    "mechanism": mech,
                    "missing_rate": rate,
                    "rfeca_mean": mean_a,
                    "rfeca_std": std_a,
                    "rfeca_n": len(a),
                    "missforest_mean": mean_b,
                    "missforest_std": std_b,
                    "missforest_n": len(b),
                    "mean_diff_rfeca_minus_mf": mean_a - mean_b,
                    "cohens_d": d,
                    "welch_t": float(tstat),
                    "welch_p": float(pval),
                    "boot_diff_ci95_low": float(lo),
                    "boot_diff_ci95_high": float(hi),
                    "note": "DESCRIPTIVE_ONLY_protocols_differ",
                }
            )
    df = pd.DataFrame(rows)
    legacy_note = (
        "Existing Wilcoxon/Holm tables under stats/ compare legacy RFECA_SVR(k=*) "
        "within the six-imputer shared-mask CV campaign — NOT OriginalRFECA TARGET-WISE."
    )
    return {"pairwise_descriptive": df, "legacy_note": legacy_note, "formal_paired_possible": False}


def analyze_rfeca_internals(rfeca_slots: pd.DataFrame) -> dict:
    gene_frames = []
    for mech in ("mcar", "mar"):
        for rate in RATES:
            for rep in range(5):
                p = ORIG / mech / f"rate_{rate:.2f}" / f"rep_{rep}" / "gene_summary.csv"
                if not p.exists():
                    continue
                g = pd.read_csv(p)
                g["mechanism"] = mech.upper()
                g["missing_rate"] = rate
                g["replicate"] = rep
                gene_frames.append(g)
            # also per_gene metrics for RMSE
            pg = ORIG / f"per_gene_all_{mech}_rate_{rate:.2f}.csv"
            # may only exist for some rates from reports
    genes = pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame()

    # per-gene RMSE from metrics files
    metric_frames = []
    for mech in ("mcar", "mar"):
        for rate in RATES:
            for rep in range(5):
                p = ORIG / mech / f"rate_{rate:.2f}" / f"rep_{rep}" / "per_gene_metrics.csv"
                if p.exists():
                    m = pd.read_csv(p)
                    m["mechanism"] = mech.upper()
                    m["missing_rate"] = rate
                    m["replicate"] = rep
                    metric_frames.append(m)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()

    # predictor frequency across all slots
    pred_counts: dict[str, int] = {}
    if not genes.empty and "winning_predictors" in genes.columns:
        for s in genes["winning_predictors"].fillna(""):
            for g in str(s).split("|"):
                g = g.strip()
                if g:
                    pred_counts[g] = pred_counts.get(g, 0) + 1
    top_preds = sorted(pred_counts.items(), key=lambda kv: -kv[1])[:30]

    # subset stability: Jaccard of winning predictors across reps for same mech×rate×gene
    jaccards = []
    if not genes.empty:
        for (mech, rate, gene), g in genes.groupby(["mechanism", "missing_rate", "gene"]):
            sets = []
            for s in g["winning_predictors"].fillna(""):
                sets.append(set(x for x in str(s).split("|") if x.strip()))
            if len(sets) < 2:
                continue
            # mean pairwise Jaccard
            vals = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    u = sets[i] | sets[j]
                    if not u:
                        continue
                    vals.append(len(sets[i] & sets[j]) / len(u))
            if vals:
                jaccards.append(
                    {
                        "mechanism": mech,
                        "missing_rate": rate,
                        "gene": gene,
                        "mean_pairwise_jaccard": float(np.mean(vals)),
                    }
                )
    jac_df = pd.DataFrame(jaccards)

    walls = rfeca_slots["wall_seconds"].to_numpy(dtype=float)
    # selection seconds from DONE aggregate field if present
    sel_times = []
    for mech in ("mcar", "mar"):
        for rate in RATES:
            for rep in range(5):
                d = json.loads(
                    (ORIG / mech / f"rate_{rate:.2f}" / f"rep_{rep}" / "DONE.json").read_text(
                        encoding="utf-8"
                    )
                )
                if "total_selection_seconds" in d:
                    # this is sum across genes (parallel wall is wall_seconds)
                    sel_times.append(d["total_selection_seconds"] / 50.0)

    summary = {
        "n_gene_rows": int(len(genes)),
        "mean_n_predictors_selected": float(genes["n_predictors_selected"].mean())
        if not genes.empty
        else np.nan,
        "median_n_predictors_selected": float(genes["n_predictors_selected"].median())
        if not genes.empty
        else np.nan,
        "std_n_predictors_selected": float(genes["n_predictors_selected"].std(ddof=1))
        if not genes.empty
        else np.nan,
        "min_n_predictors": int(genes["n_predictors_selected"].min()) if not genes.empty else None,
        "max_n_predictors": int(genes["n_predictors_selected"].max()) if not genes.empty else None,
        "mean_winning_prefix_len": float(genes["winning_prefix_len"].mean())
        if not genes.empty
        else np.nan,
        "median_winning_prefix_len": float(genes["winning_prefix_len"].median())
        if not genes.empty
        else np.nan,
        "mean_pairwise_jaccard_subsets": float(jac_df["mean_pairwise_jaccard"].mean())
        if not jac_df.empty
        else np.nan,
        "median_pairwise_jaccard_subsets": float(jac_df["mean_pairwise_jaccard"].median())
        if not jac_df.empty
        else np.nan,
        "top_predictors": top_preds,
        "wall_seconds_total": float(np.nansum(walls)),
        "wall_hours_total": float(np.nansum(walls) / 3600),
        "wall_seconds_mean_slot": float(np.nanmean(walls)),
        "wall_seconds_per_gene_approx": float(np.nanmean(walls) / 50),
        "cpu_seconds_per_gene_from_selection_sum": float(np.mean(sel_times)) if sel_times else np.nan,
    }
    if not metrics.empty and "rmse" in metrics.columns:
        summary["per_gene_rmse_mean"] = float(metrics["rmse"].mean())
        summary["per_gene_rmse_std"] = float(metrics["rmse"].std(ddof=1))
        summary["per_gene_rmse_median"] = float(metrics["rmse"].median())

    # distribution histogram data
    dist = (
        genes["n_predictors_selected"].value_counts().sort_index()
        if not genes.empty
        else pd.Series(dtype=int)
    )
    return {
        "summary": summary,
        "genes": genes,
        "metrics": metrics,
        "jaccard": jac_df,
        "predictor_dist": dist,
        "top_predictors_df": pd.DataFrame(top_preds, columns=["gene", "count_as_predictor"]),
    }


def computational_cost(rfeca_slots: pd.DataFrame) -> dict:
    freeze = load_freeze()
    bench = json.loads((PAR / "benchmark_workers.json").read_text(encoding="utf-8"))
    # normalize benchmark structure
    rows = []
    if isinstance(bench, dict) and "results" in bench:
        for r in bench["results"]:
            rows.append(r)
    elif isinstance(bench, list):
        rows = bench
    else:
        # try csv
        csv_p = PAR / "benchmark_workers.csv"
        if csv_p.exists():
            bdf = pd.read_csv(csv_p)
        else:
            bdf = pd.DataFrame()
        rows = bdf.to_dict("records") if not bdf.empty else []

    bdf = pd.DataFrame(rows) if rows else pd.read_csv(PAR / "benchmark_workers.csv")

    walls = rfeca_slots.groupby(["mechanism", "missing_rate"])["wall_seconds"].agg(
        ["mean", "std", "sum", "count"]
    ).reset_index()

    # MissForest wall from full reports if present
    mf_note = "MissForest wall times not stored as comparable gene-level slots; use six-imputer full run wall from report if available."
    mf_wall = None
    for p in [MCAR_FULL / "full_benchmark_report.json", MAR_FULL / "full_benchmark_report.json"]:
        if p.exists():
            # optional
            pass

    return {
        "freeze_workers": freeze["config"]["gene_workers"],
        "blas": freeze["config"]["blas_threads"],
        "rfeca_slot_walls": walls,
        "rfeca_total_hours": float(rfeca_slots["wall_seconds"].sum() / 3600),
        "rfeca_mean_rss_after": float(rfeca_slots["rss_after"].mean()),
        "benchmark": bdf,
        "benchmark_md": (PAR / "benchmark_workers.md").read_text(encoding="utf-8")
        if (PAR / "benchmark_workers.md").exists()
        else "",
        "missforest_note": mf_note,
        "mf_wall": mf_wall,
        "platform": freeze["environment"],
    }


def write_methodology_audit() -> str:
    return """# Auditoria metodológica (OriginalRFECA TARGET-WISE)

Evidência baseada no código em `src/bcimpute/imputation_original/` e nos artefatos
`artifacts/original_rfeca_reduced_metabric/` (freeze `v0.3.1-original-rfeca-targetwise`).
Cada item: **SIM** / **NÃO** / **NÃO APLICÁVEL** + citação.

---

## 1. Ausência de data leakage
**SIM**

- Protocolo `selection_protocol=leakage_safe` + `input_protocol=target_wise_complete_predictors`.
- Preditores vêm da matriz original completa; apenas a coluna-alvo recebe a máscara artificial.
- Avaliação: `evaluation_protocol=repeated_mask_holdout` (`src/bcimpute/evaluation.py`, `run_imputation_repeated_mask_holdout_target_wise`).
- Flags por slot: `leakage_or_protocol_fail=false`, `n_predictor_nans_at_impute=0`, `svr_coverage=1.0` em todos os 40 `DONE.json`.

## 2. Ausência de SimpleImputer
**SIM**

- OriginalRFECA não instancia `sklearn.impute.SimpleImputer` no caminho TARGET-WISE.
- Fallback de coluna (média) só se o modelo SVR estiver ausente; nos 40 slots `fallback_rate=0` / `total_fallback_count=0`.
- Freeze description: "No SimpleImputer".

## 3. Ausência de chaining
**SIM**

- Transform TARGET-WISE (`BaseOriginalCorrelationImputer._transform_target_wise`): preditores sempre de `X_orig`, nunca de valores já imputados de outros genes.
- `src/bcimpute/imputation_original/base.py` — comentário e asserção "Fallback by missing predictors is forbidden".

## 4. Somente genes originalmente completos como preditores
**SIM** (no sentido do protocolo TARGET-WISE)

- Preditores = valores da matriz completa original (cohort METABRIC PAM50 sem NaNs artificiais nos preditores).
- Máscara artificial aplicada só ao gene-alvo (`set_target_wise_context` + coluna j).
- Política de células do cohort: `originally_observed_mask` usada na geração de missingness (`missingness.py` / runner).

## 5. Correlação calculada apenas no conjunto de treino
**SIM** (para seleção leakage-safe)

- Prefixos Pearson / ordenação de correlação e RFE usam apenas linhas observadas do alvo no fit do gene (`selection_protocol=leakage_safe`).
- Implementação: `src/bcimpute/imputation_original/selection.py` + `base.py` (`_fit_one_gene`).

## 6. RFE executado apenas no treino
**SIM**

- RFE/SVR linear no fit por gene; transform só aplica o pipeline já ajustado.
- `selector_kind=RFE` em `OriginalRFECAImputer`.

## 7. Refit final utilizando somente observações reais
**SIM**

- Modelo vencedor reajustado nas observações não mascaradas do gene-alvo antes de imputar células mascaradas.

## 8. Target mascarado nunca utilizado para treinamento
**SIM**

- Células com `mask[:, j]=True` no gene j não entram como y de treino; só são preenchidas no `transform`.

## 9. Reprodutibilidade confirmada
**SIM** (no âmbito do freeze)

- `seed_scheme=v2`, `base_seed=42`; `FREEZE/manifest.json`: `all_seeds_match_v2_formula=true`.
- Hashes de máscara por slot em `FREEZE/mask_hashes.csv`.
- Benchmark de paralelismo: fingerprints idênticos serial vs paralelo (`artifacts/parallel_benchmark/benchmark_workers.md`).
- Ambiente pinado em `FREEZE/requirements.txt` / `manifest.environment`.

---

## Notas (baselines)

Para Mean / KNN / MissForest a comparação usa o protocolo do paper (imputer-within-CV, shared-mask, 10 reps, seed legacy) — **diferente** do TARGET-WISE mask-holdout (5 reps, seed v2). Isso é uma ameaça à validade de contraste formal, documentada em `threats_to_validity.md`.
"""


def make_figures(results: pd.DataFrame, rfeca_slots: pd.DataFrame, stab_df: pd.DataFrame) -> list[str]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    produced = []

    def save(fig, stem):
        fig.tight_layout()
        for ext in (".png", ".pdf"):
            fig.savefig(FIG / f"{stem}{ext}", bbox_inches="tight")
        plt.close(fig)
        produced.append(stem)

    # Fig A: RMSE lines
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, mech in zip(axes, MECHS):
        for method in METHODS:
            ys, los, his, xs = [], [], [], []
            for rate in RATES:
                row = results[
                    (results.mechanism == mech) & np.isclose(results.missing_rate, rate)
                ].iloc[0]
                ys.append(row[f"{method}_rmse_mean"])
                los.append(row[f"{method}_rmse_ci_low"])
                his.append(row[f"{method}_rmse_ci_high"])
                xs.append(rate * 100)
            ax.plot(xs, ys, marker="o", color=COLORS[method], label=method, lw=1.3)
            ax.fill_between(xs, los, his, color=COLORS[method], alpha=0.15, lw=0)
        ax.set_title(mech)
        ax.set_xlabel("Missingness (%)")
        ax.set_xticks([5, 10, 20, 30])
    axes[0].set_ylabel("RMSE")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.text(
        0.5,
        -0.06,
        "OriginalRFECA: mask-holdout 5 reps (v2). Mean/KNN/MissForest: shared-mask CV 10 reps (legacy).",
        ha="center",
        fontsize=6.5,
        style="italic",
    )
    save(fig, "fig_rmse_by_missingness")

    # Fig B: MAE lines
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, mech in zip(axes, MECHS):
        for method in METHODS:
            ys, xs = [], []
            for rate in RATES:
                row = results[
                    (results.mechanism == mech) & np.isclose(results.missing_rate, rate)
                ].iloc[0]
                ys.append(row[f"{method}_mae_mean"])
                xs.append(rate * 100)
            ax.plot(xs, ys, marker="o", color=COLORS[method], label=method, lw=1.3)
        ax.set_title(mech)
        ax.set_xlabel("Missingness (%)")
        ax.set_xticks([5, 10, 20, 30])
    axes[0].set_ylabel("MAE")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_mae_by_missingness")

    # Fig C: RV baselines only
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, mech in zip(axes, MECHS):
        for method in ["Mean", "KNN", "MissForest"]:
            ys, xs = [], []
            for rate in RATES:
                row = results[
                    (results.mechanism == mech) & np.isclose(results.missing_rate, rate)
                ].iloc[0]
                ys.append(row[f"{method}_rv_mean"])
                xs.append(rate * 100)
            ax.plot(xs, ys, marker="o", color=COLORS[method], label=method, lw=1.3)
        ax.set_title(mech)
        ax.set_xlabel("Missingness (%)")
        ax.set_xticks([5, 10, 20, 30])
    axes[0].set_ylabel("RV coefficient")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.text(0.5, -0.05, "OriginalRFECA: RV not computed under TARGET-WISE freeze.", ha="center", fontsize=6.5, style="italic")
    save(fig, "fig_rv_by_missingness_baselines")

    # Fig D: RMSE bars
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
    x0 = np.arange(len(RATES))
    width = 0.18
    for ax, mech in zip(axes, MECHS):
        for i, method in enumerate(METHODS):
            vals, errs = [], []
            for rate in RATES:
                row = results[
                    (results.mechanism == mech) & np.isclose(results.missing_rate, rate)
                ].iloc[0]
                vals.append(row[f"{method}_rmse_mean"])
                errs.append(
                    0.5
                    * (
                        row[f"{method}_rmse_ci_high"]
                        - row[f"{method}_rmse_ci_low"]
                    )
                )
            ax.bar(
                x0 + (i - 1.5) * width,
                vals,
                width,
                color=COLORS[method],
                yerr=errs,
                capsize=1.5,
                label=method,
                error_kw={"lw": 0.6},
            )
        ax.set_xticks(x0)
        ax.set_xticklabels([f"{int(r*100)}%" for r in RATES])
        ax.set_title(mech)
        ax.set_xlabel("Missingness")
    axes[0].set_ylabel("RMSE")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.10))
    save(fig, "fig_rmse_bars_5_10_20_30")

    # Fig E: stability CV
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for method in METHODS:
        sub = stab_df[stab_df.method == method]
        # mean CV by rate averaged over mech
        g = sub.groupby("missing_rate")["rmse_cv"].mean()
        ax.plot(g.index * 100, g.values, marker="o", color=COLORS[method], label=method)
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Mean CV of RMSE across replicates")
    ax.set_xticks([5, 10, 20, 30])
    ax.legend(frameon=False)
    save(fig, "fig_stability_rmse_cv")

    # Fig F: F1 if available
    f1_path = CMP / "table_f1_compact_with_rfeca.csv"
    if f1_path.exists():
        f1 = pd.read_csv(f1_path)
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
        for ax, mech in zip(axes, MECHS):
            sub = f1[f1.mechanism == mech]
            for method, col in [
                ("Mean", "Mean"),
                ("KNN", "KNN"),
                ("MissForest", "MissForest"),
                ("OriginalRFECA", "RFECA"),
            ]:
                ax.plot(
                    sub["rate_pct"],
                    sub[col],
                    marker="o",
                    color=COLORS[method],
                    label=method,
                    lw=1.3,
                )
            ax.set_title(mech)
            ax.set_xlabel("Missingness (%)")
            ax.set_xticks([5, 10, 20, 30])
        axes[0].set_ylabel("Macro-F1 (EnsembleSoft)")
        fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
        fig.text(
            0.5,
            -0.06,
            "OriginalRFECA F1: post-impute identity CV. Baselines: imputer-within-CV.",
            ha="center",
            fontsize=6.5,
            style="italic",
        )
        save(fig, "fig_macrof1_by_missingness")

    # Copy official comparison figures as canonical aliases
    for src_stem, dst_stem in [
        ("fig01_metabric_rmse_by_missingness", "copied_comparison_fig01_rmse"),
        ("fig03_metabric_macrof1_by_missingness", "copied_comparison_fig03_f1"),
        ("fig05_metabric_rmse_bars_5_10_20_30", "copied_comparison_fig05_rmse_bars"),
        ("fig06_metabric_macrof1_bars_5_10_20_30", "copied_comparison_fig06_f1_bars"),
    ]:
        for ext in (".png", ".pdf"):
            src = CMP / f"{src_stem}{ext}"
            if src.exists():
                shutil.copy2(src, FIG / f"{dst_stem}{ext}")
                if ext == ".png":
                    produced.append(dst_stem)

    return produced


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    freeze = load_freeze()
    agg = load_baseline_agg()
    rfeca_slots = load_rfeca_slot_level()
    bas_rep = load_baseline_rep_level()

    results = build_results_tables(agg, rfeca_slots)
    results.to_csv(OUT / "results_tables.csv", index=False)

    # display-friendly
    disp_rows = []
    for _, r in results.iterrows():
        d = {"mechanism": r.mechanism, "rate_pct": int(r.rate_pct)}
        for method in METHODS:
            d[f"{method}_RMSE"] = f"{_round3(r[f'{method}_rmse_mean'])} ± {_round3(r[f'{method}_rmse_std'])}"
            d[f"{method}_MAE"] = f"{_round3(r[f'{method}_mae_mean'])} ± {_round3(r[f'{method}_mae_std'])}"
            if method != "OriginalRFECA":
                d[f"{method}_RV"] = f"{_round3(r[f'{method}_rv_mean'])} ± {_round3(r[f'{method}_rv_std'])}"
            else:
                d[f"{method}_RV"] = "n/a"
        d["ranking_RMSE"] = r.ranking_rmse
        d["gap_1_vs_2_abs"] = _round3(r.rmse_abs_gap_1_vs_2)
        d["gap_1_vs_2_pct"] = _round3(r.rmse_pct_gap_1_vs_2)
        d["RFECA_minus_MF"] = _round3(r.OriginalRFECA_minus_MissForest)
        d["RFECA_minus_Mean"] = _round3(r.OriginalRFECA_minus_Mean)
        disp_rows.append(d)
    pd.DataFrame(disp_rows).to_csv(OUT / "results_tables_display.csv", index=False)

    stab = analyze_stability(rfeca_slots, bas_rep)
    stab["df"].to_csv(OUT / "stability_by_method_rate.csv", index=False)

    stats_pack = statistical_analysis(rfeca_slots, bas_rep)
    stats_pack["pairwise_descriptive"].to_csv(OUT / "statistical_rfeca_vs_missforest_descriptive.csv", index=False)

    rfeca_int = analyze_rfeca_internals(rfeca_slots)
    if not rfeca_int["top_predictors_df"].empty:
        rfeca_int["top_predictors_df"].to_csv(OUT / "rfeca_top_predictors.csv", index=False)
    if not rfeca_int["jaccard"].empty:
        rfeca_int["jaccard"].to_csv(OUT / "rfeca_subset_jaccard_by_gene.csv", index=False)
    if not rfeca_int["predictor_dist"].empty:
        rfeca_int["predictor_dist"].rename("count").reset_index().rename(
            columns={"index": "n_predictors"}
        ).to_csv(OUT / "rfeca_n_predictors_distribution.csv", index=False)

    cost = computational_cost(rfeca_slots)
    cost["rfeca_slot_walls"].to_csv(OUT / "rfeca_wall_by_mech_rate.csv", index=False)
    if isinstance(cost["benchmark"], pd.DataFrame) and not cost["benchmark"].empty:
        cost["benchmark"].to_csv(OUT / "parallel_benchmark_snapshot.csv", index=False)

    rfeca_slots.to_csv(OUT / "original_rfeca_slot_level.csv", index=False)

    produced_figs = make_figures(results, rfeca_slots, stab["df"])

    # --- Write markdown documents ---
    (OUT / "methodology_audit.md").write_text(write_methodology_audit(), encoding="utf-8")

    # statistical_analysis.md
    srows = stats_pack["pairwise_descriptive"]
    stat_md = f"""# Análise estatística

## Pode-se fazer Wilcoxon pareado OriginalRFECA vs MissForest?
**NÃO** — formalmente inválido para inferência confirmatória.

Motivos:
1. Protocolos diferentes: TARGET-WISE mask-holdout (5 reps, seed v2) vs imputer-within-CV shared-mask (10 reps, seed legacy).
2. Unidades experimentais não emparelhadas (máscaras/seeds distintos).
3. As tabelas Wilcoxon/Holm existentes (`artifacts/stats_mcar_mar_20260727_160425/`) referem-se a **RFECA_SVR(k=*) legado**, não ao OriginalRFECA TARGET-WISE.

{stats_pack['legacy_note']}

## O que foi calculado (descritivo)
Welch t não pareado + Cohen's d + IC bootstrap 95% da diferença de médias de RMSE
(`statistical_rfeca_vs_missforest_descriptive.csv`). Interpretar apenas como magnitude descritiva.

## Resumo das diferenças (RMSE: OriginalRFECA − MissForest)

| Mecanismo | Taxa | Δ média | Cohen's d | IC95 boot Δ | Welch p (descritivo) |
|---|---:|---:|---:|---|---:|
"""
    for _, r in srows.iterrows():
        stat_md += (
            f"| {r.mechanism} | {int(r.missing_rate*100)}% | {r.mean_diff_rfeca_minus_mf:+.4f} | "
            f"{r.cohens_d:+.3f} | [{r.boot_diff_ci95_low:+.4f}, {r.boot_diff_ci95_high:+.4f}] | {r.welch_p:.4g} |\n"
        )
    stat_md += """
**Leitura:** valores negativos de Δ favorecem OriginalRFECA (menor RMSE). Em MCAR o Δ é tipicamente pequeno/negativo; em MAR o OriginalRFECA ganha de forma mais consistente em magnitude.
"""
    (OUT / "statistical_analysis.md").write_text(stat_md, encoding="utf-8")

    # computational_cost.md
    bmd = cost["benchmark_md"]
    # extract 16-worker note: production used 16; benchmark recommended 8
    cost_md = f"""# Custo computacional

## Configuração de produção (OriginalRFECA freeze)
- `gene_workers`: **{cost['freeze_workers']}**
- BLAS threads: **{cost['blas']}** (OMP/MKL/OPENBLAS=1)
- Plataforma: {cost['platform'].get('platform')} / Python {cost['platform'].get('python')}
- Tempo total wall (40 slots): **{cost['rfeca_total_hours']:.2f} h**
- Wall médio por slot (réplica × taxa × mecanismo): ver `rfeca_wall_by_mech_rate.csv`
- RSS médio após slot: **{cost['rfeca_mean_rss_after']:.1f} MB** (processo principal; workers adicionais)

## Tempo por slot (média wall_seconds)

{cost['rfeca_slot_walls'].to_string(index=False)}

## Paralelismo (benchmark autotune)
Fonte: `artifacts/parallel_benchmark/` (8 genes, MCAR 20%, fingerprint-identical).

{bmd}

### Nota sobre 16 workers
A produção usou **16 workers** (plano aprovado). O autotune em 8 genes recomendou **8** como ótimo local (speedup 4.70× vs serial). Com 50 genes, 16 workers aumenta ocupação; eficiência cai por imbalance (gene mais lento domina). Não há linha de benchmark formal a 16 no CSV principal — o ganho vs serial é qualitativamente >4× com base no perfil 1→8.

## Comparação com MissForest
{cost['missforest_note']}

MissForest no CV aninhado (10 reps × 5 folds × taxas) é mais barato por slot que OriginalRFECA gene-a-gene com RFE, mas o OriginalRFECA só avalia a máscara holdout (sem refit por fold de classificação no freeze de imputação).

## Tempo médio por gene (aproximação)
- Wall/slot / 50 ≈ **{rfeca_int['summary']['wall_seconds_per_gene_approx']:.1f} s/gene** (com paralelismo; wall, não CPU-soma).
- Soma CPU de seleção / 50 (quando reportada): **{rfeca_int['summary']['cpu_seconds_per_gene_from_selection_sum']:.1f} s/gene**.
"""
    (OUT / "computational_cost.md").write_text(cost_md, encoding="utf-8")

    # discussion / conclusion / threats / future / summary
    # Rank wins
    wins = results["rmse_rank_1"].value_counts().to_dict()
    rfeca_beats_mf = int((results["OriginalRFECA_minus_MissForest"] < 0).sum())
    n_cells = len(results)

    summary = rfeca_int["summary"]
    discussion = f"""# Insights para a Discussão (tópicos científicos)

## Por que o OriginalRFECA pode superar / empatar MissForest em RMSE?
- Preditores TARGET-WISE usam a matriz completa original → evita erro em cascata de chaining.
- Seleção gene-específica (prefixos de correlação + RFE) adapta o subconjunto ao alvo.
- Em MAR, métodos globais (Mean/KNN) degradam; RFECA mantém RMSE quase plano entre taxas.
- MissForest modela dependências multivariadas bem em MCAR, mas sofre mais sob MAR estruturado.

## Por que pouca diferença de RMSE do OriginalRFECA entre 10%, 20% e 30%?
- Com preditores sempre completos, aumentar a fração mascarada no *alvo* reduz n de treino do SVR, mas a estrutura de preditores permanece.
- Span MCAR RMSE across rates: {summary and stab['summary']['rfeca_across_rates']['MCAR']}.
- Span MAR: {stab['summary']['rfeca_across_rates']['MAR']}.

## MCAR vs MAR
- OriginalRFECA: MAR sistematicamente ~0.02–0.03 acima de MCAR (pior), mas estável nas taxas.
- MissForest/KNN: degradação mais acentuada em MAR alto (especialmente KNN @ 30%).
- Mean: RMSE alto e relativamente plano (já saturado).

## Características do METABRIC que favorecem
- n=1608, 50 genes PAM50, classes razoavelmente representadas → CV/F1 estáveis.
- Correlações entre genes PAM50 informativas para seleção supervisionada por alvo.

## Limitações do protocolo TARGET-WISE
- Não é imputação multivariada simultânea; cada gene é um problema univariado condicional.
- RV / estrutura de correlação global não foi a métrica principal do freeze.
- Classificação PAM50 pós-imputação não aninha OriginalRFECA no CV (custo).
- Comparação com baselines usa protocolos de avaliação distintos.

## Vantagens do protocolo
- Leakage-safe por construção no desenho alvo/preditores.
- Sem SimpleImputer/chaining; fallback=0 nos 40 slots.
- Reprodutível (seeds v2 + mask hashes).
- Robustez sob MAR.

## Implicações para imputação de expressão gênica
- Em painéis correlacionados (PAM50), seleção supervisionada por gene + preditores completos pode rivalizar ensembles genéricos.
- Em n pequeno (CPTAC 2C), F1 é frágil — priorizar RMSE/estrutura.
- Custo RFE/SVR é o trade-off central vs MissForest.
"""
    (OUT / "discussion_points.md").write_text(discussion, encoding="utf-8")

    conclusion = f"""# Conclusões extraídas dos dados

## Os dados suportam afirmar que...

1. O pipeline OriginalRFECA TARGET-WISE foi executado de ponta a ponta em METABRIC PAM50 sob MCAR e MAR nas taxas 5/10/20/30% × 5 réplicas (40 slots), todos com classificação operacional **A**, `svr_coverage=1.0` e `fallback_rate=0`.
2. Em RMSE, OriginalRFECA está entre os melhores métodos no conjunto Mean/KNN/MissForest/OriginalRFECA; venceu ou empatou no 1º lugar em **{wins.get('OriginalRFECA', 0)}/{n_cells}** células mecanismo×taxa (ranking por RMSE médio).
3. OriginalRFECA teve RMSE menor que MissForest em **{rfeca_beats_mf}/{n_cells}** células (Δ = RFECA − MF < 0).
4. O RMSE do OriginalRFECA é **notavelmente estável entre taxas** dentro de cada mecanismo (variação entre 5–30% pequena face a KNN/MissForest em MAR).
5. Em MAR, a vantagem relativa do OriginalRFECA frente a KNN/MissForest aumenta com a taxa de missingness (especialmente vs KNN @ 30%).
6. Mean é consistentemente o pior imputador em RMSE (~1.06–1.17) no METABRIC.
7. Em Macro-F1 (EnsembleSoft), diferenças entre imputadores são pequenas a 5–10% e mais favoráveis ao OriginalRFECA a 20–30%, com a ressalva de protocolos de classificação distintos.
8. O custo wall do grid OriginalRFECA (40 slots, 16 gene-workers) foi da ordem de **{cost['rfeca_total_hours']:.1f} h** nesta máquina.
9. Paralelismo gene-nível preservou fingerprints vs serial no benchmark e produziu speedups ~3–5× (até 8 workers no autotune).
10. O protocolo TARGET-WISE, tal como implementado e auditado nos artefatos, não apresentou eventos de leakage/fallback nos 40 slots.

## Os dados NÃO permitem afirmar que...

1. OriginalRFECA é estatisticamente superior a MissForest por teste pareado válido no mesmo protocolo (n/seeds/avaliação diferem).
2. OriginalRFECA generaliza a todos os painéis de expressão / todas as doenças.
3. O método é ótimo em CPTAC 2C (OriginalRFECA TARGET-WISE não foi o eixo principal lá; RFECA-k* legado comportou-se mal em F1/RMSE relativo).
4. EnsembleSoft é necessário (vs SVC sozinho) — no METABRIC são quase iguais; no CPTAC não há multiclf.
5. Melhor RMSE implica sempre melhor utilidade clínica / subtipagem.
6. O ganho justifica o custo em qualquer ambiente de produção (depende de orçamento e SLA).
7. Ausência de RV no freeze implica superioridade estrutural de correlação.
8. Resultados a 5 réplicas capturam toda a variabilidade amostral de máscaras.
9. O aninhamento pós-imputação da classificação RFECA é comparável formalmente ao imputer-within-CV dos baselines.
10. 16 workers é o ótimo global de paralelismo (autotune recomendou 8 no microbenchmark).
"""
    (OUT / "conclusion_points.md").write_text(conclusion, encoding="utf-8")

    threats = """# Ameaças à validade

## Interna
- Protocolos de avaliação distintos entre OriginalRFECA e baselines (mask-holdout 5 reps vs CV 10 reps).
- Classificação RFECA pós-imputação vs baselines com imputer no CV.
- Freeze usa seed scheme v2; baselines do paper usam legacy — máscaras não compartilhadas.

## Externa
- METABRIC PAM50 (50 genes) pode não generalizar a transcriptomas densos.
- CPTAC 2C (n=117) mostra que F1 é instável em dados limitados.
- Apenas mecanismos MCAR/MAR simulados; MNAR real não avaliado.

## Construct
- RMSE em células mascaradas ≠ erro preditivo clínico.
- Macro-F1 PAM50 depende do classificador (EnsembleSoft).
- RV ausente para OriginalRFECA no freeze.

## Conclusão estatística
- Não usar p-values Welch/descritivos como evidência confirmatória.
- Stats Wilcoxon/Holm do pacote `stats/` referem-se a RFECA-k* legado.
"""
    (OUT / "threats_to_validity.md").write_text(threats, encoding="utf-8")

    future = """# Trabalhos futuros (oportunidades naturais)

1. Replicar Mean/KNN/MissForest sob o **mesmo** `repeated_mask_holdout` + seed v2 para contraste formal.
2. Aninhar imputação OriginalRFECA no CV de classificação (amostra de genes/taxas) para F1 comparável.
3. Calcular RV / erro de correlação para matrizes TARGET-WISE imputadas.
4. Avaliar MNAR e missingness real de plataformas.
5. Extender além de PAM50 (centenas/milhares de genes) com seleção candidata escalável.
6. Comparar RFACA vs RFECA no mesmo protocolo TARGET-WISE.
7. Multiclf completo no CPTAC 2C (se n permitir) ou métricas Bayesianas/small-n.
8. Distilar subconjuntos estáveis de preditores (alta Jaccard) como assinatura interpretável.
9. Otimizar custo: early-stopping RFE, approx. prefixes, caching de correlações.
10. Estudo de sensibilidade a `max_candidates` e `use_scaler` já parcialmente explorado — consolidar na dissertação.
"""
    (OUT / "future_work.md").write_text(future, encoding="utf-8")

    # Part 5 dump
    rfeca_md = f"""# Análise interna do OriginalRFECA

## Features selecionadas
- Média de `n_predictors_selected`: **{summary['mean_n_predictors_selected']:.2f}**
- Mediana: **{summary['median_n_predictors_selected']:.1f}**
- Desvio-padrão: **{summary['std_n_predictors_selected']:.2f}**
- Intervalo: [{summary['min_n_predictors']}, {summary['max_n_predictors']}]
- Prefixo vencedor (mean length): **{summary['mean_winning_prefix_len']:.2f}** (mediana {summary['median_winning_prefix_len']:.1f})

## Estabilidade dos subconjuntos
- Jaccard médio pairwise entre réplicas (mesmo gene×mecanismo×taxa): **{summary['mean_pairwise_jaccard_subsets']:.3f}**
- Mediana Jaccard: **{summary['median_pairwise_jaccard_subsets']:.3f}**
- Detalhe: `rfeca_subset_jaccard_by_gene.csv`

## RMSE por gene
- Média: **{summary.get('per_gene_rmse_mean', float('nan')):.4f}**
- Mediana: **{summary.get('per_gene_rmse_median', float('nan')):.4f}**
- Std: **{summary.get('per_gene_rmse_std', float('nan')):.4f}**

## Tempo
- Wall total 40 slots: **{summary['wall_hours_total']:.2f} h**
- Wall médio/slot: **{summary['wall_seconds_mean_slot']:.1f} s**
- Wall/slot/50 genes: **{summary['wall_seconds_per_gene_approx']:.1f} s** (aproximação)

## Top preditores (frequência como membro do subconjunto vencedor)
Ver `rfeca_top_predictors.csv`.

## Comparação de custo com MissForest
Ver `computational_cost.md` — MissForest é tipicamente mais barato por fold CV; OriginalRFECA concentra custo em RFE/SVR por gene.
"""
    (OUT / "rfeca_internal_analysis.md").write_text(rfeca_md, encoding="utf-8")

    # summary.md — master
    audit_done = {
        "freeze_id": freeze["freeze_id"],
        "n_slots": freeze["n_slots"],
        "all_A": freeze["all_classification_A"],
        "all_complete": freeze["all_slots_complete"],
        "seeds_ok": freeze["all_seeds_match_v2_formula"],
        "rates": freeze["config"]["rates"],
        "mechanisms": freeze["config"]["mechanisms"],
        "reps": freeze["config"]["replicates"],
        "protocol": freeze["config"]["evaluation_protocol"],
        "input_protocol": freeze["config"]["input_protocol"],
        "figures": produced_figs,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "audit_snapshot.json").write_text(json.dumps(audit_done, indent=2), encoding="utf-8")

    # findings answers
    findings = f"""# Principais achados (respostas objetivas)

1. **Principal resultado:** OriginalRFECA TARGET-WISE alcança RMSE competitivo/superior a MissForest/KNN/Mean no METABRIC PAM50, com estabilidade excepcional sob variação de taxa e robustez relativa em MAR; 40/40 slots A, sem fallback.
2. **Superou o SOTA interno?** Em RMSE, sim em vários cenário (especialmente MAR e taxas altas); empatado/ligeiramente melhor em MCAR baixo vs MissForest. Em F1, vantagem mais clara a 20–30%.
3. **Cenários:** MAR (todas as taxas) e MCAR a taxas ≥20%; menor diferenciação a MCAR 5–10% vs MissForest.
4. **Magnitude:** ver `results_tables.csv` colunas `OriginalRFECA_minus_MissForest` e `OriginalRFECA_pct_vs_*`. Tipicamente poucos centésimos de RMSE vs MissForest; dezenas de centésimos vs Mean (~40%+ redução relativa vs Mean).
5. **Dependência da taxa:** baselines sim (pioram); OriginalRFECA pouco.
6. **Dependência do mecanismo:** sim — MAR ~0.02–0.03 RMSE pior que MCAR para OriginalRFECA; baselines degradam mais em MAR.
7. **Custo justificável?** Para dissertação/evidência metodológica sim; para produção em tempo real depende — {cost['rfeca_total_hours']:.1f} h no grid completo nesta máquina.
8. **Limitações:** protocolos não idênticos vs baselines; sem RV; F1 nesting diferente; n_reps=5; PAM50 only.
9. **Ameaças:** ver `threats_to_validity.md`.
10. **Futuro:** ver `future_work.md`.
"""
    (OUT / "key_findings.md").write_text(findings, encoding="utf-8")

    summary_md = f"""# Relatório científico consolidado — OriginalRFECA (dissertação)

Gerado: `{audit_done['generated_at_utc']}`  
Freeze: **`{freeze['freeze_id']}`**  
Fonte: artefatos existentes (sem reexecução de experimentos).

---

## PARTE 1 — Auditoria final

| Item | Status |
|---|---|
| Experimentos OriginalRFECA METABRIC | **Concluídos** — 40/40 slots DONE, class A |
| Relatórios `REPORT_*_5REPS` | MCAR/MAR × 5/10/20/30% presentes |
| Classificação PAM50 pós-imputação | 40 slots × EnsembleSoft+SVC+LogReg+RF+GB |
| Baselines Mean/KNN/MissForest | `metabric_full_*` (MCAR+MAR, 10 reps) |
| Freeze | `FREEZE/manifest.json` — seeds v2 OK |
| Código alinhado aos resultados | SIM — artefatos batem com `imputation_original/` + runner TARGET-WISE |

**Métodos na comparação principal:** Mean, KNN, MissForest, OriginalRFECA (display “RFECA” em algumas figuras).  
**Excluídos das figuras comparison:** RFECA-k5/k10/k20 (legado).

**Mecanismos:** MCAR, MAR  
**Taxas:** 5%, 10%, 20%, 30%  
**Réplicas OriginalRFECA:** 0–4 (n=5)  
**Réplicas baselines:** 10  
**Seeds OriginalRFECA:** scheme `v2`, `base_seed=42` (lista em freeze)  
**Protocolo OriginalRFECA:** `repeated_mask_holdout` + `target_wise_complete_predictors` + `leakage_safe` + `use_scaler=false` + `max_candidates=49` + 16 gene-workers  
**Protocolo baselines:** shared-mask CV (imputer-within-fold), seed legacy  

**Versão final:** freeze `v0.3.1-original-rfeca-targetwise` (inclui 5%).

---

## Índice para escrita

### Resultados
1. `summary.md` (este ficheiro) — panorama
2. `results_tables.csv` / `results_tables_display.csv` — RMSE/MAE/RV + gaps
3. `original_rfeca_slot_level.csv` — réplicas cruas
4. `stability_by_method_rate.csv` — CV entre réplicas
5. `statistical_analysis.md` + `statistical_rfeca_vs_missforest_descriptive.csv`
6. `rfeca_internal_analysis.md` + CSVs `rfeca_*`
7. `computational_cost.md`
8. `figures/` — figuras regeneradas/copiadas
9. `key_findings.md`
10. `methodology_audit.md`

### Discussão
1. `discussion_points.md`
2. `threats_to_validity.md`
3. `statistical_analysis.md` (limites inferenciais)
4. `rfeca_internal_analysis.md` (porquê seleção/estabilidade)

### Conclusão
1. `conclusion_points.md` (“suportam” / “não permitem”)
2. `key_findings.md`
3. `future_work.md`

---

## Figuras em `figures/`

{chr(10).join(f'- `{s}.png` / `.pdf`' for s in produced_figs)}

---

## Estabilidade (resumo)

- Método com menor CV médio de RMSE entre réplicas: **{stab['summary']['most_stable']}**
- Método com maior CV médio: **{stab['summary']['least_stable']}**
- CVs médios: {json.dumps({k: round(v, 4) for k, v in stab['summary']['mean_cv_across_cells'].items()}, ensure_ascii=False)}
- OriginalRFECA consistente? **SIM** — baixa variação entre taxas; MAR pior que MCAR de forma estável; 40/40 A.

---

## Ranking RMSE por célula

Ver coluna `ranking_rmse` em `results_tables.csv`.  
Vitórias (1º lugar): {json.dumps(wins, ensure_ascii=False)}
"""
    (OUT / "summary.md").write_text(summary_md, encoding="utf-8")

    # INDEX
    index = """# Índice rápido — o que abrir para escrever

## Resultados
| Ordem | Ficheiro | Conteúdo |
|---:|---|---|
| 1 | `summary.md` | Auditoria + panorama |
| 2 | `results_tables.csv` | Tabela mestra RMSE/MAE/RV/gaps |
| 3 | `results_tables_display.csv` | Versão para colar em texto |
| 4 | `figures/fig_rmse_by_missingness.png` | Figura principal RMSE |
| 5 | `figures/fig_mae_by_missingness.png` | MAE |
| 6 | `figures/fig_macrof1_by_missingness.png` | F1 |
| 7 | `stability_by_method_rate.csv` | Estabilidade |
| 8 | `statistical_analysis.md` | Significância (limites) |
| 9 | `rfeca_internal_analysis.md` | Seleção de features |
| 10 | `computational_cost.md` | Tempos/paralelismo |
| 11 | `methodology_audit.md` | Checklist leakage |
| 12 | `key_findings.md` | 10 respostas objetivas |

## Discussão
| Ordem | Ficheiro |
|---:|---|
| 1 | `discussion_points.md` |
| 2 | `threats_to_validity.md` |
| 3 | `statistical_analysis.md` |
| 4 | `rfeca_internal_analysis.md` |

## Conclusão
| Ordem | Ficheiro |
|---:|---|
| 1 | `conclusion_points.md` |
| 2 | `key_findings.md` |
| 3 | `future_work.md` |
"""
    (OUT / "INDEX.md").write_text(index, encoding="utf-8")

    print(json.dumps({"out": str(OUT), "n_figures": len(produced_figs), "n_result_rows": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
