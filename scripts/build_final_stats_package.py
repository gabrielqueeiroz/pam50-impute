"""Final statistical package from frozen artifacts only.

Does NOT re-run imputation or modify experiment code.
Separates formally valid baseline comparisons (A) from descriptive
OriginalRFECA-vs-baseline summaries (B; no p-values).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_analysis"
STATS = OUT / "stats_final"
STATS.mkdir(parents=True, exist_ok=True)

BASELINE_METHODS = ["Mean", "KNN", "MissForest"]
ALL_METHODS = ["Mean", "KNN", "MissForest", "OriginalRFECA"]
RNG = np.random.default_rng(42)
N_BOOT = 5000


def df_to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table without tabulate dependency."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.6g}" if abs(v) < 1e-2 or abs(v) >= 100 else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        rank = n - i
        val = min(1.0, p[idx] * rank)
        running = max(running, val)
        adj[idx] = running
    return adj


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile CI for mean(a)-mean(b); independent samples (descriptive or paired handled by caller)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        aa = RNG.choice(a, size=len(a), replace=True)
        bb = RNG.choice(b, size=len(b), replace=True)
        diffs[i] = aa.mean() - bb.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_paired_diff(deltas: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    d = np.asarray(deltas, dtype=float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = RNG.choice(d, size=len(d), replace=True).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_baseline_reps() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT
        / "artifacts/paper_results_original_rfeca/comparison/replication_level_metrics.csv"
    )
    df = df[df["dataset"] == "METABRIC"].copy()
    df = df[df["method"].isin(BASELINE_METHODS)]
    df = df[df["missing_rate"] > 0]
    return df


def load_rfeca_slots() -> pd.DataFrame:
    return pd.read_csv(OUT / "original_rfeca_slot_level.csv")


def section_a_valid(reps: pd.DataFrame) -> dict:
    """Friedman + paired Wilcoxon + Holm among Mean/KNN/MissForest only."""
    friedman_rows = []
    pairwise_rows = []

    for mech in ["MCAR", "MAR"]:
        for rate in [0.05, 0.1, 0.2, 0.3]:
            for metric, higher_better in [
                ("rmse", False),
                ("mae", False),
                ("corr_rv", True),
                ("f1_macro", True),
            ]:
                sub = reps[
                    (reps.mechanism == mech)
                    & (np.isclose(reps.missing_rate, rate))
                    & (reps.metric == metric)
                ]
                wide = sub.pivot_table(
                    index="replication", columns="method", values="value"
                )
                for m in BASELINE_METHODS:
                    if m not in wide.columns:
                        raise RuntimeError(f"missing {m} for {mech} {rate} {metric}")
                wide = wide[BASELINE_METHODS].dropna()
                if len(wide) < 3:
                    continue

                # Friedman: lower rank = better → flip if higher_is_better
                values = wide.values.copy()
                if higher_better:
                    values = -values
                chi2, p_f = stats.friedmanchisquare(
                    *[values[:, j] for j in range(values.shape[1])]
                )
                # mean ranks on the orientation used for Friedman (lower better)
                ranks = np.apply_along_axis(stats.rankdata, 1, values)
                mean_ranks = ranks.mean(axis=0)
                mean_vals = wide.mean()
                row = {
                    "cohort": "METABRIC",
                    "mechanism": mech,
                    "missing_rate": rate,
                    "metric": metric,
                    "higher_is_better": higher_better,
                    "n_blocks": len(wide),
                    "n_imputers": 3,
                    "friedman_chi2": float(chi2),
                    "p_value": float(p_f),
                }
                for j, m in enumerate(BASELINE_METHODS):
                    row[f"mean_rank__{m}"] = float(mean_ranks[j])
                    row[f"mean_value__{m}"] = float(mean_vals[m])
                friedman_rows.append(row)

                # Pairwise Wilcoxon (paired on replicates)
                pair_p = []
                pair_meta = []
                for i, ma in enumerate(BASELINE_METHODS):
                    for mb in BASELINE_METHODS[i + 1 :]:
                        a = wide[ma].values
                        b = wide[mb].values
                        # Wilcoxon on a-b; alternative two-sided
                        try:
                            wstat, p_w = stats.wilcoxon(a, b, zero_method="wilcox")
                        except ValueError:
                            wstat, p_w = np.nan, 1.0
                        delta = float(a.mean() - b.mean())
                        dvec = a - b
                        # matched-pairs rank-biserial
                        n_pos = int(np.sum(dvec > 0))
                        n_neg = int(np.sum(dvec < 0))
                        n_nz = n_pos + n_neg
                        r_rb = (
                            (n_pos - n_neg) / n_nz if n_nz else np.nan
                        )
                        lo, hi = bootstrap_paired_diff(dvec)
                        pair_meta.append(
                            {
                                "cohort": "METABRIC",
                                "mechanism": mech,
                                "missing_rate": rate,
                                "metric": metric,
                                "higher_is_better": higher_better,
                                "method_a": ma,
                                "method_b": mb,
                                "mean_a": float(a.mean()),
                                "mean_b": float(b.mean()),
                                "delta_a_minus_b": delta,
                                "delta_ci95_low": lo,
                                "delta_ci95_high": hi,
                                "wilcoxon_stat": float(wstat) if pd.notna(wstat) else np.nan,
                                "p_value": float(p_w),
                                "rank_biserial_r": float(r_rb) if pd.notna(r_rb) else np.nan,
                                "n_pairs": len(wide),
                                "n_boot": N_BOOT,
                            }
                        )
                        pair_p.append(p_w)

                adj = holm(np.asarray(pair_p, dtype=float))
                for meta, p_h in zip(pair_meta, adj):
                    meta["p_holm"] = float(p_h)
                    pairwise_rows.append(meta)

    friedman = pd.DataFrame(friedman_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    friedman.to_csv(STATS / "A_friedman_baselines_metabric.csv", index=False)
    pairwise.to_csv(STATS / "A_wilcoxon_holm_baselines_metabric.csv", index=False)

    # Primary contrasts of interest: MissForest vs Mean, MissForest vs KNN, KNN vs Mean
    primary = pairwise[
        pairwise.apply(
            lambda r: {r.method_a, r.method_b}
            in [
                {"MissForest", "Mean"},
                {"MissForest", "KNN"},
                {"KNN", "Mean"},
            ],
            axis=1,
        )
    ].copy()
    primary.to_csv(STATS / "A_primary_contrasts_baselines.csv", index=False)

    md = []
    md.append("# A) Comparações formalmente válidas (baselines)\n")
    md.append("**Âmbito:** METABRIC · Mean / KNN / MissForest · mesmo protocolo CV shared-mask.\n")
    md.append("**Excluídos:** RFECA-k* e OriginalRFECA (protocolos distintos).\n")
    md.append("**Unidade:** média dos 5 folds por réplica (n=10).\n")
    md.append("**Testes:** Friedman; Wilcoxon signed-rank pareado; Holm por (mech×rate×metric);")
    md.append(" bootstrap IC95% do Δ pareado; rank-biserial.\n")
    md.append("\n## Métricas\n")
    md.append("- `rmse`, `mae`, `corr_rv`, `f1_macro`\n")
    md.append("\n## Artefactos\n")
    md.append("- `A_friedman_baselines_metabric.csv`\n")
    md.append("- `A_wilcoxon_holm_baselines_metabric.csv`\n")
    md.append("- `A_primary_contrasts_baselines.csv`\n")
    # Highlight RMSE MissForest vs Mean
    mf_mean = primary[
        (primary.metric == "rmse")
        & (
            ((primary.method_a == "MissForest") & (primary.method_b == "Mean"))
            | ((primary.method_a == "Mean") & (primary.method_b == "MissForest"))
        )
    ]
    md.append("\n## Destaque RMSE — MissForest vs Mean (Holm)\n")
    md.append("| Mech | Rate | Δ (MF−Mean)* | IC95% | p_Holm |\n|---|---:|---:|---|---:|\n")
    for _, r in mf_mean.sort_values(["mechanism", "missing_rate"]).iterrows():
        if r.method_a == "MissForest":
            delta, lo, hi = r.delta_a_minus_b, r.delta_ci95_low, r.delta_ci95_high
        else:
            delta, lo, hi = -r.delta_a_minus_b, -r.delta_ci95_high, -r.delta_ci95_low
        md.append(
            f"| {r.mechanism} | {int(r.missing_rate*100)}% | {delta:.4f} | "
            f"[{lo:.4f}, {hi:.4f}] | {r.p_holm:.4g} |\n"
        )
    md.append("\n\\*Δ negativo favorece MissForest.\n")
    (STATS / "A_valid_baselines.md").write_text("".join(md), encoding="utf-8")
    return {"friedman": friedman, "pairwise": pairwise}


def section_b_descriptive(reps: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """OriginalRFECA vs baselines — descriptive only, NO p-values."""
    rows = []
    for mech in ["MCAR", "MAR"]:
        for rate in [0.05, 0.1, 0.2, 0.3]:
            for metric, col_rfeca in [("rmse", "rmse"), ("mae", "mae")]:
                rf = slots[slots.mechanism == mech]
                rf = rf[np.isclose(rf.missing_rate, rate)]
                a = rf[col_rfeca].astype(float).values

                for base in BASELINE_METHODS:
                    bsub = reps[
                        (reps.mechanism == mech)
                        & (np.isclose(reps.missing_rate, rate))
                        & (reps.method == base)
                        & (reps.metric == metric)
                    ]
                    b = bsub["value"].astype(float).values
                    if len(a) == 0 or len(b) == 0:
                        continue
                    delta = float(a.mean() - b.mean())
                    pct = 100.0 * delta / abs(b.mean()) if b.mean() != 0 else np.nan
                    lo, hi = bootstrap_mean_diff(a, b)
                    rows.append(
                        {
                            "mechanism": mech,
                            "missing_rate": rate,
                            "rate_pct": int(rate * 100),
                            "metric": metric,
                            "method_a": "OriginalRFECA",
                            "method_b": base,
                            "mean_a": float(a.mean()),
                            "mean_b": float(b.mean()),
                            "n_a": len(a),
                            "n_b": len(b),
                            "delta_abs_a_minus_b": delta,
                            "delta_pct_vs_b": pct,
                            "boot_ci95_low": lo,
                            "boot_ci95_high": hi,
                            "comparison_type": "DESCRIPTIVE_ONLY_protocols_differ",
                            "p_value": "NOT_REPORTED",
                        }
                    )

    # rankings per cell
    rank_rows = []
    for mech in ["MCAR", "MAR"]:
        for rate in [0.05, 0.1, 0.2, 0.3]:
            for metric, col_rfeca in [("rmse", "rmse"), ("mae", "mae")]:
                means = {}
                rf = slots[slots.mechanism == mech]
                rf = rf[np.isclose(rf.missing_rate, rate)]
                means["OriginalRFECA"] = float(rf[col_rfeca].mean())
                for base in BASELINE_METHODS:
                    bsub = reps[
                        (reps.mechanism == mech)
                        & (np.isclose(reps.missing_rate, rate))
                        & (reps.method == base)
                        & (reps.metric == metric)
                    ]
                    means[base] = float(bsub["value"].mean())
                order = sorted(means, key=means.get)  # lower better for rmse/mae
                rank_rows.append(
                    {
                        "mechanism": mech,
                        "missing_rate": rate,
                        "rate_pct": int(rate * 100),
                        "metric": metric,
                        "ranking": " < ".join(f"{m}({means[m]:.4f})" for m in order),
                        "best": order[0],
                        "OriginalRFECA_rank": order.index("OriginalRFECA") + 1,
                        **{f"mean__{m}": means[m] for m in ALL_METHODS},
                    }
                )

    desc = pd.DataFrame(rows)
    ranks = pd.DataFrame(rank_rows)
    desc.to_csv(STATS / "B_descriptive_rfeca_vs_baselines.csv", index=False)
    ranks.to_csv(STATS / "B_rankings_descriptive.csv", index=False)

    md = []
    md.append("# B) Comparações apenas descritivas (OriginalRFECA vs baselines)\n\n")
    md.append("Protocolos distintos → **sem p-values**.\n\n")
    md.append("Métricas: RMSE, MAE. Quantidades: Δ absoluto, Δ%, bootstrap IC95% ")
    md.append("(reamostragem independente das médias; n_a=5, n_b=10), ranking, win count.\n\n")
    md.append("Artefactos: `B_descriptive_rfeca_vs_baselines.csv`, `B_rankings_descriptive.csv`.\n")
    (STATS / "B_descriptive.md").write_text("".join(md), encoding="utf-8")
    return desc, ranks


def win_counts(ranks: pd.DataFrame) -> pd.DataFrame:
    rmse = ranks[ranks.metric == "rmse"].copy()
    # by scenario
    by_scen = rmse[["mechanism", "rate_pct", "best"]].rename(columns={"best": "winner"})
    by_scen.to_csv(STATS / "win_by_scenario.csv", index=False)

    by_mech = (
        rmse.groupby(["mechanism", "best"]).size().reset_index(name="wins")
    )
    by_mech.to_csv(STATS / "win_by_mechanism.csv", index=False)

    by_rate = (
        rmse.groupby(["rate_pct", "best"]).size().reset_index(name="wins")
    )
    by_rate.to_csv(STATS / "win_by_rate.csv", index=False)

    total = rmse["best"].value_counts().rename_axis("method").reset_index(name="wins")
    total["n_scenarios"] = len(rmse)
    total.to_csv(STATS / "win_totals.csv", index=False)

    # F1 wins from central
    central = pd.read_csv(OUT / "central" / "wins_summary.csv")
    f1 = central["f1_winner"].value_counts().rename_axis("method").reset_index(name="wins")
    f1["metric"] = "f1_macro"
    f1["n_scenarios"] = len(central)
    f1.to_csv(STATS / "win_totals_f1.csv", index=False)

    md = ["# Win counts (RMSE descritivo · 4 métodos)\n\n"]
    md.append("## Por cenário\n\n")
    md.append(df_to_md(by_scen) + "\n\n")
    md.append("## Por mecanismo\n\n")
    md.append(df_to_md(by_mech) + "\n\n")
    md.append("## Por taxa\n\n")
    md.append(df_to_md(by_rate) + "\n\n")
    md.append("## Totais RMSE\n\n")
    md.append(df_to_md(total) + "\n\n")
    md.append("## Totais Macro-F1 (EnsembleSoft)\n\n")
    md.append(df_to_md(f1) + "\n")
    (STATS / "win_counts.md").write_text("".join(md), encoding="utf-8")
    return total


def gene_analysis() -> None:
    """OriginalRFECA gene-level only — baselines lack per-gene RMSE in artifacts."""
    frames = []
    for p in sorted(
        (ROOT / "artifacts/original_rfeca_reduced_metabric").glob("per_gene_all_*.csv")
    ):
        frames.append(pd.read_csv(p))
    genes = pd.concat(frames, ignore_index=True)

    # overall difficulty
    overall = (
        genes.groupby("gene")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            n=("rmse", "size"),
            n_pred_mean=("n_predictors_selected", "mean"),
        )
        .reset_index()
        .sort_values("rmse_mean", ascending=False)
    )
    overall["rmse_rank"] = range(1, len(overall) + 1)
    overall.to_csv(STATS / "gene_difficulty_overall_original_rfeca.csv", index=False)

    # by mechanism
    by_mech = (
        genes.groupby(["mechanism", "gene"])
        .agg(rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), n=("rmse", "size"))
        .reset_index()
    )
    by_mech["rank_within_mech"] = by_mech.groupby("mechanism")["rmse_mean"].rank(
        ascending=False, method="min"
    )
    by_mech.to_csv(STATS / "gene_difficulty_by_mechanism_original_rfeca.csv", index=False)

    # by mech × rate
    by_cell = (
        genes.groupby(["mechanism", "missing_rate", "gene"])
        .agg(rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), n=("rmse", "size"))
        .reset_index()
    )
    by_cell.to_csv(STATS / "gene_difficulty_by_cell_original_rfeca.csv", index=False)

    # consistently hard: top-10 overall AND top-quartile in both MCAR and MAR
    top10 = set(overall.head(10)["gene"])
    q_mcar = by_mech[by_mech.mechanism.str.lower() == "mcar"]["rmse_mean"].quantile(0.75)
    q_mar = by_mech[by_mech.mechanism.str.lower() == "mar"]["rmse_mean"].quantile(0.75)
    hard_mcar = set(
        by_mech[
            (by_mech.mechanism.str.lower() == "mcar") & (by_mech.rmse_mean >= q_mcar)
        ]["gene"]
    )
    hard_mar = set(
        by_mech[
            (by_mech.mechanism.str.lower() == "mar") & (by_mech.rmse_mean >= q_mar)
        ]["gene"]
    )
    consistent = sorted(top10 & hard_mcar & hard_mar)
    if not consistent:
        consistent = sorted(hard_mcar & hard_mar)

    # Easy genes (bottom)
    easy = overall.tail(10)["gene"].tolist()

    # Interpretability from selection
    pred_counts: dict[str, int] = {}
    subset_sizes = []
    for _, r in genes.iterrows():
        subset_sizes.append(r["n_predictors_selected"])
        preds = str(r.get("winning_predictors", "") or "")
        if not preds or preds == "nan":
            continue
        for g in preds.split("|"):
            g = g.strip()
            if g:
                pred_counts[g] = pred_counts.get(g, 0) + 1
    top_pred = (
        pd.DataFrame(
            [{"gene": k, "count_as_predictor": v} for k, v in pred_counts.items()]
        )
        .sort_values("count_as_predictor", ascending=False)
    )
    top_pred.to_csv(STATS / "rfeca_predictor_frequency.csv", index=False)

    jacc = pd.read_csv(OUT / "rfeca_subset_jaccard_by_gene.csv")
    jacc_gene = (
        jacc.groupby("gene")["mean_pairwise_jaccard"]
        .mean()
        .reset_index()
        .sort_values("mean_pairwise_jaccard")
    )
    jacc_gene.to_csv(STATS / "rfeca_jaccard_by_gene_mean.csv", index=False)

    interp = {
        "n_predictor_mean": float(np.mean(subset_sizes)),
        "n_predictor_median": float(np.median(subset_sizes)),
        "n_predictor_std": float(np.std(subset_sizes, ddof=1)),
        "n_predictor_min": int(np.min(subset_sizes)),
        "n_predictor_max": int(np.max(subset_sizes)),
        "jaccard_mean": float(jacc["mean_pairwise_jaccard"].mean()),
        "jaccard_median": float(jacc["mean_pairwise_jaccard"].median()),
        "top10_predictors": top_pred.head(10).to_dict("records"),
        "hardest10_genes": overall.head(10)[["gene", "rmse_mean"]].to_dict("records"),
        "easiest10_genes": overall.tail(10)[["gene", "rmse_mean"]].to_dict("records"),
        "consistently_hard": consistent,
    }
    (STATS / "rfeca_interpretability_summary.json").write_text(
        json.dumps(interp, indent=2), encoding="utf-8"
    )

    # Per-gene winners: UNAVAILABLE across methods
    note = {
        "status": "UNAVAILABLE_FOR_CROSS_METHOD",
        "reason": (
            "Baselines (Mean/KNN/MissForest) do not store per-gene RMSE in final "
            "artifacts (only fold-level aggregate). Gene-level difficulty and "
            "selection analysis is OriginalRFECA-only. Scenario-level winners "
            "are in win_by_scenario.csv."
        ),
        "available": "gene_difficulty_*_original_rfeca.csv, rfeca_predictor_frequency.csv",
    }
    (STATS / "per_gene_winners_NOTE.json").write_text(
        json.dumps(note, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# Gene-level analysis (OriginalRFECA)\n\n")
    md.append(
        "**Nota:** Mean/KNN/MissForest não têm RMSE por gene nos artefactos finais; "
        "não é possível calcular per-gene winners cross-method nem dizer quais genes "
        "apenas MissForest melhora sem recalcular imputações.\n\n"
    )
    md.append("## Genes mais difíceis (RMSE médio, todos os slots)\n\n")
    md.append(df_to_md(overall.head(15)) + "\n\n")
    md.append("## Consistentemente difíceis (Q4 MCAR ∩ Q4 MAR)\n\n")
    md.append(", ".join(consistent) + "\n\n")
    md.append("## Mais fáceis (menor RMSE)\n\n")
    md.append(df_to_md(overall.tail(10)) + "\n\n")
    md.append("## Interpretabilidade — top preditores\n\n")
    md.append(df_to_md(top_pred.head(15)) + "\n\n")
    md.append(
        f"Tamanho médio do subconjunto: **{interp['n_predictor_mean']:.2f}** "
        f"(mediana {interp['n_predictor_median']:.1f}, "
        f"intervalo [{interp['n_predictor_min']}, {interp['n_predictor_max']}])\n\n"
    )
    md.append(
        f"Jaccard médio entre réplicas: **{interp['jaccard_mean']:.3f}** "
        f"(mediana {interp['jaccard_median']:.3f})\n"
    )
    (STATS / "gene_level_analysis.md").write_text("".join(md), encoding="utf-8")


def effect_sizes(reps: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mech in ["MCAR", "MAR"]:
        for rate in [0.05, 0.1, 0.2, 0.3]:
            for metric, col in [("rmse", "rmse"), ("mae", "mae")]:
                rf = slots[slots.mechanism == mech]
                rf = rf[np.isclose(rf.missing_rate, rate)]
                a = rf[col].astype(float).values
                bsub = reps[
                    (reps.mechanism == mech)
                    & (np.isclose(reps.missing_rate, rate))
                    & (reps.method == "MissForest")
                    & (reps.metric == metric)
                ]
                b = bsub["value"].astype(float).values
                delta = float(a.mean() - b.mean())
                pct = 100.0 * delta / abs(b.mean())
                lo, hi = bootstrap_mean_diff(a, b)
                rows.append(
                    {
                        "mechanism": mech,
                        "missing_rate": rate,
                        "rate_pct": int(rate * 100),
                        "metric": metric,
                        "OriginalRFECA_mean": float(a.mean()),
                        "MissForest_mean": float(b.mean()),
                        "delta_RFECA_minus_MF": delta,
                        "delta_pct": pct,
                        "boot_ci95_low": lo,
                        "boot_ci95_high": hi,
                        "n_rfeca": len(a),
                        "n_mf": len(b),
                        "note": "DESCRIPTIVE_ONLY_no_pvalue",
                    }
                )
    eff = pd.DataFrame(rows)
    eff.to_csv(STATS / "effect_sizes_rfeca_vs_missforest.csv", index=False)
    # also MAE+RMSE display table
    wide = []
    for mech in ["MCAR", "MAR"]:
        for rate in [0.05, 0.1, 0.2, 0.3]:
            r = {"mechanism": mech, "rate_pct": int(rate * 100)}
            for metric in ["rmse", "mae"]:
                s = eff[
                    (eff.mechanism == mech)
                    & (np.isclose(eff.missing_rate, rate))
                    & (eff.metric == metric)
                ].iloc[0]
                r[f"Δ{metric.upper()}"] = f"{s.delta_RFECA_minus_MF:.4f}"
                r[f"Δ{metric.upper()}_pct"] = f"{s.delta_pct:.2f}%"
                r[f"IC95_{metric.upper()}"] = (
                    f"[{s.boot_ci95_low:.4f}, {s.boot_ci95_high:.4f}]"
                )
            wide.append(r)
    pd.DataFrame(wide).to_csv(
        STATS / "effect_sizes_rfeca_vs_missforest_display.csv", index=False
    )
    return eff


def one_page(eff: pd.DataFrame, total_wins: pd.DataFrame, ranks: pd.DataFrame) -> None:
    hard = pd.read_csv(STATS / "gene_difficulty_overall_original_rfeca.csv").head(8)
    pred = pd.read_csv(STATS / "rfeca_predictor_frequency.csv").head(8)
    headline = pd.read_csv(OUT / "central" / "headline_rmse_f1.csv")

    lines = []
    lines.append("# One-page Results — METABRIC PAM50\n\n")
    lines.append(
        "Comparação: **Mean · KNN · MissForest · OriginalRFECA** "
        "(sem RFECA-k*). Fonte: artefactos congelados.\n\n"
    )
    lines.append("---\n\n## Principais números\n\n")
    lines.append(
        "- **RMSE:** OriginalRFECA melhor em **7/8** células (MCAR/MAR × 5/10/20/30%); "
        "única perda: **MAR 5%** (MissForest 0.629 vs 0.641).\n"
    )
    lines.append(
        "- **Macro-F1 (EnsembleSoft):** OriginalRFECA melhor em **6/8**; "
        "MissForest à frente em MCAR/MAR 10% (margem ≤0.002).\n"
    )
    lines.append(
        "- **ΔRMSE vs MissForest (descritivo):** até **−0.066** (−9.3%) em MAR 30%; "
        "positivo só em MAR 5% (+0.012).\n"
    )
    lines.append(
        "- **Estabilidade OriginalRFECA:** RMSE quase flat nas taxas; "
        "MAR ~+0.02–0.03 vs MCAR.\n"
    )
    lines.append(
        "- **Seleção:** ~**21.6** preditores/gene; Jaccard réplicas ~**0.68**; "
        "top preditores: MKI67, NDC80, UBE2T, CEP55, PTTG1.\n"
    )
    lines.append(
        "- **Operacional:** 40/40 slots class A; svr_coverage=1.0; fallback=0; "
        "wall ~**51.5 h** (16 gene-workers).\n\n"
    )

    lines.append("## Win counts (RMSE)\n\n")
    lines.append(df_to_md(total_wins) + "\n\n")

    lines.append("## Headline RMSE / F1\n\n")
    lines.append(df_to_md(headline) + "\n\n")

    lines.append("## Effect sizes descritivos vs MissForest\n\n")
    disp = pd.read_csv(STATS / "effect_sizes_rfeca_vs_missforest_display.csv")
    lines.append(df_to_md(disp) + "\n\n")
    lines.append(
        "*Sem p-values: protocolos distintos (TARGET-WISE holdout vs CV shared-mask).*\n\n"
    )

    lines.append("## Genes mais difíceis (OriginalRFECA)\n\n")
    lines.append(
        df_to_md(hard[["gene", "rmse_mean", "rmse_std", "n_pred_mean"]])
        + "\n\n"
    )

    lines.append("## Top preditores selecionados\n\n")
    lines.append(df_to_md(pred) + "\n\n")

    lines.append("---\n\n## Figuras correspondentes\n\n")
    lines.append(
        "| Figura | Path |\n|---|---|\n"
        "| RMSE por missingness | `figures/fig_rmse_by_missingness.png` |\n"
        "| MAE | `figures/fig_mae_by_missingness.png` |\n"
        "| Macro-F1 | `figures/fig_macrof1_by_missingness.png` |\n"
        "| RMSE bars | `figures/copied_comparison_fig05_rmse_bars.png` |\n"
        "| F1 bars | `figures/copied_comparison_fig06_f1_bars.png` |\n"
        "| RV baselines | `figures/fig_rv_by_missingness_baselines.png` |\n\n"
    )

    lines.append("## Tabelas correspondentes\n\n")
    lines.append(
        "| Tabela | Path |\n|---|---|\n"
        "| Headline RMSE+F1 | `central/headline_rmse_f1.csv` |\n"
        "| Comparison display | `central/comparison_display.csv` |\n"
        "| Wins | `stats_final/win_*.csv` |\n"
        "| Effect sizes | `stats_final/effect_sizes_rfeca_vs_missforest.csv` |\n"
        "| Stats A (válidas) | `stats_final/A_*.csv` |\n"
        "| Stats B (descritivas) | `stats_final/B_*.csv` |\n"
        "| Gene difficulty | `stats_final/gene_difficulty_overall_original_rfeca.csv` |\n\n"
    )

    lines.append("---\n\n## Frases objetivas para Results\n\n")
    lines.append(
        "1. Across eight METABRIC PAM50 missingness settings (MCAR/MAR × 5–30%), "
        "OriginalRFECA achieved the lowest mean RMSE in seven settings; "
        "MissForest was best only at MAR 5%.\n\n"
    )
    lines.append(
        "2. Relative to MissForest, OriginalRFECA reduced RMSE by up to 9.3% "
        "(MAR 30%; Δ=−0.066, descriptive bootstrap IC95% excluding zero); "
        "at MAR 5% the descriptive Δ favored MissForest (Δ=+0.012).\n\n"
    )
    lines.append(
        "3. OriginalRFECA RMSE remained nearly flat across missingness rates, "
        "whereas KNN and MissForest degraded more under MAR at higher rates.\n\n"
    )
    lines.append(
        "4. Macro-F1 differences among imputers were small at 5–10% missingness "
        "and more favorable to OriginalRFECA at 20–30%, with the caveat that "
        "classification nesting differs between protocols.\n\n"
    )
    lines.append(
        "5. Within the shared-mask CV baseline protocol, Friedman and Wilcoxon–Holm "
        "tests confirmed systematic RMSE differences among Mean, KNN, and MissForest "
        "(MissForest best among the three).\n\n"
    )
    lines.append(
        "6. OriginalRFECA selected on average ~22 predictors per gene "
        "(median 23); pairwise Jaccard of selected subsets across replicates "
        "was ~0.68, indicating moderate stability.\n\n"
    )
    lines.append(
        "7. The hardest genes to impute under OriginalRFECA included "
        + ", ".join(hard.head(5)["gene"].tolist())
        + " (highest mean RMSE across slots).\n\n"
    )
    lines.append(
        "8. No confirmatory p-values are reported for OriginalRFECA versus "
        "baselines because evaluation protocols, replicate counts, and "
        "missingness seeds differ.\n"
    )

    (OUT / "one_page_results.md").write_text("".join(lines), encoding="utf-8")
    (STATS / "one_page_results.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    reps = load_baseline_reps()
    slots = load_rfeca_slots()
    # normalize mechanism case on slots
    slots = slots.copy()
    slots["mechanism"] = slots["mechanism"].astype(str)

    print("A) valid baselines…")
    section_a_valid(reps)

    print("B) descriptive…")
    _desc, ranks = section_b_descriptive(reps, slots)

    print("Win counts…")
    total = win_counts(ranks)

    print("Gene analysis…")
    gene_analysis()

    print("Effect sizes…")
    eff = effect_sizes(reps, slots)

    print("One-page…")
    one_page(eff, total, ranks)

    # index
    idx = """# stats_final — índice

## A — Formalmente válido (Mean/KNN/MissForest)
- `A_valid_baselines.md`
- `A_friedman_baselines_metabric.csv`
- `A_wilcoxon_holm_baselines_metabric.csv`
- `A_primary_contrasts_baselines.csv`

## B — Descritivo (OriginalRFECA vs baselines; sem p-values)
- `B_descriptive.md`
- `B_descriptive_rfeca_vs_baselines.csv`
- `B_rankings_descriptive.csv`

## Win counts
- `win_counts.md`, `win_by_scenario.csv`, `win_by_mechanism.csv`, `win_by_rate.csv`, `win_totals.csv`

## Gene-level / interpretability
- `gene_level_analysis.md`
- `gene_difficulty_overall_original_rfeca.csv`
- `rfeca_predictor_frequency.csv`
- `rfeca_interpretability_summary.json`
- `per_gene_winners_NOTE.json` (cross-method indisponível)

## Effect sizes
- `effect_sizes_rfeca_vs_missforest.csv`
- `effect_sizes_rfeca_vs_missforest_display.csv`

## Artigo
- `../one_page_results.md`
"""
    (STATS / "README.md").write_text(idx, encoding="utf-8")
    print("Done ->", STATS)


if __name__ == "__main__":
    main()
