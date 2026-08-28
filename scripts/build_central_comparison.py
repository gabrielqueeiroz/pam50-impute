"""Aggregate final comparison tables: OriginalRFECA vs Mean/KNN/MissForest only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_analysis" / "central"
METHODS = ["Mean", "KNN", "MissForest", "OriginalRFECA"]


def fmt(mean: float, std: float, nd: int = 3) -> str:
    if pd.isna(mean):
        return "n/a"
    if pd.isna(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    imp = pd.read_csv(ROOT / "artifacts/final_analysis/results_tables.csv")
    rows = []
    for _, r in imp.iterrows():
        mech = r["mechanism"]
        rate = float(r["missing_rate"])
        for m in METHODS:
            n = r.get(f"{m}_rmse_n")
            rows.append(
                {
                    "mechanism": mech,
                    "missing_rate": rate,
                    "rate_pct": int(rate * 100),
                    "method": m,
                    "rmse_mean": r.get(f"{m}_rmse_mean"),
                    "rmse_std": r.get(f"{m}_rmse_std"),
                    "mae_mean": r.get(f"{m}_mae_mean"),
                    "mae_std": r.get(f"{m}_mae_std"),
                    "rv_mean": r.get(f"{m}_rv_mean"),
                    "rv_std": r.get(f"{m}_rv_std"),
                    "n_reps": int(n) if pd.notna(n) else np.nan,
                    "protocol": (
                        "repeated_mask_holdout"
                        if m == "OriginalRFECA"
                        else "stratified_cv_shared_mask"
                    ),
                }
            )
    imp_long = pd.DataFrame(rows)
    imp_long.to_csv(OUT / "imputation_long.csv", index=False)

    t6 = pd.read_csv(
        ROOT
        / "artifacts/paper_results_original_rfeca/comparison/table06_metabric_classification.csv"
    )
    t6 = t6[t6["metric"] == "f1_macro"].copy()
    t6 = t6[~t6["method"].astype(str).str.contains("RFECA-k", na=False)]
    t6 = t6[t6["method"].isin(["Mean", "KNN", "MissForest"])]
    t6 = t6[t6["missing_rate"] > 0]

    rfeca_clf = pd.read_csv(
        ROOT
        / "artifacts/paper_results_original_rfeca/comparison/original_rfeca_classification_summary.csv"
    )
    rfeca_clf = rfeca_clf[rfeca_clf["model"] == "EnsembleSoft"].copy()

    clf_rows = []
    for _, r in t6.iterrows():
        clf_rows.append(
            {
                "mechanism": r["mechanism"],
                "missing_rate": float(r["missing_rate"]),
                "rate_pct": int(round(float(r["missing_rate"]) * 100)),
                "method": r["method"],
                "model": "EnsembleSoft",
                "f1_mean": r["mean"],
                "f1_std": r["std"],
                "f1_ci95_low": r["ci95_low"],
                "f1_ci95_high": r["ci95_high"],
                "n_reps": int(r["n_replications"]),
                "protocol": "imputer_within_cv",
            }
        )
    for _, r in rfeca_clf.iterrows():
        clf_rows.append(
            {
                "mechanism": r["mechanism"],
                "missing_rate": float(r["missing_rate"]),
                "rate_pct": int(round(float(r["missing_rate"]) * 100)),
                "method": "OriginalRFECA",
                "model": "EnsembleSoft",
                "f1_mean": r["f1_mean"],
                "f1_std": r["f1_std"],
                "f1_ci95_low": r["f1_ci95_low"],
                "f1_ci95_high": r["f1_ci95_high"],
                "n_reps": int(r["n_reps"]),
                "protocol": "post_target_wise_impute",
            }
        )
    clf_long = pd.DataFrame(clf_rows).sort_values(
        ["mechanism", "missing_rate", "method"]
    )
    clf_long.to_csv(OUT / "classification_long.csv", index=False)

    wide_rows = []
    wins = []
    for mech in ["MCAR", "MAR"]:
        for rate in [0.05, 0.10, 0.20, 0.30]:
            row: dict = {
                "mechanism": mech,
                "missing_rate": rate,
                "rate_pct": int(rate * 100),
            }
            imp_cell = imp_long[
                (imp_long.mechanism == mech)
                & (np.isclose(imp_long.missing_rate, rate))
            ]
            clf_cell = clf_long[
                (clf_long.mechanism == mech)
                & (np.isclose(clf_long.missing_rate, rate))
            ]
            rmse_vals: dict[str, float] = {}
            f1_vals: dict[str, float] = {}
            for m in METHODS:
                ir = imp_cell[imp_cell.method == m].iloc[0]
                cr = clf_cell[clf_cell.method == m].iloc[0]
                row[f"{m}_RMSE"] = fmt(ir.rmse_mean, ir.rmse_std)
                row[f"{m}_MAE"] = fmt(ir.mae_mean, ir.mae_std)
                row[f"{m}_RV"] = (
                    fmt(ir.rv_mean, ir.rv_std) if pd.notna(ir.rv_mean) else "n/a"
                )
                row[f"{m}_F1"] = fmt(cr.f1_mean, cr.f1_std)
                row[f"{m}_rmse_mean"] = float(ir.rmse_mean)
                row[f"{m}_f1_mean"] = float(cr.f1_mean)
                rmse_vals[m] = float(ir.rmse_mean)
                f1_vals[m] = float(cr.f1_mean)

            rmse_rank = sorted(rmse_vals, key=rmse_vals.get)
            f1_rank = sorted(f1_vals, key=f1_vals.get, reverse=True)
            row["rmse_best"] = rmse_rank[0]
            row["rmse_ranking"] = " < ".join(
                f"{m}({rmse_vals[m]:.3f})" for m in rmse_rank
            )
            row["f1_best"] = f1_rank[0]
            row["f1_ranking"] = " > ".join(
                f"{m}({f1_vals[m]:.3f})" for m in f1_rank
            )
            row["OriginalRFECA_minus_MissForest_RMSE"] = (
                rmse_vals["OriginalRFECA"] - rmse_vals["MissForest"]
            )
            row["OriginalRFECA_minus_MissForest_F1"] = (
                f1_vals["OriginalRFECA"] - f1_vals["MissForest"]
            )
            wide_rows.append(row)
            wins.append(
                {
                    "mechanism": mech,
                    "rate_pct": int(rate * 100),
                    "rmse_winner": rmse_rank[0],
                    "f1_winner": f1_rank[0],
                    "OriginalRFECA_rmse_rank": rmse_rank.index("OriginalRFECA") + 1,
                    "OriginalRFECA_f1_rank": f1_rank.index("OriginalRFECA") + 1,
                    "delta_rmse_vs_MF": rmse_vals["OriginalRFECA"]
                    - rmse_vals["MissForest"],
                    "delta_f1_vs_MF": f1_vals["OriginalRFECA"] - f1_vals["MissForest"],
                    "delta_rmse_vs_KNN": rmse_vals["OriginalRFECA"] - rmse_vals["KNN"],
                    "delta_f1_vs_KNN": f1_vals["OriginalRFECA"] - f1_vals["KNN"],
                    "delta_rmse_vs_Mean": rmse_vals["OriginalRFECA"]
                    - rmse_vals["Mean"],
                    "delta_f1_vs_Mean": f1_vals["OriginalRFECA"] - f1_vals["Mean"],
                }
            )

    wide = pd.DataFrame(wide_rows)
    disp_cols = (
        ["mechanism", "rate_pct"]
        + [c for c in wide.columns if c.endswith(("_RMSE", "_MAE", "_RV", "_F1"))]
        + [
            "rmse_best",
            "rmse_ranking",
            "f1_best",
            "f1_ranking",
            "OriginalRFECA_minus_MissForest_RMSE",
            "OriginalRFECA_minus_MissForest_F1",
        ]
    )
    wide[disp_cols].to_csv(OUT / "comparison_display.csv", index=False)
    wide.to_csv(OUT / "comparison_wide.csv", index=False)

    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(OUT / "wins_summary.csv", index=False)

    compact = wide[
        [
            "mechanism",
            "rate_pct",
            "Mean_RMSE",
            "KNN_RMSE",
            "MissForest_RMSE",
            "OriginalRFECA_RMSE",
            "Mean_F1",
            "KNN_F1",
            "MissForest_F1",
            "OriginalRFECA_F1",
            "rmse_best",
            "f1_best",
        ]
    ].copy()
    compact.to_csv(OUT / "headline_rmse_f1.csv", index=False)

    summary = {
        "methods_compared": METHODS,
        "excluded": ["RFECA-k5", "RFECA-k10", "RFECA-k20"],
        "n_cells": 8,
        "rmse_wins": wins_df["rmse_winner"].value_counts().to_dict(),
        "f1_wins": wins_df["f1_winner"].value_counts().to_dict(),
        "original_rfeca_rmse_wins": int(
            (wins_df.rmse_winner == "OriginalRFECA").sum()
        ),
        "original_rfeca_f1_wins": int((wins_df.f1_winner == "OriginalRFECA").sum()),
        "only_rmse_loss": wins_df.loc[
            wins_df.rmse_winner != "OriginalRFECA",
            ["mechanism", "rate_pct", "rmse_winner"],
        ].to_dict("records"),
        "sources": {
            "imputation": "artifacts/final_analysis/results_tables.csv",
            "classification_baselines": (
                "artifacts/paper_results_original_rfeca/comparison/"
                "table06_metabric_classification.csv"
            ),
            "classification_rfeca": (
                "artifacts/paper_results_original_rfeca/comparison/"
                "original_rfeca_classification_summary.csv"
            ),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Wrote to", OUT)


if __name__ == "__main__":
    main()
