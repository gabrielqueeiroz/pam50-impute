#!/usr/bin/env python3
"""
Consolidated post-hoc stats across MCAR + MAR full benchmarks.

Runs Wilcoxon signed-rank (paired by replicate), matched-pairs rank-biserial,
Holm correction, and Friedman omnibus across imputers.
Primary families: RFECA/MissForest vs SimpleMean/KNN.

Usage (repo root):
  python scripts/run_stats_posthoc.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.stats import (  # noqa: E402
    friedman_table,
    pairwise_wilcoxon_table,
    primary_contrasts_table,
)

RUNS = [
    {
        "cohort": "discovery",
        "mechanism": "mcar",
        "dir": ROOT / "artifacts" / "discovery_full_20260724_172348",
    },
    {
        "cohort": "metabric",
        "mechanism": "mcar",
        "dir": ROOT / "artifacts" / "metabric_full_20260724_185916",
    },
    {
        "cohort": "discovery",
        "mechanism": "mar",
        "dir": ROOT / "artifacts" / "discovery_full_mar_20260725_045052",
    },
    {
        "cohort": "metabric",
        "mechanism": "mar",
        "dir": ROOT / "artifacts" / "metabric_full_mar_20260725_062517",
    },
]

IMPUTERS = [
    "SimpleMean",
    "KNN(k=5,dist)",
    "RFECA_SVR(k=5)",
    "RFECA_SVR(k=10)",
    "RFECA_SVR(k=20)",
    "MissForest",
]

# Reduced set for a cleaner Friedman (one RFECA k pre-specified as k=20).
FRIEDMAN_IMPUTERS = [
    "SimpleMean",
    "KNN(k=5,dist)",
    "RFECA_SVR(k=20)",
    "MissForest",
]


def _load_run(run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = run["dir"]
    cls = pd.read_csv(d / "exp2_classification_raw.csv")
    if "model" in cls.columns:
        cls = cls[cls["model"] == "EnsembleSoft"].copy()
    imp = pd.read_csv(d / "exp1_imputation_raw.csv")
    # drop rate 0 imputation rows that are all-NaN metrics (still keep for cls)
    return imp, cls


def _summary_descriptives(imp: pd.DataFrame, cls: pd.DataFrame, run: dict) -> pd.DataFrame:
    rows = []
    for rate in sorted(cls["missing_rate"].unique()):
        for imputer in IMPUTERS:
            c = cls[(cls["imputer"] == imputer) & (cls["missing_rate"] == rate)]
            i = imp[(imp["imputer"] == imputer) & (imp["missing_rate"] == rate)]
            if c.empty:
                continue
            row = {
                "cohort": run["cohort"],
                "mechanism": run["mechanism"],
                "imputer": imputer,
                "missing_rate": float(rate),
                "f1_mean": float(c["f1_macro"].mean()),
                "f1_std": float(c["f1_macro"].std(ddof=1)) if len(c) > 1 else float("nan"),
                "bal_mean": float(c["bal_acc"].mean()),
                "bal_std": float(c["bal_acc"].std(ddof=1)) if len(c) > 1 else float("nan"),
                "n_cls_rows": int(len(c)),
            }
            if float(rate) > 0 and not i.empty and i["rmse"].notna().any():
                row.update(
                    {
                        "rmse_mean": float(i["rmse"].mean()),
                        "rmse_std": float(i["rmse"].std(ddof=1)) if len(i) > 1 else float("nan"),
                        "mae_mean": float(i["mae"].mean()),
                        "mae_std": float(i["mae"].std(ddof=1)) if len(i) > 1 else float("nan"),
                    }
                )
                if "corr_rv" in i.columns and i["corr_rv"].notna().any():
                    row["corr_rv_mean"] = float(i["corr_rv"].mean())
                    row["corr_frobenius_rel_mean"] = float(i["corr_frobenius_rel"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(out: Path, bullets: list[str], files: list[str]) -> None:
    lines = [
        "# Consolidated stats report (MCAR + MAR)",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Method",
        "- Unit: mean over 5 CV folds within each replicate (n_blocks=n_pairs=10).",
        "- Omnibus: Friedman chi-square across imputers (blocks = replicates).",
        "- Post-hoc: Wilcoxon signed-rank (paired) + matched-pairs rank-biserial.",
        "- Effect CI: percentile bootstrap 95% for mean paired delta (n_boot=5000, replicate-level).",
        "- Multiple testing: Holm within each (cohort × mechanism × metric × rate).",
        "- Primary family: MissForest / RFECA(k) vs SimpleMean / KNN.",
        "- Friedman tables: (a) all 6 imputers; (b) reduced set Mean/KNN/RFECA(k=20)/MissForest.",
        "",
        "## Highlights",
    ]
    for b in bullets:
        lines.append(f"- {b}")
    lines += ["", "## Artifacts"]
    for f in files:
        lines.append(f"- `{f}`")
    lines.append("")
    (out / "stats_report.md").write_text("\n".join(lines), encoding="utf-8")


def _append_metric_tests(
    pair_parts: list,
    fried_all: list,
    fried_red: list,
    raw: pd.DataFrame,
    value_col: str,
    run: dict,
    *,
    higher_is_better: bool,
) -> None:
    present = [i for i in IMPUTERS if i in set(raw["imputer"])]
    if len(present) < 2:
        return
    pair_parts.append(
        pairwise_wilcoxon_table(
            raw,
            value_col,
            cohort=run["cohort"],
            mechanism=run["mechanism"],
            imputers=present,
            higher_is_better=higher_is_better,
        )
    )
    fried_all.append(
        friedman_table(
            raw,
            value_col,
            cohort=run["cohort"],
            mechanism=run["mechanism"],
            imputers=present,
            higher_is_better=higher_is_better,
        )
    )
    red = [i for i in FRIEDMAN_IMPUTERS if i in set(raw["imputer"])]
    if len(red) >= 2:
        fried_red.append(
            friedman_table(
                raw,
                value_col,
                cohort=run["cohort"],
                mechanism=run["mechanism"],
                imputers=red,
                higher_is_better=higher_is_better,
            )
        )


def main() -> int:
    out = ROOT / "artifacts" / f"stats_mcar_mar_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    desc_parts = []
    pair_parts: list[pd.DataFrame] = []
    fried_all_parts: list[pd.DataFrame] = []
    fried_red_parts: list[pd.DataFrame] = []
    for run in RUNS:
        if not run["dir"].exists():
            raise FileNotFoundError(run["dir"])
        imp, cls = _load_run(run)
        desc_parts.append(_summary_descriptives(imp, cls, run))

        _append_metric_tests(
            pair_parts, fried_all_parts, fried_red_parts, cls, "f1_macro", run, higher_is_better=True
        )
        imp_pos = imp[imp["missing_rate"] > 0].dropna(subset=["rmse"])
        if not imp_pos.empty:
            _append_metric_tests(
                pair_parts,
                fried_all_parts,
                fried_red_parts,
                imp_pos,
                "rmse",
                run,
                higher_is_better=False,
            )
        if "corr_rv" in imp.columns:
            imp_cov = imp[imp["missing_rate"] > 0].dropna(subset=["corr_rv"])
            if not imp_cov.empty:
                _append_metric_tests(
                    pair_parts,
                    fried_all_parts,
                    fried_red_parts,
                    imp_cov,
                    "corr_rv",
                    run,
                    higher_is_better=True,
                )

    descriptives = pd.concat(desc_parts, ignore_index=True)
    pairwise = pd.concat([p for p in pair_parts if not p.empty], ignore_index=True)
    primary = primary_contrasts_table(pairwise)
    friedman_all = pd.concat([p for p in fried_all_parts if not p.empty], ignore_index=True)
    friedman_reduced = pd.concat([p for p in fried_red_parts if not p.empty], ignore_index=True)

    sig = primary[
        (primary["metric"] == "f1_macro")
        & (primary["cohort"] == "metabric")
        & (primary["p_holm_primary_family"] < 0.05)
    ].copy()

    files = {
        "descriptives_by_imputer.csv": descriptives,
        "pairwise_all.csv": pairwise,
        "primary_contrasts.csv": primary,
        "primary_metabric_f1_sig_holm05.csv": sig,
        "friedman_all_imputers.csv": friedman_all,
        "friedman_reduced_imputers.csv": friedman_reduced,
        "primary_metabric_rmse_vs_mean.csv": primary[
            (primary["cohort"] == "metabric")
            & (primary["metric"] == "rmse")
            & (primary["method_b"] == "SimpleMean")
        ].copy(),
        "primary_metabric_f1_vs_mean.csv": primary[
            (primary["cohort"] == "metabric")
            & (primary["metric"] == "f1_macro")
            & (primary["method_b"] == "SimpleMean")
        ].copy(),
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False)

    bullets = []
    for mech in ["mcar", "mar"]:
        sub = sig[sig["mechanism"] == mech]
        bullets.append(
            f"METABRIC {mech.upper()} F1 primary contrasts with Holm p<0.05: **{len(sub)}** "
            f"(of {len(primary[(primary.cohort=='metabric')&(primary.mechanism==mech)&(primary.metric=='f1_macro')])} tested)."
        )

    # Friedman highlights: METABRIC F1 / RMSE
    fr_meta = friedman_reduced[
        (friedman_reduced["cohort"] == "metabric")
        & (friedman_reduced["metric"].isin(["f1_macro", "rmse"]))
    ].sort_values(["metric", "mechanism", "missing_rate"])
    for _, r in fr_meta.iterrows():
        bullets.append(
            f"Friedman reduced | METABRIC {r['mechanism'].upper()} {r['metric']} @ {r['missing_rate']:.0%}: "
            f"chi2={r['friedman_chi2']:.3f}, p={r['p_value']:.4g}, n_blocks={int(r['n_blocks'])}."
        )

    # RMSE primary contrasts with CI (METABRIC highlight)
    rmse_ci = primary[
        (primary["cohort"] == "metabric")
        & (primary["metric"] == "rmse")
        & (primary["method_a"].isin(["MissForest", "RFECA_SVR(k=20)"]))
        & (primary["method_b"] == "SimpleMean")
    ].sort_values(["method_a", "mechanism", "missing_rate"])
    for _, r in rmse_ci.iterrows():
        bullets.append(
            f"ΔRMSE {r['method_a']} vs Mean | {r['mechanism'].upper()} @ {r['missing_rate']:.0%}: "
            f"{r['delta_a_minus_b']:+.4f} "
            f"[{r['delta_ci95_low']:+.4f}, {r['delta_ci95_high']:+.4f}] "
            f"(bootstrap CI95, n={int(r['n_pairs'])})."
        )

    mf = primary[
        (primary["cohort"] == "metabric")
        & (primary["metric"] == "f1_macro")
        & (primary["method_a"] == "MissForest")
        & (primary["method_b"] == "SimpleMean")
    ]
    if not mf.empty:
        for _, r in mf.sort_values(["mechanism", "missing_rate"]).iterrows():
            bullets.append(
                f"MissForest vs Mean | {r['mechanism'].upper()} @ {r['missing_rate']:.0%}: "
                f"ΔF1={r['delta_a_minus_b']:+.4f}, r_rb={r['rank_biserial_r']:.3f}, "
                f"p={r['p_value']:.4g}, p_Holm={r['p_holm_primary_family']:.4g}."
            )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [{k: (str(v) if k == "dir" else v) for k, v in r.items()} for r in RUNS],
        "n_pairwise_rows": int(len(pairwise)),
        "n_primary_rows": int(len(primary)),
        "n_friedman_all_rows": int(len(friedman_all)),
        "n_friedman_reduced_rows": int(len(friedman_reduced)),
        "n_metabric_f1_sig_holm05": int(len(sig)),
        "output_dir": str(out),
        "bullets": bullets,
    }
    (out / "stats_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(out, bullets, list(files))

    print(f"Wrote stats to {out}")
    for b in bullets:
        print(" -", b.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
