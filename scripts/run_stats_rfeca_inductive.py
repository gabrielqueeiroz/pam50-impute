#!/usr/bin/env python3
"""
Post-hoc stats for inductive RFECA-only full runs (seed scheme v2).

Compares RFECA(k=5/10/20) within each cohort×mechanism. Does NOT merge with
legacy Mean/KNN/MissForest artifacts (different masks under v2).

Usage (repo root):
  python scripts/run_stats_rfeca_inductive.py
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

from bcimpute.stats import friedman_table, pairwise_wilcoxon_table  # noqa: E402

# Fill in whatever inductive RFECA artifacts currently exist.
CANDIDATE_RUNS = [
    {
        "cohort": "cptac_2c",
        "mechanism": "mcar",
        "dir": ROOT / "artifacts" / "discovery_full_rfeca_20260727_161656",
    },
    {
        "cohort": "metabric",
        "mechanism": "mcar",
        "dir": ROOT / "artifacts" / "metabric_full_rfeca_20260727_163213",
    },
    # MAR placeholders — picked up automatically when present (latest match below).
]

RFECA_IMPUTERS = ["RFECA_SVR(k=5)", "RFECA_SVR(k=10)", "RFECA_SVR(k=20)"]


def _latest_rfeca_dirs() -> list[dict]:
    """Prefer explicit candidates; also discover any newer *_full*_rfeca_* dirs."""
    runs = [r for r in CANDIDATE_RUNS if r["dir"].exists()]
    art = ROOT / "artifacts"
    # Discover MAR rfeca dirs if not listed yet
    for cohort_key, cohort_name in [
        ("discovery_full_mar_rfeca_", "cptac_2c"),
        ("metabric_full_mar_rfeca_", "metabric"),
        ("discovery_full_rfeca_", "cptac_2c"),
        ("metabric_full_rfeca_", "metabric"),
    ]:
        matches = sorted(art.glob(f"{cohort_key}*"), key=lambda p: p.name)
        if not matches:
            continue
        d = matches[-1]
        mech = "mar" if "_mar_" in d.name else "mcar"
        # skip if already present
        if any(r["dir"] == d for r in runs):
            continue
        # for mcar, prefer explicit candidates already added
        if mech == "mcar" and any(
            r["cohort"] == cohort_name and r["mechanism"] == "mcar" for r in runs
        ):
            continue
        runs.append({"cohort": cohort_name, "mechanism": mech, "dir": d})
    return runs


def _load(run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = run["dir"]
    cls = pd.read_csv(d / "exp2_classification_raw.csv")
    if "model" in cls.columns:
        cls = cls[cls["model"] == "EnsembleSoft"].copy()
    imp = pd.read_csv(d / "exp1_imputation_raw.csv")
    return imp, cls


def _descriptives(imp: pd.DataFrame, cls: pd.DataFrame, run: dict) -> pd.DataFrame:
    rows = []
    for rate in sorted(cls["missing_rate"].unique()):
        for imputer in RFECA_IMPUTERS:
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
                "artifact_dir": str(run["dir"]),
            }
            if float(rate) > 0 and not i.empty and i["rmse"].notna().any():
                row["rmse_mean"] = float(i["rmse"].mean())
                row["rmse_std"] = float(i["rmse"].std(ddof=1)) if len(i) > 1 else float("nan")
                if "corr_rv" in i.columns and i["corr_rv"].notna().any():
                    row["corr_rv_mean"] = float(i["corr_rv"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    runs = _latest_rfeca_dirs()
    if not runs:
        raise FileNotFoundError("No inductive RFECA artifacts found under artifacts/")

    out = ROOT / "artifacts" / (
        f"stats_rfeca_inductive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    desc_parts = []
    pair_parts = []
    fried_parts = []
    for run in runs:
        imp, cls = _load(run)
        desc_parts.append(_descriptives(imp, cls, run))
        present = [i for i in RFECA_IMPUTERS if i in set(cls["imputer"])]
        pair_parts.append(
            pairwise_wilcoxon_table(
                cls,
                "f1_macro",
                cohort=run["cohort"],
                mechanism=run["mechanism"],
                imputers=present,
                higher_is_better=True,
            )
        )
        fried_parts.append(
            friedman_table(
                cls,
                "f1_macro",
                cohort=run["cohort"],
                mechanism=run["mechanism"],
                imputers=present,
                higher_is_better=True,
            )
        )
        imp_pos = imp[imp["missing_rate"] > 0].dropna(subset=["rmse"])
        if not imp_pos.empty:
            present_i = [i for i in RFECA_IMPUTERS if i in set(imp_pos["imputer"])]
            pair_parts.append(
                pairwise_wilcoxon_table(
                    imp_pos,
                    "rmse",
                    cohort=run["cohort"],
                    mechanism=run["mechanism"],
                    imputers=present_i,
                    higher_is_better=False,
                )
            )
            fried_parts.append(
                friedman_table(
                    imp_pos,
                    "rmse",
                    cohort=run["cohort"],
                    mechanism=run["mechanism"],
                    imputers=present_i,
                    higher_is_better=False,
                )
            )

    descriptives = pd.concat(desc_parts, ignore_index=True)
    pairwise = pd.concat([p for p in pair_parts if not p.empty], ignore_index=True)
    friedman = pd.concat([p for p in fried_parts if not p.empty], ignore_index=True)

    files = {
        "descriptives_rfeca.csv": descriptives,
        "pairwise_rfeca.csv": pairwise,
        "friedman_rfeca.csv": friedman,
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False)

    bullets = [
        "Inductive RFECA-only consolidation (seed scheme v2).",
        "Compares k=5/10/20 only; do not merge with legacy Mean/KNN/MissForest masks.",
        f"Runs included: {len(runs)}.",
    ]
    for run in runs:
        bullets.append(f"- {run['cohort']} {run['mechanism'].upper()}: `{run['dir'].name}`")

    # METABRIC F1 / RMSE Friedman highlights
    fr = friedman[friedman["cohort"] == "metabric"].sort_values(
        ["metric", "mechanism", "missing_rate"]
    )
    for _, r in fr.iterrows():
        bullets.append(
            f"Friedman | METABRIC {r['mechanism'].upper()} {r['metric']} @ {r['missing_rate']:.0%}: "
            f"chi2={r['friedman_chi2']:.3f}, p={r['p_value']:.4g}."
        )

    # pairwise k20 vs k5 RMSE / F1 on METABRIC MCAR
    sub = pairwise[
        (pairwise["cohort"] == "metabric")
        & (pairwise["mechanism"] == "mcar")
        & (
            (
                (pairwise["method_a"] == "RFECA_SVR(k=20)")
                & (pairwise["method_b"] == "RFECA_SVR(k=5)")
            )
            | (
                (pairwise["method_a"] == "RFECA_SVR(k=5)")
                & (pairwise["method_b"] == "RFECA_SVR(k=20)")
            )
        )
    ].sort_values(["metric", "missing_rate"])
    for _, r in sub.iterrows():
        # orient as k20 - k5
        if r["method_a"] == "RFECA_SVR(k=20)":
            d, lo, hi = r["delta_a_minus_b"], r["delta_ci95_low"], r["delta_ci95_high"]
        else:
            d = -r["delta_a_minus_b"]
            lo, hi = -r["delta_ci95_high"], -r["delta_ci95_low"]
        bullets.append(
            f"k20 vs k5 | METABRIC MCAR {r['metric']} @ {r['missing_rate']:.0%}: "
            f"Δ={d:+.4f} [{lo:+.4f}, {hi:+.4f}], p_Holm={r['p_holm']:.4g}."
        )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "rfeca_inductive_v2",
        "runs": [{**{k: v for k, v in r.items() if k != "dir"}, "dir": str(r["dir"])} for r in runs],
        "n_descriptives": int(len(descriptives)),
        "n_pairwise": int(len(pairwise)),
        "n_friedman": int(len(friedman)),
        "output_dir": str(out),
        "bullets": bullets,
        "note": (
            "Cannot fairly compare these RFECA results to legacy full-benchmark "
            "Mean/KNN/MissForest without re-running those imputers under seed scheme v2."
        ),
    }
    (out / "stats_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        "# Inductive RFECA stats (seed scheme v2)",
        "",
        f"- Generated (UTC): `{meta['generated_at_utc']}`",
        "",
        "## Method",
        "- Unit: mean over 5 CV folds within replicate (n=10).",
        "- Imputers: RFECA_SVR(k=5/10/20) only (inductive fit).",
        "- Tests: Friedman omnibus; Wilcoxon + Holm; bootstrap CI95 on paired delta.",
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

    print(f"Wrote stats to {out}")
    for b in bullets:
        print(" -", b.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
