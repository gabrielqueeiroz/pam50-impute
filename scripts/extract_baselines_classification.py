#!/usr/bin/env python3
"""
Extract Mean / KNN / MissForest classification metrics from METABRIC full runs.

Sources:
  MCAR: artifacts/metabric_full_20260724_185916 (EnsembleSoft @ 5/10/20/30)
  MAR:  artifacts/metabric_full_mar_20260725_062517 (EnsembleSoft @ 5/10/20/30)
  Multi-clf (20/30 only): metabric_multiclf_{mcar,mar}_*

Writes:
  artifacts/original_rfeca_reduced_metabric/classification/baselines_*.csv
  artifacts/original_rfeca_reduced_metabric/classification/unified_imputer_clf.csv
    (baselines + RFECA when RFECA classification exists)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "original_rfeca_reduced_metabric" / "classification"

MCAR_FULL = ART / "metabric_full_20260724_185916"
MAR_FULL = ART / "metabric_full_mar_20260725_062517"
MCAR_MULTI = ART / "metabric_multiclf_mcar_20260729_031215"
MAR_MULTI = ART / "metabric_multiclf_mar_20260729_184033"

IMPUTER_MAP = {
    "SimpleMean": "Mean",
    "KNN(k=5,dist)": "KNN",
    "MissForest": "MissForest",
}
KEEP_IMPUTERS = set(IMPUTER_MAP)
RATES = [0.05, 0.10, 0.20, 0.30]


def _load_raw(path: Path, mechanism: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["imputer"].isin(KEEP_IMPUTERS)].copy()

    def _norm_rate(r: float) -> float:
        for t in RATES:
            if abs(float(r) - t) < 1e-9:
                return t
        return float(r)

    df["missing_rate"] = df["missing_rate"].map(_norm_rate)
    df = df[df["missing_rate"].isin(RATES)].copy()
    df["mechanism"] = mechanism
    df["imputer_display"] = df["imputer"].map(IMPUTER_MAP)
    df["protocol"] = "imputer_within_cv"
    df["source_artifact"] = path.parent.name
    return df


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    g = (
        raw.groupby(
            ["mechanism", "imputer_display", "imputer", "missing_rate", "model"],
            as_index=False,
        )
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            bal_mean=("bal_acc", "mean"),
            bal_std=("bal_acc", "std"),
            n_rows=("f1_macro", "count"),
        )
    )
    return g


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []

    # EnsembleSoft from primary fulls (all 4 rates)
    for mech, path in (("mcar", MCAR_FULL), ("mar", MAR_FULL)):
        raw_path = path / "exp2_classification_raw.csv"
        if not raw_path.exists():
            print(f"MISSING {raw_path}", file=sys.stderr)
            return 1
        df = _load_raw(raw_path, mech)
        # fulls only have EnsembleSoft
        parts.append(df)
        print(f"{mech} full: {len(df)} rows models={sorted(df.model.unique())} rates={sorted(df.missing_rate.unique())}")

    # Additional classifiers from multiclf (20/30)
    for mech, path in (("mcar", MCAR_MULTI), ("mar", MAR_MULTI)):
        raw_path = path / "exp2_classification_raw.csv"
        if not raw_path.exists():
            print(f"WARN missing multiclf {raw_path}")
            continue
        df = _load_raw(raw_path, mech)
        # drop EnsembleSoft duplicates already covered by fulls at 20/30
        df = df[df["model"] != "EnsembleSoft"].copy()
        parts.append(df)
        print(
            f"{mech} multiclf (non-EnsembleSoft): {len(df)} rows "
            f"models={sorted(df.model.unique())} rates={sorted(df.missing_rate.unique())}"
        )

    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(OUT / "baselines_classification_raw.csv", index=False)
    summary = _summarize(raw)
    summary.to_csv(OUT / "baselines_classification_summary.csv", index=False)

    # EnsembleSoft-only compact for figures
    soft = summary[summary["model"] == "EnsembleSoft"].copy()
    soft.to_csv(OUT / "baselines_ensemblesoft_by_rate.csv", index=False)

    # Unified with RFECA if available
    rfeca_raw_path = OUT / "exp2_classification_raw.csv"
    unified_rows = []
    for _, r in summary.iterrows():
        unified_rows.append(
            {
                "imputer": r["imputer_display"],
                "source_imputer": r["imputer"],
                "mechanism": r["mechanism"],
                "missing_rate": r["missing_rate"],
                "model": r["model"],
                "f1_mean": r["f1_mean"],
                "f1_std": r["f1_std"],
                "bal_mean": r["bal_mean"],
                "bal_std": r["bal_std"],
                "n_rows": r["n_rows"],
                "protocol": "imputer_within_cv",
            }
        )

    if rfeca_raw_path.exists():
        rf = pd.read_csv(rfeca_raw_path)
        if "mechanism" not in rf.columns:
            # try to infer from path columns if present
            raise SystemExit("RFECA classification raw missing mechanism column")
        rf_sum = (
            rf.groupby(["mechanism", "missing_rate", "model"], as_index=False)
            .agg(
                f1_mean=("f1_macro", "mean"),
                f1_std=("f1_macro", "std"),
                bal_mean=("bal_acc", "mean"),
                bal_std=("bal_acc", "std"),
                n_rows=("f1_macro", "count"),
            )
        )
        for _, r in rf_sum.iterrows():
            unified_rows.append(
                {
                    "imputer": "RFECA",
                    "source_imputer": "OriginalRFECA",
                    "mechanism": r["mechanism"],
                    "missing_rate": r["missing_rate"],
                    "model": r["model"],
                    "f1_mean": r["f1_mean"],
                    "f1_std": r["f1_std"],
                    "bal_mean": r["bal_mean"],
                    "bal_std": r["bal_std"],
                    "n_rows": r["n_rows"],
                    "protocol": "post_target_wise_impute",
                }
            )
        print(f"Merged RFECA classification: {len(rf_sum)} summary rows")
    else:
        print("RFECA classification not yet available; baselines-only unified CSV")

    unified = pd.DataFrame(unified_rows)
    unified.to_csv(OUT / "unified_imputer_clf.csv", index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mcar_full": str(MCAR_FULL.relative_to(ROOT)).replace("\\", "/"),
        "mar_full": str(MAR_FULL.relative_to(ROOT)).replace("\\", "/"),
        "mcar_multiclf": str(MCAR_MULTI.relative_to(ROOT)).replace("\\", "/")
        if MCAR_MULTI.exists()
        else None,
        "mar_multiclf": str(MAR_MULTI.relative_to(ROOT)).replace("\\", "/")
        if MAR_MULTI.exists()
        else None,
        "n_baseline_raw_rows": int(len(raw)),
        "n_unified_rows": int(len(unified)),
        "rates": RATES,
        "baseline_imputers": list(IMPUTER_MAP.values()),
        "note": (
            "EnsembleSoft available at 5/10/20/30 from metabric_full. "
            "SVC/LogReg/RF/GB from multiclf at 20/30 only."
        ),
    }
    (OUT / "baselines_extract_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
