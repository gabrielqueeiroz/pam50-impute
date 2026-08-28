#!/usr/bin/env python3
"""
Discovery-cohort smoke test using the same protocol as METABRIC.

Also reports metric context vs historical Colab results (not exact parity:
historical RFECA used a precomputed full-cohort correlation matrix and a
different CV budget).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.assertions import (  # noqa: E402
    assert_artificial_mask_observed_only,
    assert_fold_isolation,
    assert_mask_feature_only,
    assert_metrics_on_masked_only,
    assert_no_inplace_mutation,
    assert_no_legacy_imputed_in_metric_target,
    assert_shared_mask_across_imputers,
)
from bcimpute.config import ARTIFACT_ROOT, smoke_discovery_config  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.evaluation import (  # noqa: E402
    FoldAudit,
    run_classification_cv,
    run_imputation_cv,
    summarize_classification,
    summarize_classification_per_class,
    summarize_imputation,
)
from bcimpute.imputers import build_imputers  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402


def _historical_reference() -> dict:
    """Load comparable historical EnsembleSoft @ 10% missing if available."""
    path = (
        ROOT
        / "archive"
        / "colab_experiments"
        / "experimento 2 - k5 com scaler"
        / "exp2_classification_summary.csv"
    )
    out: dict = {"available": False, "path": str(path)}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    hit = df[
        (df["imputer"] == "RFECA_SVR(k=5)")
        & (df["model"] == "EnsembleSoft")
        & (np.isclose(df["missing_rate"], 0.1))
    ]
    knn = df[
        (df["imputer"] == "KNN(k=5,dist)")
        & (df["model"] == "EnsembleSoft")
        & (np.isclose(df["missing_rate"], 0.1))
    ] if "KNN(k=5,dist)" in set(df["imputer"]) else pd.DataFrame()
    mean = df[
        (df["imputer"] == "SimpleMean")
        & (df["model"] == "EnsembleSoft")
        & (np.isclose(df["missing_rate"], 0.1))
    ] if "SimpleMean" in set(df["imputer"]) else pd.DataFrame()

    out["available"] = True
    out["protocol_notes"] = [
        "Historical run: n_reps=10, n_splits=5 (smoke uses 2 reps / 3 splits).",
        "Historical RFECA used precomputed full-cohort correlation CSV (leakage).",
        "Corrected pipeline recomputes correlations inside each training fold.",
        "Therefore exact numeric parity is NOT expected; deviations are informative.",
    ]
    if len(hit):
        out["RFECA_SVR(k=5)"] = {
            "f1_mean": float(hit.iloc[0]["f1_mean"]),
            "bal_mean": float(hit.iloc[0]["bal_mean"]),
        }
    # KNN/Mean may live in other experiment files; try paper table / raw all
    raw_all = ROOT / "archive" / "legacy_root" / "exp2_classification_raw_all.csv"
    if raw_all.exists():
        raw = pd.read_csv(raw_all)
        for name in ["KNN(k=5,dist)", "SimpleMean", "RFECA_SVR(k=5)"]:
            sub = raw[
                (raw["imputer"] == name)
                & (raw["model"] == "EnsembleSoft")
                & (np.isclose(raw["missing_rate"], 0.1))
            ]
            if len(sub):
                out[name] = {
                    "f1_mean": float(sub["f1_macro"].mean()),
                    "f1_std": float(sub["f1_macro"].std()),
                    "bal_mean": float(sub["bal_acc"].mean()),
                    "bal_std": float(sub["bal_acc"].std()),
                    "n": int(len(sub)),
                    "source": str(raw_all),
                }
    elif len(knn) or len(mean):
        if len(knn):
            out["KNN(k=5,dist)"] = {
                "f1_mean": float(knn.iloc[0]["f1_mean"]),
                "bal_mean": float(knn.iloc[0]["bal_mean"]),
            }
        if len(mean):
            out["SimpleMean"] = {
                "f1_mean": float(mean.iloc[0]["f1_mean"]),
                "bal_mean": float(mean.iloc[0]["bal_mean"]),
            }
    return out


def _compare_prior_smoke(imp_summary, cls_summary, missing_sets) -> dict:
    """Compare against the last corrected discovery smoke (all-cell masking)."""
    prior_dir = ROOT / "artifacts" / "discovery_smoke_20260719_213142"
    out: dict = {
        "available": False,
        "prior_dir": str(prior_dir),
        "expected_difference_cause": (
            "Prior smoke sampled artificial missingness from all 117x50 cells; "
            "current primary policy restricts masking and RMSE/MAE to originally "
            "observed cells only (legacy-imputed cells excluded)."
        ),
    }
    if not prior_dir.exists():
        return out
    prior_imp = pd.read_csv(prior_dir / "exp1_imputation_summary.csv")
    prior_cls = pd.read_csv(prior_dir / "exp2_classification_summary.csv")
    out["available"] = True
    rows = []
    for _, row in cls_summary.iterrows():
        name = row["imputer"]
        prev = prior_cls[prior_cls["imputer"] == name]
        prev_i = prior_imp[prior_imp["imputer"] == name]
        entry = {"imputer": name}
        if len(prev):
            entry["prior_f1"] = float(prev.iloc[0]["f1_mean"])
            entry["current_f1"] = float(row["f1_mean"])
            entry["delta_f1"] = float(row["f1_mean"] - prev.iloc[0]["f1_mean"])
            entry["prior_bal"] = float(prev.iloc[0]["bal_mean"])
            entry["current_bal"] = float(row["bal_mean"])
            entry["delta_bal"] = float(row["bal_mean"] - prev.iloc[0]["bal_mean"])
        if len(prev_i):
            cur_i = imp_summary[imp_summary["imputer"] == name]
            if len(cur_i):
                entry["prior_rmse"] = float(prev_i.iloc[0]["rmse_mean"])
                entry["current_rmse"] = float(cur_i.iloc[0]["rmse_mean"])
                entry["delta_rmse"] = float(
                    cur_i.iloc[0]["rmse_mean"] - prev_i.iloc[0]["rmse_mean"]
                )
                entry["prior_mae"] = float(prev_i.iloc[0]["mae_mean"])
                entry["current_mae"] = float(cur_i.iloc[0]["mae_mean"])
                entry["delta_mae"] = float(
                    cur_i.iloc[0]["mae_mean"] - prev_i.iloc[0]["mae_mean"]
                )
        rows.append(entry)
    out["rows"] = rows
    # Eligible/masked accounting
    cur = next(iter(missing_sets.values()))[0]
    out["current_n_eligible"] = cur.n_eligible_cells
    out["current_n_masked"] = cur.n_artificially_masked_cells
    out["prior_n_matrix_cells"] = 117 * 50
    out["prior_approx_masked_at_10pct"] = int(round(0.10 * 117 * 50))
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mechanism",
        choices=["mcar", "mar"],
        default="mcar",
        help="Artificial missingness mechanism (default: mcar).",
    )
    args = parser.parse_args()

    cfg = smoke_discovery_config(missingness_mechanism=args.mechanism)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "discovery_smoke" if args.mechanism == "mcar" else f"discovery_smoke_{args.mechanism}"
    out_dir = ARTIFACT_ROOT / f"{prefix}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "technically_ready_for_full_benchmark": False,
        "assertions": {},
        "warnings": [],
        "errors": [],
    }
    t_wall0 = time.perf_counter()

    try:
        cohort = load_cohort("discovery")
        metabric = load_cohort("metabric")  # schema parity check

        # Same object schema / metadata fields
        schema_fields = {
            "discovery": sorted(cohort.metadata.keys()),
            "metabric": sorted(metabric.metadata.keys()),
        }
        required_meta = {
            "cohort_name",
            "n_samples",
            "n_genes",
            "gene_order",
            "label_name",
            "class_distribution",
            "project_root",
            "source_path",
            "has_native_sample_id_column",
            "matrix_schema",
        }
        missing_disc = required_meta - set(cohort.metadata)
        missing_meta = required_meta - set(metabric.metadata)
        assert not missing_disc and not missing_meta
        assert cohort.gene_names == metabric.gene_names
        assert cohort.label_name == metabric.label_name

        X_ref = cohort.X.to_numpy(dtype=float, copy=True)
        assert cohort.originally_observed_mask is not None
        obs = cohort.originally_observed_mask
        assert obs.shape == cohort.X.shape
        assert list(obs.columns) == cohort.gene_names
        # Biological IDs recovered (not synthetic discovery_row_*)
        assert not cohort.sample_ids.astype(str).str.startswith("discovery_row_").any()
        assert cohort.n_samples == 117
        assert cohort.metadata.get("n_legacy_imputed_cells", 0) > 0

        report["cohort"] = {
            "name": cohort.name,
            "scientific_identity": cohort.metadata.get("scientific_identity"),
            "n_samples": cohort.n_samples,
            "n_features": cohort.n_features,
            "class_distribution": cohort.class_distribution(),
            "sample_id_examples": list(cohort.sample_ids[:5].astype(str)),
            "n_originally_observed_cells": int(obs.to_numpy().sum()),
            "n_legacy_imputed_cells": int((~obs.to_numpy()).sum()),
            "metadata": cohort.metadata,
            "schema_fields": schema_fields,
            "metabric_loader_ok": True,
            "metabric_n": metabric.n_samples,
            "metabric_has_obs_mask": metabric.originally_observed_mask is not None,
        }

        # Value fingerprint vs preparation artifact
        fp_path = ROOT / "data" / "processed" / "discovery" / "fingerprint.json"
        if not fp_path.exists():
            fp_path = (
                ROOT / "data" / "processed" / "discovery" / "discovery_fingerprint.json"
            )
        if fp_path.exists():
            report["preparation_fingerprint"] = json.loads(
                fp_path.read_text(encoding="utf-8")
            )

        missing_sets = generate_missingness_sets(
            cohort.X,
            missing_rates=cfg.missing_rates,
            n_repetitions=cfg.n_repetitions,
            base_seed=cfg.random_state,
            originally_observed_mask=obs,
            target_cell_policy=cfg.target_cell_policy,
            mechanism=cfg.missingness_mechanism,
        )
        shared_mask_checklist = assert_shared_mask_across_imputers(
            missing_sets, cfg.imputers
        )
        for reps in missing_sets.values():
            for item in reps:
                assert_mask_feature_only(item.mask, cohort.X)
                assert_artificial_mask_observed_only(item.mask, obs.to_numpy())
                assert item.n_legacy_imputed_cells_in_cohort == int((~obs.to_numpy()).sum())
        assert_no_inplace_mutation(cohort.X, X_ref)
        report["assertions"]["mask_feature_only"] = "PASS"
        report["assertions"]["complete_matrix_not_mutated"] = "PASS"
        report["assertions"]["shared_masks_across_imputers"] = "PASS"
        report["assertions"]["artificial_mask_observed_only"] = "PASS"
        report["assertions"]["biological_ids_recovered"] = "PASS"
        report["assertions"]["observation_mask_aligned"] = "PASS"
        report["shared_mask_checklist"] = shared_mask_checklist

        imputers = build_imputers(cfg)
        for name, imp in imputers.items():
            if name.startswith("RFECA"):
                assert getattr(imp, "corr_csv_path", None) is None

        audit = FoldAudit()
        imp_parts: list[pd.DataFrame] = []
        cls_parts: list[pd.DataFrame] = []
        timings: dict[str, float] = {}

        for rate, reps in missing_sets.items():
            for item in reps:
                t0 = time.perf_counter()
                imp_part = run_imputation_cv(
                    X_full=cohort.X,
                    X_missing=item.X_missing,
                    mask=item.mask,
                    y=cohort.y,
                    imputers=imputers,
                    n_splits=cfg.n_splits,
                    random_state=cfg.random_state,
                    missing_rate=rate,
                    replicate=item.replicate,
                    audit=audit,
                    originally_observed_mask=obs.to_numpy(),
                    n_eligible_cells=item.n_eligible_cells,
                    n_legacy_imputed_cells_in_cohort=item.n_legacy_imputed_cells_in_cohort,
                    target_cell_policy=item.target_cell_policy,
                )
                timings[f"imputation_rate{rate}_rep{item.replicate}"] = (
                    time.perf_counter() - t0
                )
                imp_part["seed"] = item.seed
                imp_parts.append(imp_part)

                t0 = time.perf_counter()
                cls_part = run_classification_cv(
                    X_missing=item.X_missing,
                    y=cohort.y,
                    imputers=imputers,
                    classifier_name=cfg.primary_classifier,
                    n_splits=cfg.n_splits,
                    random_state=cfg.random_state,
                    missing_rate=rate,
                    replicate=item.replicate,
                    audit=audit,
                )
                timings[f"classification_rate{rate}_rep{item.replicate}"] = (
                    time.perf_counter() - t0
                )
                cls_part["seed"] = item.seed
                cls_parts.append(cls_part)

        assert_no_inplace_mutation(cohort.X, X_ref)
        imp_raw = pd.concat(imp_parts, ignore_index=True)
        cls_raw = pd.concat(cls_parts, ignore_index=True)
        assert_metrics_on_masked_only(imp_raw)
        assert_no_legacy_imputed_in_metric_target(imp_raw)
        assert_fold_isolation(audit)
        report["assertions"]["metrics_on_masked_cells_only"] = "PASS"
        report["assertions"]["no_legacy_imputed_in_metric_target"] = "PASS"
        report["assertions"]["train_val_no_overlap"] = "PASS"
        report["assertions"]["no_shared_imputer_or_scaler_or_corr_across_folds"] = "PASS"
        report["assertions"]["rfeca_training_fold_correlation_only"] = "PASS"
        report["assertions"]["loader_schema_parity_with_metabric"] = "PASS"

        imp_summary = summarize_imputation(imp_raw)
        cls_summary = summarize_classification(cls_raw)
        cls_per_class = summarize_classification_per_class(cls_raw)
        wall = time.perf_counter() - t_wall0
        timings["wall_clock_seconds"] = wall

        imp_raw.to_csv(out_dir / "exp1_imputation_raw.csv", index=False)
        imp_summary.to_csv(out_dir / "exp1_imputation_summary.csv", index=False)
        cls_raw.to_csv(out_dir / "exp2_classification_raw.csv", index=False)
        cls_summary.to_csv(out_dir / "exp2_classification_summary.csv", index=False)
        cls_per_class.to_csv(out_dir / "exp2_classification_per_class.csv", index=False)
        pd.DataFrame(audit.rows).to_csv(out_dir / "fold_audit.csv", index=False)

        config_snapshot = cfg.snapshot()
        config_snapshot["seeds"] = {
            "base_random_state": cfg.random_state,
            "cv_random_state": cfg.random_state,
            "missingness_seeds": [
                {
                    "missing_rate": r,
                    "replicate": item.replicate,
                    "seed": item.seed,
                }
                for r, reps in missing_sets.items()
                for item in reps
            ],
        }
        (out_dir / "config_snapshot.json").write_text(
            json.dumps(config_snapshot, indent=2), encoding="utf-8"
        )

        historical = _historical_reference()
        parity_rows = []
        for row in cls_summary.to_dict(orient="records"):
            name = row["imputer"]
            hist = historical.get(name)
            entry = {
                "imputer": name,
                "smoke_f1_mean": row["f1_mean"],
                "smoke_bal_mean": row["bal_mean"],
                "historical_available": bool(hist),
            }
            if hist:
                entry["historical_f1_mean"] = hist.get("f1_mean")
                entry["historical_bal_mean"] = hist.get("bal_mean")
                entry["delta_f1"] = float(row["f1_mean"] - hist["f1_mean"])
                entry["delta_bal"] = float(row["bal_mean"] - hist["bal_mean"])
                entry["deviation_cause"] = (
                    "Different CV budget (smoke 2x3 vs historical 10x5) and/or "
                    "RFECA correlation leakage fix (training-fold only)."
                )
            parity_rows.append(entry)

        # Compare to previous corrected discovery smoke (all-cells masking).
        prior_cmp = _compare_prior_smoke(imp_summary, cls_summary, missing_sets)
        report["prior_smoke_comparison"] = prior_cmp

        # Cell accounting
        cell_stats = []
        for rate, reps in missing_sets.items():
            for item in reps:
                cell_stats.append(
                    {
                        "missing_rate": rate,
                        "replicate": item.replicate,
                        "n_eligible_cells": item.n_eligible_cells,
                        "n_artificially_masked_cells": item.n_artificially_masked_cells,
                        "n_legacy_imputed_cells_in_cohort": item.n_legacy_imputed_cells_in_cohort,
                        "target_cell_policy": item.target_cell_policy,
                    }
                )

        report.update(
            {
                "status": "PASS",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(out_dir),
                "config": config_snapshot,
                "timings_seconds": timings,
                "imputation_summary": imp_summary.to_dict(orient="records"),
                "classification_summary": cls_summary.to_dict(orient="records"),
                "historical_reference": historical,
                "metric_parity": parity_rows,
                "cell_provenance_stats": cell_stats,
                "warnings": audit.warnings,
                "n_warnings": len(audit.warnings),
                "degenerate_imputation_folds": int(
                    imp_raw["degenerate_constant_preds"].sum()
                ),
                "degenerate_classification_folds": int(
                    cls_raw["degenerate_single_class_preds"].sum()
                ),
                "sample_id_status": cohort.metadata.get("sample_id_status"),
                "provenance_confidence": (
                    (cohort.metadata.get("inventory") or {})
                    .get("provenance", {})
                    .get("confidence")
                ),
            }
        )
        report["technically_ready_for_full_benchmark"] = (
            report["status"] == "PASS"
            and all(v == "PASS" for v in report["assertions"].values())
            and report["degenerate_classification_folds"] == 0
        )
        report["readiness_notes"] = [
            "Cohort key remains 'discovery' (CPTAC-derived laboratory discovery cohort).",
            "Biological Patient_IDs recovered; observation mask integrated.",
            "Primary policy: artificial missingness + RMSE/MAE on originally observed cells only.",
            "Smoke protocol matched METABRIC smoke (rate=0.10, reps=2, folds=3).",
            "Full discovery/METABRIC benchmarks not started.",
        ]

    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAIL"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["errors"].append(str(exc))
        report["traceback"] = traceback.format_exc()
        report["technically_ready_for_full_benchmark"] = False
        (out_dir / "smoke_test_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, default=str))
        raise

    md = _format_md(report)
    (out_dir / "smoke_test_report.md").write_text(md, encoding="utf-8")
    (out_dir / "smoke_test_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nArtifacts written to: {out_dir}")
    return 0


def _format_md(report: dict) -> str:
    c = report.get("cohort", {})
    lines = [
        "# Discovery smoke-test report",
        "",
        f"- **Status:** {report.get('status')}",
        f"- **Ready for full discovery benchmark:** {report.get('technically_ready_for_full_benchmark')}",
        f"- **Run ID:** {report.get('run_id')}",
        f"- **Output:** `{report.get('output_dir')}`",
        "",
        "## Cohort",
        f"- n: **{c.get('n_samples')}**",
        f"- Class distribution: `{c.get('class_distribution')}`",
        f"- Provenance confidence: **{report.get('provenance_confidence')}**",
        f"- Sample-ID status: `{report.get('sample_id_status')}`",
        "",
        "## Fingerprint",
    ]
    fp = report.get("preparation_fingerprint") or {}
    lines.append(f"- values+labels: `{fp.get('values_and_labels_sha256')}`")
    lines.append("")
    lines.append("## Assertions")
    for k, v in report.get("assertions", {}).items():
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")
    lines.append("## Imputation")
    for row in report.get("imputation_summary", []):
        cov = ""
        rv = row.get("corr_rv_mean")
        if rv is not None and rv == rv:
            cov = (
                f", corr_RV={rv:.4f}±{row.get('corr_rv_std', float('nan')):.4f}"
                f", corr_Frel={row.get('corr_frobenius_rel_mean', float('nan')):.4f}"
            )
        lines.append(
            f"- **{row['imputer']}**: RMSE={row['rmse_mean']:.4f}±{row.get('rmse_std', float('nan')):.4f}, "
            f"MAE={row['mae_mean']:.4f}±{row.get('mae_std', float('nan')):.4f}{cov}"
        )
    lines.append("")
    lines.append("## Classification (EnsembleSoft)")
    for row in report.get("classification_summary", []):
        lines.append(
            f"- **{row['imputer']}**: F1={row['f1_mean']:.4f}±{row.get('f1_std', float('nan')):.4f}, "
            f"BalAcc={row['bal_mean']:.4f}±{row.get('bal_std', float('nan')):.4f}"
        )
    lines.append("")
    lines.append("## Prior corrected smoke comparison")
    prior = report.get("prior_smoke_comparison") or {}
    if prior.get("available"):
        lines.append(
            f"- Eligible cells now: **{prior.get('current_n_eligible')}** "
            f"(prior matrix cells: {prior.get('prior_n_matrix_cells')})"
        )
        lines.append(
            f"- Masked @10% now: **{prior.get('current_n_masked')}** "
            f"(prior approx: {prior.get('prior_approx_masked_at_10pct')})"
        )
        lines.append(f"- Cause: {prior.get('expected_difference_cause')}")
        for row in prior.get("rows", []):
            if "delta_rmse" in row:
                lines.append(
                    f"- **{row['imputer']}**: RMSE {row['prior_rmse']:.4f}->{row['current_rmse']:.4f} "
                    f"(d={row['delta_rmse']:+.4f}); "
                    f"F1 {row.get('prior_f1', float('nan')):.4f}->{row.get('current_f1', float('nan')):.4f} "
                    f"(d={row.get('delta_f1', float('nan')):+.4f})"
                )
    else:
        lines.append("- Prior smoke artifact not found.")
    lines.append("")
    lines.append("## Metric parity vs historical Colab")
    for row in report.get("metric_parity", []):
        if row.get("historical_available"):
            lines.append(
                f"- **{row['imputer']}**: smoke F1={row['smoke_f1_mean']:.4f} vs hist "
                f"{row['historical_f1_mean']:.4f} (delta={row['delta_f1']:+.4f}); "
                f"BalAcc delta={row['delta_bal']:+.4f}. Cause: {row['deviation_cause']}"
            )
        else:
            lines.append(f"- **{row['imputer']}**: no historical reference row found.")
    hist = report.get("historical_reference") or {}
    for note in hist.get("protocol_notes", []):
        lines.append(f"- Note: {note}")
    lines.append("")
    lines.append("## Runtime")
    lines.append(
        f"- Wall clock: **{report.get('timings_seconds', {}).get('wall_clock_seconds', float('nan')):.2f}s**"
    )
    lines.append("")
    lines.append("**Stopped after discovery smoke test — full benchmark not started.**")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
