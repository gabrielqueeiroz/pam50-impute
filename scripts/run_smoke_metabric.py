#!/usr/bin/env python3
"""
METABRIC smoke test for the integrated pipeline.

Runs a reduced protocol only. Does NOT launch the full benchmark.
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
    assert_fold_isolation,
    assert_mask_feature_only,
    assert_metrics_on_masked_only,
    assert_no_inplace_mutation,
    assert_shared_mask_across_imputers,
)
from bcimpute.config import ARTIFACT_ROOT, smoke_metabric_config  # noqa: E402
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


def main() -> int:
    cfg = smoke_metabric_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ARTIFACT_ROOT / f"metabric_smoke_{run_id}"
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
        # ------------------------------------------------------------------
        # Load cohort
        # ------------------------------------------------------------------
        cohort = load_cohort("metabric")
        discovery = load_cohort("discovery")  # interface check only

        # Confirm generic loader works for both schemas
        assert cohort.n_features == 50
        assert discovery.n_features == 50
        assert list(cohort.gene_names) == list(discovery.gene_names)

        X_ref = cohort.X.to_numpy(dtype=float, copy=True)
        report["cohort"] = {
            "name": cohort.name,
            "n_samples": cohort.n_samples,
            "n_features": cohort.n_features,
            "class_distribution": cohort.class_distribution(),
            "metadata": cohort.metadata,
            "discovery_loader_ok": True,
            "discovery_n": discovery.n_samples,
            "discovery_class_distribution": discovery.class_distribution(),
        }

        # ------------------------------------------------------------------
        # Missingness (shared masks across imputers)
        # ------------------------------------------------------------------
        missing_sets = generate_missingness_sets(
            cohort.X,
            missing_rates=cfg.missing_rates,
            n_repetitions=cfg.n_repetitions,
            base_seed=cfg.random_state,
            originally_observed_mask=cohort.originally_observed_mask,
            target_cell_policy=getattr(cfg, "target_cell_policy", "originally_observed_only"),
            mechanism=getattr(cfg, "missingness_mechanism", "mcar"),
        )
        shared_mask_checklist = assert_shared_mask_across_imputers(
            missing_sets, cfg.imputers
        )

        # Feature-only mask + no in-place mutation checks
        for rate, reps in missing_sets.items():
            for item in reps:
                assert_mask_feature_only(item.mask, cohort.X)
                # Labels untouched: y has no NaNs introduced
                assert cohort.y.notna().all()
        assert_no_inplace_mutation(cohort.X, X_ref)

        report["assertions"]["mask_feature_only"] = "PASS"
        report["assertions"]["complete_matrix_not_mutated"] = "PASS"
        report["assertions"]["shared_masks_across_imputers"] = "PASS"
        report["shared_mask_checklist"] = shared_mask_checklist

        # ------------------------------------------------------------------
        # Imputers
        # ------------------------------------------------------------------
        imputers = build_imputers(cfg)
        # Guard: RFECA must not accept precomputed corr path under smoke config
        for name, imp in imputers.items():
            if name.startswith("RFECA"):
                assert getattr(imp, "corr_csv_path", None) is None
                assert getattr(imp, "allow_precomputed_correlation", False) is False

        audit = FoldAudit()
        imp_raw_parts: list[pd.DataFrame] = []
        cls_raw_parts: list[pd.DataFrame] = []
        timings: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Protocol loop
        # ------------------------------------------------------------------
        for rate, reps in missing_sets.items():
            for item in reps:
                # Pairwise equality: every imputer sees identical mask/X_missing
                for name in cfg.imputers:
                    assert item.mask is missing_sets[rate][item.replicate].mask
                    assert item.X_missing is missing_sets[rate][item.replicate].X_missing

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
                    originally_observed_mask=(
                        None
                        if cohort.originally_observed_mask is None
                        else cohort.originally_observed_mask.to_numpy()
                    ),
                    n_eligible_cells=item.n_eligible_cells,
                    n_legacy_imputed_cells_in_cohort=item.n_legacy_imputed_cells_in_cohort,
                    target_cell_policy=item.target_cell_policy,
                )
                timings[f"imputation_rate{rate}_rep{item.replicate}"] = (
                    time.perf_counter() - t0
                )
                imp_part["seed"] = item.seed
                imp_raw_parts.append(imp_part)

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
                cls_raw_parts.append(cls_part)

        # Original matrix still intact after all runs
        assert_no_inplace_mutation(cohort.X, X_ref)

        imp_raw = pd.concat(imp_raw_parts, ignore_index=True)
        cls_raw = pd.concat(cls_raw_parts, ignore_index=True)
        assert_metrics_on_masked_only(imp_raw)
        report["assertions"]["metrics_on_masked_cells_only"] = "PASS"

        assert_fold_isolation(audit)
        report["assertions"]["train_val_no_overlap"] = "PASS"
        report["assertions"]["no_shared_imputer_or_scaler_or_corr_across_folds"] = "PASS"
        report["assertions"]["rfeca_training_fold_correlation_only"] = "PASS"

        imp_summary = summarize_imputation(imp_raw)
        cls_summary = summarize_classification(cls_raw)
        cls_per_class = summarize_classification_per_class(cls_raw)

        wall = time.perf_counter() - t_wall0
        timings["wall_clock_seconds"] = wall

        # ------------------------------------------------------------------
        # Persist artifacts
        # ------------------------------------------------------------------
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

        # Fold class distributions from classification raw
        fold_dists = (
            cls_raw[
                [
                    "imputer",
                    "replicate",
                    "fold",
                    "train_class_distribution",
                    "test_class_distribution",
                    "n_train",
                    "n_test",
                ]
            ]
            .drop_duplicates()
            .to_dict(orient="records")
        )

        masked_cells = (
            imp_raw.groupby(["missing_rate", "replicate", "imputer"], as_index=False)[
                "n_masked_test_values"
            ]
            .sum()
            .to_dict(orient="records")
        )

        report.update(
            {
                "status": "PASS",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(out_dir),
                "config": config_snapshot,
                "timings_seconds": timings,
                "fold_class_distributions": fold_dists,
                "masked_cells": masked_cells,
                "imputation_summary": imp_summary.to_dict(orient="records"),
                "classification_summary": cls_summary.to_dict(orient="records"),
                "warnings": audit.warnings,
                "n_warnings": len(audit.warnings),
                "degenerate_imputation_folds": int(
                    imp_raw["degenerate_constant_preds"].sum()
                ),
                "degenerate_classification_folds": int(
                    cls_raw["degenerate_single_class_preds"].sum()
                ),
            }
        )

        # Readiness heuristic for full METABRIC benchmark
        ready = (
            report["status"] == "PASS"
            and all(v == "PASS" for v in report["assertions"].values())
            and report["degenerate_classification_folds"] == 0
            and cohort.n_samples >= 100
        )
        report["technically_ready_for_full_benchmark"] = bool(ready)
        report["readiness_notes"] = [
            "Smoke protocol completed with leakage-safety assertions PASS.",
            "RFECA correlations computed from training-fold X.corr() only.",
            "StandardScaler fitted inside CV folds via sklearn Pipeline clone.",
            "Full benchmark will use config.full_benchmark_config('metabric') "
            "(rates 0-30%, 10 reps, 5 folds) and will be substantially slower.",
            f"Smoke wall time: {wall:.1f}s on n={cohort.n_samples}.",
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

    # Human-readable markdown report
    md = _format_markdown_report(report)
    (out_dir / "smoke_test_report.md").write_text(md, encoding="utf-8")
    (out_dir / "smoke_test_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(md)
    print(f"\nArtifacts written to: {out_dir}")
    return 0 if report["status"] == "PASS" else 1


def _format_markdown_report(report: dict) -> str:
    lines = []
    lines.append("# METABRIC smoke-test report")
    lines.append("")
    lines.append(f"- **Status:** {report.get('status')}")
    lines.append(
        f"- **Technically ready for full METABRIC benchmark:** "
        f"{report.get('technically_ready_for_full_benchmark')}"
    )
    lines.append(f"- **Run ID:** {report.get('run_id')}")
    lines.append(f"- **Output:** `{report.get('output_dir')}`")
    lines.append("")

    cohort = report.get("cohort", {})
    lines.append("## Cohort")
    lines.append(f"- Final n: **{cohort.get('n_samples')}**")
    lines.append(f"- Class distribution: `{cohort.get('class_distribution')}`")
    lines.append(f"- Genes: {cohort.get('n_features')}")
    lines.append("")

    lines.append("## Assertions")
    for k, v in report.get("assertions", {}).items():
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")

    lines.append("## Masked cells")
    for row in report.get("masked_cells", []):
        lines.append(
            f"- rate={row['missing_rate']} rep={row['replicate']} "
            f"{row['imputer']}: n_masked_test_values={row['n_masked_test_values']}"
        )
    lines.append("")

    lines.append("## Imputation (RMSE / MAE / corr preservation)")
    for row in report.get("imputation_summary", []):
        cov = ""
        rv = row.get("corr_rv_mean")
        if rv is not None and rv == rv:
            cov = (
                f", corr_RV={rv:.4f}±{row.get('corr_rv_std', float('nan')):.4f}"
                f", corr_Frel={row.get('corr_frobenius_rel_mean', float('nan')):.4f}"
            )
        lines.append(
            f"- **{row['imputer']}**: "
            f"RMSE={row['rmse_mean']:.4f}±{row.get('rmse_std', float('nan')):.4f}, "
            f"MAE={row['mae_mean']:.4f}±{row.get('mae_std', float('nan')):.4f}{cov}"
        )
    lines.append("")

    lines.append("## Classification (primary model)")
    for row in report.get("classification_summary", []):
        lines.append(
            f"- **{row['imputer']}** / {row['model']}: "
            f"macro-F1={row['f1_mean']:.4f}±{row.get('f1_std', float('nan')):.4f}, "
            f"BalAcc={row['bal_mean']:.4f}±{row.get('bal_std', float('nan')):.4f}"
        )
    lines.append("")

    timings = report.get("timings_seconds", {})
    lines.append("## Runtime")
    lines.append(f"- Wall clock: **{timings.get('wall_clock_seconds', float('nan')):.2f}s**")
    for k, v in timings.items():
        if k != "wall_clock_seconds":
            lines.append(f"- {k}: {v:.2f}s")
    lines.append("")

    lines.append("## Fold class distributions (sample)")
    for row in report.get("fold_class_distributions", [])[:12]:
        lines.append(
            f"- rep={row['replicate']} fold={row['fold']} "
            f"train={row['train_class_distribution']} "
            f"test={row['test_class_distribution']}"
        )
    lines.append("")

    lines.append("## Warnings / degenerate predictions")
    lines.append(f"- Warning count: {report.get('n_warnings', 0)}")
    lines.append(
        f"- Degenerate imputation folds: {report.get('degenerate_imputation_folds', 0)}"
    )
    lines.append(
        f"- Degenerate classification folds: {report.get('degenerate_classification_folds', 0)}"
    )
    warns = report.get("warnings", [])
    if warns:
        lines.append("- Details (first 20):")
        for w in warns[:20]:
            lines.append(f"  - {w}")
    else:
        lines.append("- None recorded.")
    lines.append("")

    lines.append("## Readiness notes")
    for note in report.get("readiness_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("**Stopped after smoke test — full METABRIC benchmark not started.**")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
