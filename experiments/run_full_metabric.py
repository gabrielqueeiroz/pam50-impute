#!/usr/bin/env python3
"""
Full METABRIC benchmark (sequential).

Opt-in: refuses to start unless --confirm is passed.
Protocol: full_benchmark_config("metabric") — 5 rates × 10 reps × 5 folds × 6 imputers
(incl. MissForest-like IterativeImputer + ExtraTrees).

Expected wall time: longer than the prior ~3.2 h METABRIC full (MissForest is slower
than KNN/RFECA). Prefer Discovery full smoke->full before METABRIC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXP = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from bcimpute.assertions import (  # noqa: E402
    assert_artificial_mask_observed_only,
    assert_fold_isolation,
    assert_mask_feature_only,
    assert_metrics_on_masked_only,
    assert_no_inplace_mutation,
    assert_no_legacy_imputed_in_metric_target,
    assert_shared_mask_across_imputers,
)
from bcimpute.config import ARTIFACT_ROOT, full_benchmark_config  # noqa: E402
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

from cli_common import (  # noqa: E402
    add_full_run_arguments,
    artifact_prefix,
    configure_full_run,
)


def _format_md(report: dict) -> str:
    c = report.get("cohort", {})
    lines = [
        "# METABRIC full-benchmark report",
        "",
        f"- **Status:** {report.get('status')}",
        f"- **Run ID:** {report.get('run_id')}",
        f"- **Output:** `{report.get('output_dir')}`",
        "",
        "## Cohort",
        f"- n: **{c.get('n_samples')}**",
        f"- Class distribution: `{c.get('class_distribution')}`",
        "",
        "## Protocol",
        f"- Rates: `{report.get('config', {}).get('missing_rates')}`",
        f"- Reps × folds: {report.get('config', {}).get('n_repetitions')} × "
        f"{report.get('config', {}).get('n_splits')}",
        f"- Imputers: `{report.get('config', {}).get('imputers')}`",
        f"- Target cell policy: `{report.get('config', {}).get('target_cell_policy')}`",
        f"- Execution: **sequential** (n_jobs forced to 1)",
        "",
        "## Assertions",
    ]
    for k, v in report.get("assertions", {}).items():
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")
    lines.append("## Imputation (by imputer × rate)")
    for row in report.get("imputation_summary", []):
        cov = ""
        if "corr_rv_mean" in row and row.get("corr_rv_mean") == row.get("corr_rv_mean"):
            cov = (
                f", corr_RV={row['corr_rv_mean']:.4f}±{row.get('corr_rv_std', float('nan')):.4f}"
                f", corr_Frel={row.get('corr_frobenius_rel_mean', float('nan')):.4f}"
            )
        lines.append(
            f"- **{row['imputer']}** @ {row['missing_rate']}: "
            f"RMSE={row['rmse_mean']:.4f}±{row.get('rmse_std', float('nan')):.4f}, "
            f"MAE={row['mae_mean']:.4f}±{row.get('mae_std', float('nan')):.4f}{cov}"
        )
    lines.append("")
    lines.append("## Classification EnsembleSoft (by imputer × rate)")
    for row in report.get("classification_summary", []):
        lines.append(
            f"- **{row['imputer']}** @ {row['missing_rate']}: "
            f"F1={row['f1_mean']:.4f}±{row.get('f1_std', float('nan')):.4f}, "
            f"BalAcc={row['bal_mean']:.4f}±{row.get('bal_std', float('nan')):.4f}"
        )
    lines.append("")
    wall = report.get("timings_seconds", {}).get("wall_clock_seconds", float("nan"))
    lines.append("## Runtime")
    lines.append(f"- Wall clock: **{wall:.1f}s** ({wall / 3600.0:.2f} h)")
    lines.append(f"- Completed slots: {report.get('n_slots_done')} / {report.get('n_slots_total')}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_full_run_arguments(parser)
    args = parser.parse_args()

    cfg = full_benchmark_config("metabric", missingness_mechanism=args.mechanism)
    configure_full_run(cfg, args)

    n_slots = (
        len(cfg.missing_rates) * cfg.n_repetitions * cfg.n_splits * len(cfg.imputers)
    )
    print("Configured full METABRIC protocol:")
    print(json.dumps(cfg.snapshot(), indent=2))
    print(f"Approximate fold×imputer slots: {n_slots}")
    print(f"Missingness mechanism: {cfg.missingness_mechanism}")
    print(f"Missingness seed scheme: {cfg.missingness_seed_scheme}")
    if args.only_rfeca and args.seed_scheme == "v2":
        print(
            "NOTE: --only-rfeca + seed scheme v2 -> masks differ from 2026-07 full "
            "artifacts; do not merge with Mean/KNN/MissForest from those runs."
        )

    if not args.confirm:
        print(
            "\nRefusing to start. Re-run with --confirm when ready "
            "(expected wall time: hours; less with --only-rfeca)."
        )
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = artifact_prefix("metabric", args.mechanism, only_rfeca=args.only_rfeca)
    out_dir = ARTIFACT_ROOT / f"{prefix}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}", flush=True)

    report: dict = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "assertions": {},
        "warnings": [],
        "errors": [],
        "output_dir": str(out_dir),
    }
    t_wall0 = time.perf_counter()
    all_audit_rows: list[dict] = []
    all_warnings: list[str] = []

    try:
        cohort = load_cohort("metabric")
        assert cohort.originally_observed_mask is not None
        obs = cohort.originally_observed_mask
        X_ref = cohort.X.to_numpy(dtype=float, copy=True)

        report["cohort"] = {
            "name": cohort.name,
            "n_samples": cohort.n_samples,
            "n_features": cohort.n_features,
            "class_distribution": cohort.class_distribution(),
            "n_originally_observed_cells": int(obs.to_numpy().sum()),
            "n_legacy_imputed_cells": int((~obs.to_numpy()).sum()),
            "metadata": cohort.metadata,
        }

        missing_sets = generate_missingness_sets(
            cohort.X,
            missing_rates=cfg.missing_rates,
            n_repetitions=cfg.n_repetitions,
            base_seed=cfg.random_state,
            originally_observed_mask=obs,
            target_cell_policy=cfg.target_cell_policy,
            mechanism=cfg.missingness_mechanism,
            seed_scheme=cfg.missingness_seed_scheme,
        )
        assert_shared_mask_across_imputers(missing_sets, cfg.imputers)
        for reps in missing_sets.values():
            for item in reps:
                assert_mask_feature_only(item.mask, cohort.X)
                assert_artificial_mask_observed_only(item.mask, obs.to_numpy())
        assert_no_inplace_mutation(cohort.X, X_ref)
        report["assertions"]["mask_feature_only"] = "PASS"
        report["assertions"]["complete_matrix_not_mutated"] = "PASS"
        report["assertions"]["shared_masks_across_imputers"] = "PASS"
        report["assertions"]["artificial_mask_observed_only"] = "PASS"

        imputers = build_imputers(cfg)
        imp_parts: list[pd.DataFrame] = []
        cls_parts: list[pd.DataFrame] = []
        timings: dict[str, float] = {}

        rate_list = list(missing_sets.keys())
        n_rate_rep = sum(len(v) for v in missing_sets.values())
        slot_i = 0
        report["n_slots_total"] = n_rate_rep

        progress_path = out_dir / "progress.jsonl"

        for rate in rate_list:
            for item in missing_sets[rate]:
                slot_i += 1
                print(
                    f"[{slot_i}/{n_rate_rep}] rate={rate} rep={item.replicate} "
                    f"({len(cfg.imputers)} imputers × {cfg.n_splits} folds)...",
                    flush=True,
                )

                # Per-slot audit avoids retaining thousands of fitted EnsembleSoft
                # models in memory across a multi-hour METABRIC run.
                slot_audit = FoldAudit()

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
                    audit=slot_audit,
                    originally_observed_mask=obs.to_numpy(),
                    n_eligible_cells=item.n_eligible_cells,
                    n_legacy_imputed_cells_in_cohort=item.n_legacy_imputed_cells_in_cohort,
                    target_cell_policy=item.target_cell_policy,
                )
                t_imp = time.perf_counter() - t0
                timings[f"imputation_rate{rate}_rep{item.replicate}"] = t_imp
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
                    audit=slot_audit,
                )
                t_cls = time.perf_counter() - t0
                timings[f"classification_rate{rate}_rep{item.replicate}"] = t_cls
                cls_part["seed"] = item.seed
                cls_parts.append(cls_part)

                assert_fold_isolation(slot_audit)
                all_audit_rows.extend(slot_audit.rows)
                all_warnings.extend(slot_audit.warnings)

                elapsed = time.perf_counter() - t_wall0
                eta = (elapsed / slot_i) * (n_rate_rep - slot_i)
                print(
                    f"    done in {t_imp + t_cls:.1f}s "
                    f"(imp={t_imp:.1f}s, cls={t_cls:.1f}s) | "
                    f"elapsed={elapsed / 3600:.2f}h ETA={eta / 3600:.2f}h",
                    flush=True,
                )

                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "slot": slot_i,
                                "n_total": n_rate_rep,
                                "missing_rate": rate,
                                "replicate": item.replicate,
                                "t_imp": t_imp,
                                "t_cls": t_cls,
                                "elapsed_s": elapsed,
                                "eta_s": eta,
                            }
                        )
                        + "\n"
                    )

                pd.concat(imp_parts, ignore_index=True).to_csv(
                    out_dir / "exp1_imputation_raw.csv", index=False
                )
                pd.concat(cls_parts, ignore_index=True).to_csv(
                    out_dir / "exp2_classification_raw.csv", index=False
                )

        assert_no_inplace_mutation(cohort.X, X_ref)
        imp_raw = pd.concat(imp_parts, ignore_index=True)
        cls_raw = pd.concat(cls_parts, ignore_index=True)
        assert_metrics_on_masked_only(imp_raw)
        assert_no_legacy_imputed_in_metric_target(imp_raw)

        report["assertions"]["metrics_on_masked_cells_only"] = "PASS"
        report["assertions"]["no_legacy_imputed_in_metric_target"] = "PASS"
        report["assertions"]["train_val_no_overlap"] = "PASS"
        report["assertions"]["no_shared_imputer_or_scaler_or_corr_across_folds"] = "PASS"
        report["assertions"]["rfeca_training_fold_correlation_only"] = "PASS"

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
        pd.DataFrame(all_audit_rows).to_csv(out_dir / "fold_audit.csv", index=False)

        config_snapshot = cfg.snapshot()
        config_snapshot["seeds"] = {
            "base_random_state": cfg.random_state,
            "cv_random_state": cfg.random_state,
            "missingness_seed_scheme": cfg.missingness_seed_scheme,
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

        report.update(
            {
                "status": "PASS",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config_snapshot,
                "timings_seconds": timings,
                "n_slots_done": slot_i,
                "imputation_summary": imp_summary.to_dict(orient="records"),
                "classification_summary": cls_summary.to_dict(orient="records"),
                "warnings": all_warnings,
                "n_warnings": len(all_warnings),
                "degenerate_imputation_folds": int(
                    imp_raw["degenerate_constant_preds"].sum()
                ),
                "degenerate_classification_folds": int(
                    cls_raw["degenerate_single_class_preds"].sum()
                ),
            }
        )

    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAIL"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["errors"].append(str(exc))
        report["traceback"] = traceback.format_exc()
        report["timings_seconds"] = {
            "wall_clock_seconds": time.perf_counter() - t_wall0
        }
        (out_dir / "full_benchmark_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, default=str))
        raise

    md = _format_md(report)
    (out_dir / "full_benchmark_report.md").write_text(md, encoding="utf-8")
    (out_dir / "full_benchmark_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nArtifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
