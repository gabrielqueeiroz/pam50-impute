#!/usr/bin/env python3
"""
METABRIC multi-classifier interaction run (classification only).

Rates: 20% and 30% only.
Classifiers: SVC, LogReg, RF, GB, EnsembleSoft.
Skips exp1 imputation metrics (already available from six-imputer fulls).

Default seed scheme: legacy (shared masks with paper METABRIC tables).

Opt-in: requires --confirm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    assert_no_inplace_mutation,
    assert_shared_mask_across_imputers,
)
from bcimpute.config import ARTIFACT_ROOT, full_benchmark_config  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.evaluation import (  # noqa: E402
    FoldAudit,
    run_classification_cv,
    summarize_classification,
    summarize_classification_per_class,
)
from bcimpute.imputers import build_imputers  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402

CLASSIFIERS = ["SVC", "LogReg", "RF", "GB", "EnsembleSoft"]
RATES = [0.20, 0.30]


def _format_md(report: dict) -> str:
    lines = [
        "# METABRIC multi-classifier report",
        "",
        f"- **Status:** {report.get('status')}",
        f"- **Run ID:** {report.get('run_id')}",
        f"- **Output:** `{report.get('output_dir')}`",
        "",
        "## Protocol",
        f"- Rates: `{report.get('config', {}).get('missing_rates')}`",
        f"- Classifiers: `{report.get('classifiers')}`",
        f"- Imputers: `{report.get('config', {}).get('imputers')}`",
        f"- Mechanism: `{report.get('config', {}).get('missingness_mechanism')}`",
        f"- Seed scheme: `{report.get('config', {}).get('missingness_seed_scheme')}`",
        f"- Classification only (exp1 skipped)",
        "",
        "## Assertions",
    ]
    for k, v in report.get("assertions", {}).items():
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")
    lines.append("## Classification (imputer × model × rate)")
    for row in report.get("classification_summary", []):
        lines.append(
            f"- **{row['imputer']}** / {row['model']} @ {row['missing_rate']}: "
            f"F1={row['f1_mean']:.4f}±{row.get('f1_std', float('nan')):.4f}"
        )
    wall = report.get("timings_seconds", {}).get("wall_clock_seconds", float("nan"))
    lines.append("")
    lines.append("## Runtime")
    lines.append(f"- Wall clock: **{wall:.1f}s** ({wall / 3600.0:.2f} h)")
    lines.append(f"- Completed slots: {report.get('n_slots_done')} / {report.get('n_slots_total')}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--mechanism", choices=["mcar", "mar"], default="mcar")
    parser.add_argument(
        "--seed-scheme",
        choices=["v2", "legacy"],
        default="legacy",
        help="legacy aligns masks with paper METABRIC six-imputer tables.",
    )
    args = parser.parse_args()

    cfg = full_benchmark_config("metabric", missingness_mechanism=args.mechanism)
    cfg.n_jobs = 1
    cfg.missing_rates = list(RATES)
    cfg.missingness_seed_scheme = args.seed_scheme
    cfg.tag = f"multiclf_{args.mechanism}"

    n_rate_rep = len(cfg.missing_rates) * cfg.n_repetitions
    print("Configured METABRIC multi-classifier protocol:")
    print(json.dumps(cfg.snapshot(), indent=2))
    print(f"Classifiers: {CLASSIFIERS}")
    print(f"Slots (rate×rep): {n_rate_rep}")
    print(
        f"Per slot: {len(cfg.imputers)} imputers × {len(CLASSIFIERS)} classifiers "
        f"× {cfg.n_splits} folds"
    )

    if not args.confirm:
        print("\nRefusing to start. Re-run with --confirm when ready.")
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ARTIFACT_ROOT / f"metabric_multiclf_{args.mechanism}_{run_id}"
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
        "classifiers": list(CLASSIFIERS),
        "note": "classification-only; exp1 imputation metrics skipped",
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
        }
        report["config"] = cfg.snapshot()

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
        cls_parts: list[pd.DataFrame] = []
        progress_path = out_dir / "progress.jsonl"
        (out_dir / "config_snapshot.json").write_text(
            json.dumps(cfg.snapshot(), indent=2), encoding="utf-8"
        )

        rate_list = list(missing_sets.keys())
        report["n_slots_total"] = n_rate_rep
        slot_i = 0

        for rate in rate_list:
            for item in missing_sets[rate]:
                slot_i += 1
                print(
                    f"[{slot_i}/{n_rate_rep}] rate={rate} rep={item.replicate} "
                    f"({len(cfg.imputers)} imputers × {len(CLASSIFIERS)} classifiers)...",
                    flush=True,
                )
                slot_audit = FoldAudit()
                t0 = time.perf_counter()
                for clf_name in CLASSIFIERS:
                    cls_part = run_classification_cv(
                        X_missing=item.X_missing,
                        y=cohort.y,
                        imputers=imputers,
                        classifier_name=clf_name,
                        n_splits=cfg.n_splits,
                        random_state=cfg.random_state,
                        missing_rate=rate,
                        replicate=item.replicate,
                        audit=slot_audit,
                    )
                    cls_part["seed"] = item.seed
                    cls_parts.append(cls_part)
                t_cls = time.perf_counter() - t0

                assert_fold_isolation(slot_audit)
                all_audit_rows.extend(slot_audit.rows)
                all_warnings.extend(slot_audit.warnings)

                elapsed = time.perf_counter() - t_wall0
                eta = (elapsed / slot_i) * (n_rate_rep - slot_i)
                print(
                    f"    done in {t_cls:.1f}s | elapsed={elapsed / 3600:.2f}h "
                    f"ETA={eta / 3600:.2f}h",
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
                                "t_cls": t_cls,
                                "elapsed_s": elapsed,
                                "eta_s": eta,
                            }
                        )
                        + "\n"
                    )
                pd.concat(cls_parts, ignore_index=True).to_csv(
                    out_dir / "exp2_classification_raw.csv", index=False
                )

        assert_no_inplace_mutation(cohort.X, X_ref)
        cls_raw = pd.concat(cls_parts, ignore_index=True)
        cls_summary = summarize_classification(cls_raw)
        cls_per_class = summarize_classification_per_class(cls_raw)

        cls_raw.to_csv(out_dir / "exp2_classification_raw.csv", index=False)
        cls_summary.to_csv(out_dir / "exp2_classification_summary.csv", index=False)
        cls_per_class.to_csv(
            out_dir / "exp2_classification_per_class.csv", index=False
        )
        pd.DataFrame(all_audit_rows).to_csv(out_dir / "fold_audit.csv", index=False)

        report["assertions"]["train_val_no_overlap"] = "PASS"
        report["assertions"]["no_shared_imputer_or_scaler_or_corr_across_folds"] = "PASS"
        report["status"] = "PASS"
        report["n_slots_done"] = slot_i
        report["classification_summary"] = cls_summary.to_dict(orient="records")
        report["warnings"] = all_warnings
        report["n_warnings"] = len(all_warnings)

    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAIL"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        import traceback

        report["errors"].append(traceback.format_exc())
        print("FAILED:", exc, flush=True)

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["timings_seconds"] = {
        "wall_clock_seconds": time.perf_counter() - t_wall0,
    }
    (out_dir / "multiclf_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "multiclf_report.md").write_text(_format_md(report), encoding="utf-8")
    print(f"\nStatus: {report['status']}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
