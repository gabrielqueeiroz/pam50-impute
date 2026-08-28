#!/usr/bin/env python3
"""
Run faithful OriginalRFECA / OriginalRFACA (and optionally legacy baselines)
into an isolated artifact namespace that never overwrites existing results.

Examples
--------
# Only new methods, METABRIC smoke-scale (fast check):
python experiments/run_original_rfeca_rfaca.py --confirm --cohort metabric --smoke

# CPTAC/discovery with LOOCV (faithful small-n protocol):
python experiments/run_original_rfeca_rfaca.py --confirm --cohort discovery \\
    --validation loocv --mechanism mcar

# Compare legacy + original (appends Original* to full six-imputer list):
python experiments/run_original_rfeca_rfaca.py --confirm --cohort metabric \\
    --include-legacy --validation kfold

Outputs land under:
  artifacts/original_rfeca_rfaca/<cohort>_<mechanism>_<tag>_<timestamp>/
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.assertions import (  # noqa: E402
    assert_artificial_mask_observed_only,
    assert_fold_isolation,
    assert_mask_feature_only,
    assert_no_inplace_mutation,
    assert_shared_mask_across_imputers,
)
from bcimpute.config import (  # noqa: E402
    ARTIFACT_ROOT,
    ORIGINAL_RFECA_RFACA_IMPUTERS,
    apply_original_rfeca_rfaca_only,
    full_benchmark_config,
    smoke_discovery_config,
    smoke_metabric_config,
    with_original_rfeca_rfaca,
)
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
from bcimpute.imputation_original import write_json  # noqa: E402
from bcimpute.imputation_original.utils import safe_mkdir  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402


NAMESPACE = ARTIFACT_ROOT / "original_rfeca_rfaca"


def _make_out_dir(cohort: str, mechanism: str, tag: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{cohort}_{mechanism}_{tag}_{run_id}"
    out = NAMESPACE / name
    return safe_mkdir(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--cohort", choices=["metabric", "discovery"], default="metabric")
    parser.add_argument("--mechanism", choices=["mcar", "mar"], default="mcar")
    parser.add_argument(
        "--validation",
        choices=["loocv", "kfold", "stratified_kfold"],
        default=None,
        help="Internal subset-selection CV for Original* (default: loocv for "
        "discovery, kfold for metabric).",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke budget (2×3).")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Keep Mean/KNN/legacy-RFECA/MissForest and append Original*.",
    )
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help="Skip exp1 imputation metrics.",
    )
    parser.add_argument(
        "--seed-scheme",
        choices=["v2", "legacy"],
        default="v2",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on ranked candidates (speeds METABRIC).",
    )
    parser.add_argument(
        "--gene-audit",
        action="store_true",
        help=(
            "Extra full-matrix fit per rate×rep×Original* to dump gene-level "
            "subset audits (expensive; off by default)."
        ),
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
        help="Override missing rates (e.g. --rates 0.2). Opt-in; default from config.",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        help="Override n_repetitions (e.g. 1 for intermediate runs).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Override outer StratifiedKFold splits (evaluation CV).",
    )
    args = parser.parse_args()

    if args.smoke:
        if args.cohort == "metabric":
            cfg = smoke_metabric_config()
            cfg.missingness_mechanism = args.mechanism
            if args.mechanism == "mar":
                cfg.tag = f"smoke_{args.mechanism}"
        else:
            cfg = smoke_discovery_config(missingness_mechanism=args.mechanism)
    else:
        cfg = full_benchmark_config(args.cohort, missingness_mechanism=args.mechanism)

    cfg.n_jobs = 1
    cfg.missingness_seed_scheme = args.seed_scheme
    validation = args.validation or (
        "loocv" if args.cohort == "discovery" else "kfold"
    )
    cfg.original_rfeca_validation = validation
    cfg.original_rfeca_n_splits = 5
    cfg.original_rfeca_max_candidates = args.max_candidates
    if args.rates is not None:
        cfg.missing_rates = [float(r) for r in args.rates]
    if args.n_repetitions is not None:
        cfg.n_repetitions = int(args.n_repetitions)
    if args.n_splits is not None:
        cfg.n_splits = int(args.n_splits)

    if args.include_legacy:
        cfg = with_original_rfeca_rfaca(cfg)
    else:
        cfg = apply_original_rfeca_rfaca_only(cfg)

    print("Configured Original RFECA/RFACA run:")
    print(json.dumps(cfg.snapshot(), indent=2))
    if not args.confirm:
        print("\nRefusing to start. Re-run with --confirm when ready.")
        return 0

    out_dir = _make_out_dir(args.cohort, args.mechanism, cfg.tag)
    print(f"Output directory: {out_dir}", flush=True)

    report: dict = {
        "run_id": out_dir.name,
        "namespace": str(NAMESPACE),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "assertions": {},
        "warnings": [],
        "errors": [],
        "output_dir": str(out_dir),
        "note": (
            "Faithful OriginalRFECA/OriginalRFACA run. "
            "Does not overwrite legacy artifact trees."
        ),
    }
    t0 = time.perf_counter()
    all_audit_rows: list[dict] = []
    gene_audits: list[dict] = []

    try:
        cohort = load_cohort(args.cohort)
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

        imputers = build_imputers(cfg)
        # Attach PAM50 strata for stratified_kfold without using it as SVR features.
        if cfg.original_rfeca_validation == "stratified_kfold":
            for name, imp in imputers.items():
                if name in ORIGINAL_RFECA_RFACA_IMPUTERS:
                    imp.strata = cohort.y.to_numpy()

        write_json(out_dir / "config_snapshot.json", cfg.snapshot())

        imp_parts: list[pd.DataFrame] = []
        cls_parts: list[pd.DataFrame] = []
        slot_i = 0
        n_slots = len(cfg.missing_rates) * cfg.n_repetitions

        for rate, reps in missing_sets.items():
            for item in reps:
                slot_i += 1
                print(
                    f"[{slot_i}/{n_slots}] rate={rate} rep={item.replicate}...",
                    flush=True,
                )
                slot_audit = FoldAudit()
                if not args.classification_only:
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
                        target_cell_policy=cfg.target_cell_policy,
                    )
                    imp_part["seed"] = item.seed
                    imp_parts.append(imp_part)

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
                cls_part["seed"] = item.seed
                cls_parts.append(cls_part)

                assert_fold_isolation(slot_audit)
                all_audit_rows.extend(slot_audit.rows)

                if args.gene_audit:
                    for name in ORIGINAL_RFECA_RFACA_IMPUTERS:
                        if name not in imputers:
                            continue
                        from sklearn.base import clone

                        audit_imp = clone(imputers[name])
                        if cfg.original_rfeca_validation == "stratified_kfold":
                            audit_imp.strata = cohort.y.to_numpy()
                        audit_imp.fit(item.X_missing)
                        gene_audits.append(
                            {
                                "missing_rate": rate,
                                "replicate": item.replicate,
                                "seed": item.seed,
                                "imputer": name,
                                "audit": audit_imp.get_audit_dict(),
                            }
                        )

        assert_no_inplace_mutation(cohort.X, X_ref)

        if imp_parts:
            imp_raw = pd.concat(imp_parts, ignore_index=True)
            imp_raw.to_csv(out_dir / "exp1_imputation_raw.csv", index=False)
            summarize_imputation(imp_raw).to_csv(
                out_dir / "exp1_imputation_summary.csv", index=False
            )

        cls_raw = pd.concat(cls_parts, ignore_index=True)
        cls_raw.to_csv(out_dir / "exp2_classification_raw.csv", index=False)
        summarize_classification(cls_raw).to_csv(
            out_dir / "exp2_classification_summary.csv", index=False
        )
        summarize_classification_per_class(cls_raw).to_csv(
            out_dir / "exp2_classification_per_class.csv", index=False
        )
        pd.DataFrame(all_audit_rows).to_csv(out_dir / "fold_audit.csv", index=False)
        write_json(out_dir / "gene_selection_audit.json", {"records": gene_audits})

        report["assertions"]["train_val_no_overlap"] = "PASS"
        report["status"] = "PASS"
        report["n_slots_done"] = slot_i
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAIL"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        import traceback

        report["errors"].append(traceback.format_exc())
        print("FAILED:", exc, flush=True)

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["timings_seconds"] = {"wall_clock_seconds": time.perf_counter() - t0}
    write_json(out_dir / "original_rfeca_rfaca_report.json", report)
    (out_dir / "original_rfeca_rfaca_report.md").write_text(
        f"# Original RFECA/RFACA report\n\n"
        f"- Status: **{report['status']}**\n"
        f"- Output: `{out_dir}`\n"
        f"- Imputers: `{cfg.imputers}`\n"
        f"- Validation: `{cfg.original_rfeca_validation}`\n",
        encoding="utf-8",
    )
    print(f"\nStatus: {report['status']}\nOutput: {out_dir}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
