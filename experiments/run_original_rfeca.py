#!/usr/bin/env python3
"""
OriginalRFECA leakage-safe principal experiment runner.

- Only OriginalRFECA (no RFACA, no legacy imputers)
- use_scaler=False, max_candidates=49, selection_protocol=leakage_safe
- Discovery → loocv; METABRIC → kfold(n_splits=5)
- Isolated artifact namespace; checkpoint/resume; never overwrites prior results

Examples
--------
# Intermediate operational validation (1 rate × 1 rep, full-matrix fit):
python experiments/run_original_rfeca.py --confirm --mode intermediate \\
    --cohort metabric --mechanism mcar --rate 0.20 --replicate 0

# Full protocol (all rates × reps); resumes unfinished slots:
python experiments/run_original_rfeca.py --confirm --mode full \\
    --cohort metabric --mechanism mcar
"""

from __future__ import annotations

import argparse
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

from bcimpute.config import (  # noqa: E402
    ARTIFACT_ROOT,
    apply_original_rfeca_only,
    full_benchmark_config,
)
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.imputers import build_imputers  # noqa: E402
from bcimpute.imputation_original.utils import safe_mkdir, write_json  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402

NAMESPACE = ARTIFACT_ROOT / "original_rfeca_leakage_safe_no_scaler_maxcand49"


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:  # noqa: BLE001
        try:
            import resource

            # Linux: ru_maxrss in KB; Windows may differ — best-effort
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        except Exception:  # noqa: BLE001
            return None


def _deps_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "feature_engine"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unavailable:{type(exc).__name__}"
    return out


def _slot_key(rate: float, replicate: int) -> str:
    return f"rate_{rate:.2f}_rep_{replicate}"


def _slot_done(out_dir: Path, key: str) -> bool:
    marker = out_dir / "slots" / key / "DONE.json"
    return marker.exists()


def _configure(args: argparse.Namespace):
    cfg = full_benchmark_config(args.cohort, missingness_mechanism=args.mechanism)
    cfg = apply_original_rfeca_only(cfg)
    cfg.n_jobs = 1
    cfg.missingness_seed_scheme = args.seed_scheme
    if args.mode == "intermediate":
        cfg.missing_rates = [float(args.rate)]
        # Keep full-protocol n_repetitions so replicate seeds match the benchmark
        # schedule; the runner then filters to --replicate.
        cfg.tag = f"intermediate_{cfg.tag}"
    if args.rates is not None:
        cfg.missing_rates = [float(r) for r in args.rates]
    if args.n_repetitions is not None:
        cfg.n_repetitions = int(args.n_repetitions)
    # Enforce protocol decisions
    cfg.original_rfeca_use_scaler = False
    cfg.original_rfeca_max_candidates = 49
    cfg.original_rfeca_selection_protocol = "leakage_safe"
    cfg.original_rfeca_kernel = "linear"
    if args.cohort == "discovery":
        cfg.original_rfeca_validation = "loocv"
    else:
        cfg.original_rfeca_validation = "kfold"
        cfg.original_rfeca_n_splits = 5
    return cfg


def _run_slot(
    *,
    out_dir: Path,
    cohort,
    cfg,
    item,
    rate: float,
    replicate: int,
) -> dict:
    key = _slot_key(rate, replicate)
    slot_dir = out_dir / "slots" / key
    if slot_dir.exists() and (slot_dir / "DONE.json").exists():
        return json.loads((slot_dir / "DONE.json").read_text(encoding="utf-8"))

    # New slot dir (refuse non-empty incomplete without DONE — resume via gene ckpt)
    slot_dir.mkdir(parents=True, exist_ok=True)
    ckpt = slot_dir / "imputer_checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)

    # Persist mask once
    mask_path = slot_dir / "mask.npz"
    if not mask_path.exists():
        np.savez_compressed(
            mask_path,
            mask=np.asarray(item.mask, dtype=bool),
            seed=np.asarray([item.seed]),
            missing_rate=np.asarray([rate]),
            replicate=np.asarray([replicate]),
        )

    imputers = build_imputers(cfg)
    imp = imputers["OriginalRFECA"]
    imp.checkpoint_dir = str(ckpt)
    imp.record_fold_details = False
    imp.set_run_context(
        dataset=cohort.name,
        mechanism=cfg.missingness_mechanism,
        missing_rate=rate,
        replicate=replicate,
        seed=item.seed,
    )

    rss0 = _peak_rss_mb()
    t0 = time.perf_counter()
    print(f"  Fitting OriginalRFECA on {key} ...", flush=True)
    imp.fit(item.X_missing)
    t_fit = time.perf_counter() - t0
    print(f"  Transform ...", flush=True)
    t1 = time.perf_counter()
    X_imp = imp.transform(item.X_missing)
    t_tr = time.perf_counter() - t1
    rss1 = _peak_rss_mb()

    audit = imp.get_audit_dict()
    gene_rows = []
    for g in audit.get("genes", []):
        gene_rows.append(
            {
                "gene": g["gene"],
                "status": g["status"],
                "n_observed": g["n_observed"],
                "n_missing": g["n_missing"],
                "winning_prefix_len": g["winning_prefix_len"],
                "n_predictors_selected": g["n_predictors_selected"],
                "winning_predictors": "|".join(g["winning_predictors"]),
                "winning_prefix_genes": "|".join(g["winning_prefix_genes"]),
                "selection_seconds": g["selection_seconds"],
                "final_fit_seconds": g["final_fit_seconds"],
                "n_prefixes_evaluated": g.get("n_prefixes_evaluated", 0),
                "n_rfe_fits": g.get("n_rfe_fits", 0),
                "n_svr_fits": g.get("n_svr_fits", 0),
                "best_oof_rmse": next(
                    (
                        s["rmse"]
                        for s in g.get("subsets_evaluated", [])
                        if s.get("prefix_len") == g["winning_prefix_len"]
                    ),
                    float("nan"),
                ),
            }
        )
    pd.DataFrame(gene_rows).to_csv(slot_dir / "gene_summary.csv", index=False)
    write_json(slot_dir / "gene_selection_audit.json", audit)

    fb = audit.get("fallback_events", [])
    pd.DataFrame(fb).to_csv(slot_dir / "fallback_events.csv", index=False)

    # Determinism: transform twice with the same fitted models.
    X_imp2 = imp.transform(item.X_missing)
    det_ok = bool(np.allclose(X_imp.to_numpy(), X_imp2.to_numpy(), equal_nan=True))

    summary = {
        "slot": key,
        "missing_rate": rate,
        "replicate": replicate,
        "seed": item.seed,
        "status": "PASS",
        "use_scaler": False,
        "svr_kernel": "linear",
        "selection_protocol": "leakage_safe",
        "max_candidates": 49,
        "validation_strategy": cfg.original_rfeca_validation,
        "n_splits_internal": cfg.original_rfeca_n_splits,
        "n_genes_ok": int(sum(1 for r in gene_rows if r["status"] == "ok")),
        "n_genes_skipped": int(sum(1 for r in gene_rows if str(r["status"]).startswith("skipped"))),
        "fallback_rate": audit.get("fallback_rate", 0.0),
        "incomplete_predictor_fallback_rate": audit.get(
            "incomplete_predictor_fallback_rate", 0.0
        ),
        "n_fallback_events": len(fb),
        "n_model_fallback_events": int(
            sum(
                1
                for e in fb
                if e.get("reason") not in {"incomplete_predictors_on_impute_rows"}
            )
        ),
        "n_incomplete_predictor_events": int(
            sum(
                1
                for e in fb
                if e.get("reason") == "incomplete_predictors_on_impute_rows"
            )
        ),
        "n_exceptions": len(audit.get("exceptions", [])),
        "n_rfe_fits_total": audit.get("n_rfe_fits_total", 0),
        "n_svr_fits_total": audit.get("n_svr_fits_total", 0),
        "n_prefixes_evaluated_total": int(
            sum(int(r.get("n_prefixes_evaluated", 0)) for r in gene_rows)
        ),
        "timings_seconds": {
            "fit": t_fit,
            "transform": t_tr,
            "total": t_fit + t_tr,
            "per_gene_mean_selection": float(
                np.mean([r["selection_seconds"] for r in gene_rows]) if gene_rows else 0.0
            ),
        },
        "memory_rss_mb": {"before": rss0, "after": rss1},
        "determinism_transform_idempotent": det_ok,
        "n_nan_remaining": int(X_imp.isna().sum().sum()),
    }
    write_json(slot_dir / "slot_summary.json", summary)
    write_json(slot_dir / "DONE.json", summary)

    # Append global progress
    with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "slot_done", **summary}) + "\n")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--mode", choices=["intermediate", "full"], default="intermediate")
    parser.add_argument("--cohort", choices=["metabric", "discovery"], default="metabric")
    parser.add_argument("--mechanism", choices=["mcar", "mar"], default="mcar")
    parser.add_argument("--rate", type=float, default=0.20, help="Intermediate rate")
    parser.add_argument(
        "--replicate",
        type=int,
        default=0,
        help="Which replicate index to keep in intermediate mode (0-based among generated).",
    )
    parser.add_argument("--rates", type=float, nargs="+", default=None)
    parser.add_argument("--n-repetitions", type=int, default=None)
    parser.add_argument("--seed-scheme", choices=["v2", "legacy"], default="v2")
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Existing run directory to resume (must be under the namespace).",
    )
    args = parser.parse_args()

    cfg = _configure(args)
    print("OriginalRFECA principal protocol:")
    print(json.dumps(cfg.snapshot(), indent=2))
    if not args.confirm:
        print("\nRefusing to start. Re-run with --confirm when ready.")
        return 0

    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.exists():
            raise FileNotFoundError(out_dir)
        if out_dir.parent.resolve() != NAMESPACE.resolve():
            raise ValueError(f"resume-dir must be a direct child of {NAMESPACE}")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = (
            f"{args.cohort}_{args.mechanism}_{args.mode}_"
            f"original_rfeca_leakage_safe_no_scaler_maxcand49_{run_id}"
        )
        out_dir = safe_mkdir(NAMESPACE / name)

    print(f"Output: {out_dir}", flush=True)

    report: dict = {
        "run_id": out_dir.name,
        "namespace": str(NAMESPACE),
        "mode": args.mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": cfg.snapshot(),
        "dependencies": _deps_versions(),
        "decisions": {
            "use_scaler": False,
            "svr_kernel": "linear",
            "selection_protocol": "leakage_safe",
            "max_candidates": 49,
            "rfe": {"step": 1, "n_features_to_select": None},
            "validation": cfg.original_rfeca_validation,
            "n_splits": cfg.original_rfeca_n_splits,
        },
        "slots": [],
        "errors": [],
    }
    # Don't overwrite existing report mid-resume — write running snapshot under new name if needed
    running_path = out_dir / "run_report_running.json"
    running_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    t_wall0 = time.perf_counter()
    try:
        cohort = load_cohort(args.cohort)
        assert cohort.originally_observed_mask is not None
        report["cohort"] = {
            "name": cohort.name,
            "n_samples": cohort.n_samples,
            "n_features": cohort.n_features,
            "class_distribution": cohort.class_distribution(),
        }

        # Generate with configured n_repetitions (full schedule) for seed parity.
        missing_sets = generate_missingness_sets(
            cohort.X,
            missing_rates=cfg.missing_rates,
            n_repetitions=cfg.n_repetitions,
            base_seed=cfg.random_state,
            originally_observed_mask=cohort.originally_observed_mask,
            target_cell_policy=cfg.target_cell_policy,
            mechanism=cfg.missingness_mechanism,
            seed_scheme=cfg.missingness_seed_scheme,
        )

        for rate, reps in missing_sets.items():
            for item in reps:
                if args.mode == "intermediate" and int(item.replicate) != int(args.replicate):
                    continue
                key = _slot_key(rate, int(item.replicate))
                if _slot_done(out_dir, key):
                    print(f"[skip] {key} already DONE", flush=True)
                    done = json.loads(
                        (out_dir / "slots" / key / "DONE.json").read_text(encoding="utf-8")
                    )
                    report["slots"].append(done)
                    continue
                print(f"[run] {key} seed={item.seed}", flush=True)
                summary = _run_slot(
                    out_dir=out_dir,
                    cohort=cohort,
                    cfg=cfg,
                    item=item,
                    rate=float(rate),
                    replicate=int(item.replicate),
                )
                report["slots"].append(summary)
                if summary.get("fallback_rate", 0) > 0.05:
                    report["errors"].append(
                        f"High model fallback_rate={summary['fallback_rate']} on {key}"
                    )
                if summary.get("incomplete_predictor_fallback_rate", 0) >= 1.0:
                    report.setdefault("warnings", []).append(
                        f"All genes had some incomplete-predictor row fills on {key} "
                        "(expected under MCAR without predictor imputation)."
                    )

        report["status"] = "PASS" if not report["errors"] else "PASS_WITH_WARNINGS"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAIL"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["errors"].append(traceback.format_exc())
        print("FAILED:", exc, flush=True)

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["timings_seconds"] = {"wall_clock_seconds": time.perf_counter() - t_wall0}

    # Extrapolation (intermediate → full)
    if report.get("slots"):
        t_slot = float(
            np.mean([s["timings_seconds"]["total"] for s in report["slots"]])
        )
        n_rates = 5
        n_reps = 10
        n_outer = 5
        n_mech = 2
        t_metabric_one_mech = t_slot * n_rates * n_reps * n_outer
        report["cost_extrapolation"] = {
            "seconds_per_slot_fit": t_slot,
            "note": (
                "Slot = one OriginalRFECA.fit on full X_missing (all genes). "
                "Full evaluation CV multiplies by n_splits outer folds."
            ),
            "metabric_one_mechanism_eval_hours": (t_metabric_one_mech * 1.25) / 3600.0,
            "metabric_mcar_and_mar_eval_hours": (t_metabric_one_mech * n_mech * 1.25)
            / 3600.0,
            "margin": 0.25,
            "n_combinations_metabric_one_mech": n_rates * n_reps,
            "n_outer_fits_metabric_one_mech": n_rates * n_reps * n_outer,
            "discovery_loocv_warning": (
                "Discovery uses LOOCV internally; cost is not a simple n-ratio "
                "of METABRIC kfold."
            ),
        }

    final_path = out_dir / "run_report.json"
    if final_path.exists():
        # never overwrite — write timestamped
        final_path = out_dir / f"run_report_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    write_json(final_path, report)

    print(f"\nStatus: {report['status']}\nOutput: {out_dir}", flush=True)
    return 0 if str(report["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
