#!/usr/bin/env python3
"""
PAM50 classification on OriginalRFECA TARGET-WISE imputed matrices.

Protocol (post-imputation, not nested):
  1. Load DONE slot mask + gene_models
  2. Reconstruct complete X_imp via set_target_wise_context + transform
  3. StratifiedKFold(5) with identity imputer + scaler/classifier pipelines

Classifiers: EnsembleSoft (primary) + SVC, LogReg, RF, GB.

Rates: 0.05 / 0.10 / 0.20 / 0.30 × MCAR+MAR × reps 0–4 (skip missing DONE).

Usage (repo root):
  python experiments/run_original_rfeca_classification.py --confirm
  python experiments/run_original_rfeca_classification.py --confirm --rates 0.10 0.20 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.config import PAM50_GENES  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.evaluation import (  # noqa: E402
    FoldAudit,
    run_classification_cv,
    summarize_classification,
    summarize_classification_per_class,
)
from bcimpute.imputation_original.rfeca import OriginalRFECAImputer  # noqa: E402
from bcimpute.imputation_original.types import FitAudit  # noqa: E402

ART = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
OUT = ART / "classification"
CLASSIFIERS = ["EnsembleSoft", "SVC", "LogReg", "RF", "GB"]
DEFAULT_RATES = [0.05, 0.10, 0.20, 0.30]
MECHANISMS = ("mcar", "mar")
N_SPLITS = 5
RANDOM_STATE = 42


class IdentityImputer(BaseEstimator, TransformerMixin):
    """Passthrough: matrix is already imputed (TARGET-WISE freeze)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def _slot_dir(mechanism: str, rate: float, rep: int) -> Path:
    return ART / mechanism / f"rate_{rate:.2f}" / f"rep_{rep}"


def _load_gene_models(checkpoint: Path) -> dict:
    gm = checkpoint / "gene_models.joblib"
    if gm.exists():
        return joblib.load(gm)
    models_dir = checkpoint / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"No gene models under {checkpoint}")
    out: dict = {}
    for mp in sorted(models_dir.glob("*.joblib")):
        out[mp.stem] = joblib.load(mp)
    if not out:
        raise FileNotFoundError(f"Empty models dir {models_dir}")
    return out


def reconstruct_imputed_matrix(
    *,
    X_full: pd.DataFrame,
    slot_dir: Path,
    target_genes: list[str],
) -> pd.DataFrame:
    """Rebuild X_imp from checkpoint models + mask (same transform as freeze)."""
    mask_path = slot_dir / "mask.npz"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask = np.asarray(np.load(mask_path)["mask"], dtype=bool)
    gene_models = _load_gene_models(slot_dir / "checkpoint")

    imp = OriginalRFECAImputer(
        validation_strategy="kfold",
        n_splits=5,
        random_state=RANDOM_STATE,
        use_scaler=False,
        max_candidates=49,
        feature_names=[str(c) for c in X_full.columns],
    )
    imp.target_genes = list(target_genes)
    imp.feature_names_in_ = [str(c) for c in X_full.columns]
    imp._gene_models_ = {g: gene_models[g] for g in target_genes if g in gene_models}
    missing = [g for g in target_genes if g not in imp._gene_models_]
    if missing:
        raise RuntimeError(
            f"{slot_dir}: missing gene models for {len(missing)} genes "
            f"(e.g. {missing[:5]})"
        )
    imp._fallback_values_ = {str(c): float(X_full[c].mean()) for c in X_full.columns}
    imp._n_svr_imputed_cells_ = 0
    imp._n_target_masked_cells_transform_ = 0
    imp.audit_ = FitAudit(
        method="OriginalRFECA",
        validation_strategy="kfold",
        random_state=RANDOM_STATE,
        selection_protocol="leakage_safe",
        use_scaler=False,
        svr_kernel="linear",
        max_candidates=49,
    )
    imp.set_target_wise_context(X_full, mask, for_transform=True)
    X_imp = imp.transform(X_full)
    if X_imp.isna().any().any():
        raise AssertionError(f"{slot_dir}: X_imp still contains NaNs")
    return X_imp


def classify_slot(
    *,
    X_imp: pd.DataFrame,
    y: pd.Series,
    missing_rate: float,
    replicate: int,
    seed: int | None,
    classifiers: list[str],
) -> pd.DataFrame:
    audit = FoldAudit()
    parts: list[pd.DataFrame] = []
    imputers = {"OriginalRFECA": IdentityImputer()}
    for clf_name in classifiers:
        part = run_classification_cv(
            X_missing=X_imp,
            y=y,
            imputers=imputers,
            classifier_name=clf_name,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            missing_rate=missing_rate,
            replicate=replicate,
            audit=audit,
        )
        if seed is not None:
            part["seed"] = seed
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["protocol"] = "post_target_wise_impute"
    out["imputer_display"] = "RFECA"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=DEFAULT_RATES,
        help="Missingness rates to process (default: 0.05 0.10 0.20 0.30)",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(MECHANISMS),
        choices=["mcar", "mar"],
    )
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=CLASSIFIERS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip slots that already have exp2_classification_raw.csv",
    )
    args = parser.parse_args()

    print("OriginalRFECA post-imputation PAM50 classification")
    print(
        json.dumps(
            {
                "rates": args.rates,
                "mechanisms": args.mechanisms,
                "replicates": args.replicates,
                "classifiers": args.classifiers,
                "n_splits": N_SPLITS,
                "random_state": RANDOM_STATE,
                "protocol": "identity_imputer_on_X_imp",
            },
            indent=2,
        )
    )
    if not args.confirm:
        print("\nRefusing to start. Re-run with --confirm when ready.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort("metabric")
    target_genes = list(PAM50_GENES)
    t0 = time.perf_counter()
    all_raw: list[pd.DataFrame] = []
    skipped = 0
    done = 0
    missing_slots: list[str] = []

    for mech in args.mechanisms:
        for rate in args.rates:
            for rep in args.replicates:
                slot = _slot_dir(mech, float(rate), int(rep))
                done_json = slot / "DONE.json"
                out_raw = slot / "exp2_classification_raw.csv"
                if not done_json.exists():
                    missing_slots.append(str(slot.relative_to(ART)))
                    print(f"[skip-missing] {slot.relative_to(ART)} (no DONE.json)")
                    continue
                if args.resume and out_raw.exists():
                    print(f"[resume] {slot.relative_to(ART)}")
                    all_raw.append(pd.read_csv(out_raw))
                    skipped += 1
                    continue

                summary = json.loads(done_json.read_text(encoding="utf-8"))
                seed = summary.get("seed")
                print(
                    f"[clf] {mech} rate={float(rate):.2f} rep={rep} "
                    f"seed={seed} ...",
                    flush=True,
                )
                t_slot = time.perf_counter()
                X_imp = reconstruct_imputed_matrix(
                    X_full=cohort.X,
                    slot_dir=slot,
                    target_genes=target_genes,
                )
                # Cache imputed matrix for reuse / audit
                cache = slot / "X_imputed.parquet"
                try:
                    X_imp.to_parquet(cache)
                except Exception:
                    X_imp.to_csv(slot / "X_imputed.csv")

                raw = classify_slot(
                    X_imp=X_imp,
                    y=cohort.y,
                    missing_rate=float(rate),
                    replicate=int(rep),
                    seed=seed,
                    classifiers=list(args.classifiers),
                )
                raw["mechanism"] = mech
                raw.to_csv(out_raw, index=False)
                summ = summarize_classification(raw)
                summ["mechanism"] = mech
                summ.to_csv(slot / "exp2_classification_summary.csv", index=False)
                try:
                    pc = summarize_classification_per_class(raw)
                    pc["mechanism"] = mech
                    pc.to_csv(slot / "exp2_classification_per_class.csv", index=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"  warn per_class: {exc}", flush=True)

                all_raw.append(raw)
                done += 1
                print(
                    f"  ok wall_s={time.perf_counter() - t_slot:.1f} "
                    f"rows={len(raw)}",
                    flush=True,
                )

    if not all_raw:
        print("No classification rows produced.")
        report = {
            "status": "EMPTY",
            "missing_slots": missing_slots,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (OUT / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1 if missing_slots else 0

    consolidated = pd.concat(all_raw, ignore_index=True)
    consolidated.to_csv(OUT / "exp2_classification_raw.csv", index=False)
    summary = summarize_classification(consolidated)
    if "mechanism" in consolidated.columns:
        # re-summarize with mechanism
        summary = (
            consolidated.groupby(
                ["mechanism", "imputer", "missing_rate", "model"], as_index=False
            )
            .agg(
                f1_mean=("f1_macro", "mean"),
                f1_std=("f1_macro", "std"),
                bal_mean=("bal_acc", "mean"),
                bal_std=("bal_acc", "std"),
                n_rows=("f1_macro", "count"),
            )
        )
    summary.to_csv(OUT / "exp2_classification_summary.csv", index=False)

    # Display-friendly EnsembleSoft pivot
    soft = summary[summary["model"] == "EnsembleSoft"].copy()
    soft.to_csv(OUT / "ensemblesoft_by_rate.csv", index=False)

    report = {
        "status": "OK",
        "protocol": "post_target_wise_impute_identity_cv",
        "note": (
            "RFECA classification uses already-imputed matrices (identity imputer). "
            "Baselines Mean/KNN/MissForest keep imputer-within-CV from metabric_full."
        ),
        "n_slots_classified": done,
        "n_slots_resumed": skipped,
        "n_slots_missing_done": len(missing_slots),
        "missing_slots": missing_slots,
        "classifiers": list(args.classifiers),
        "rates": [float(r) for r in args.rates],
        "mechanisms": list(args.mechanisms),
        "wall_seconds": time.perf_counter() - t0,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "raw": str((OUT / "exp2_classification_raw.csv").relative_to(ROOT)),
            "summary": str((OUT / "exp2_classification_summary.csv").relative_to(ROOT)),
        },
    }
    (OUT / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
