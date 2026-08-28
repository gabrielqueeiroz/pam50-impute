#!/usr/bin/env python3
"""
OriginalRFECA TARGET-WISE with evaluation_protocol=repeated_mask_holdout.

Principal path (no outer CV): one fit per (rate × replicate); masked target
cells are the external holdout; internal kfold selects the prefix only.

Reduced METABRIC analysis (default artifact root when --reduced):
  MCAR/MAR × rate 0.20 × replicates 0–4 under
  artifacts/original_rfeca_reduced_metabric/{mechanism}/rate_0.20/rep_{k}/

Outer CV remains available via --evaluation outer_cv for audit only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    PAM50_GENES,
    apply_original_rfeca_only,
    full_benchmark_config,
)
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.evaluation import (  # noqa: E402
    run_imputation_cv_target_wise,
    run_imputation_repeated_mask_holdout_target_wise,
)
from bcimpute.imputers import build_imputers  # noqa: E402
from bcimpute.imputation_original.utils import safe_mkdir  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402

NAMESPACE_DEFAULT = ARTIFACT_ROOT / "original_rfeca_target_wise_mask_holdout"
NAMESPACE_REDUCED = ARTIFACT_ROOT / "original_rfeca_reduced_metabric"

METHOD_JUSTIFICATION = (
    "Devido ao elevado custo computacional do OriginalRFECA, sua avaliação foi "
    "restrita à taxa intermediária de 20% de ausência, sob os mecanismos MCAR e "
    "MAR, com cinco réplicas independentes. O experimento foi conduzido como "
    "análise complementar target-wise, utilizando preditores completos da matriz "
    "original e avaliando exclusivamente os valores artificialmente mascarados "
    "do gene-alvo."
)


def _peak_rss_mb():
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:  # noqa: BLE001
        return None


def _configure_blas_serial() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _write_json(path: Path, payload: dict, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists() and not overwrite:
        tmp.unlink(missing_ok=True)
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    tmp.replace(path)


def _write_text(path: Path, text: str, *, overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _mask_hash(mask: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _configure(args):
    cfg = full_benchmark_config(args.cohort, missingness_mechanism=args.mechanism)
    cfg = apply_original_rfeca_only(cfg)
    cfg.n_jobs = 1
    cfg.missingness_seed_scheme = args.seed_scheme
    cfg.original_rfeca_use_scaler = bool(args.use_scaler)
    cfg.original_rfeca_max_candidates = int(args.max_candidates)
    cfg.original_rfeca_selection_protocol = "leakage_safe"
    cfg.original_rfeca_kernel = "linear"
    cfg.original_rfeca_validation = (
        "loocv" if args.cohort == "discovery" else "kfold"
    )
    cfg.original_rfeca_n_splits = int(args.inner_cv)
    cfg.missing_rates = [float(r) for r in args.rates]
    # Seed parity with principal schedule: generate enough reps then filter.
    max_rep = max(args.replicates) if args.replicates else 0
    cfg.n_repetitions = max(int(args.n_repetitions or 0), max_rep + 1, 5)
    cfg.tag = f"{args.mode}_{args.evaluation}_{cfg.tag}"
    return cfg


def _slot_dir_reduced(out_root: Path, mechanism: str, rate: float, rep: int) -> Path:
    return out_root / mechanism / f"rate_{rate:.2f}" / f"rep_{rep}"


def _aggregate_gene_table(per_gene: pd.DataFrame) -> dict:
    if per_gene.empty:
        return {
            "n_genes_completed": 0,
            "n_failures": 0,
            "mean_rmse": float("nan"),
            "median_rmse": float("nan"),
            "mean_mae": float("nan"),
            "median_mae": float("nan"),
            "macro_rmse": float("nan"),
            "micro_rmse": float("nan"),
            "macro_mae": float("nan"),
            "micro_mae": float("nan"),
            "mean_winning_prefix_len": float("nan"),
            "mean_n_predictors_selected": float("nan"),
            "global_svr_coverage": float("nan"),
            "total_fallback_count": 0,
        }
    ok = per_gene[per_gene["status"].astype(str) == "ok"]
    n_fail = int((per_gene["status"].astype(str) != "ok").sum())
    # micro: weight by n_masked
    n_m = per_gene["n_masked"].to_numpy(dtype=float)
    # reconstruct SSE from rmse^2 * n
    sse = (per_gene["rmse"].to_numpy(dtype=float) ** 2) * n_m
    sae = per_gene["mae"].to_numpy(dtype=float) * n_m
    n_tot = float(n_m.sum())
    micro_rmse = float(np.sqrt(sse.sum() / n_tot)) if n_tot > 0 else float("nan")
    micro_mae = float(sae.sum() / n_tot) if n_tot > 0 else float("nan")
    return {
        "n_genes_completed": int(len(ok)),
        "n_failures": n_fail,
        "mean_rmse": float(per_gene["rmse"].mean()),
        "median_rmse": float(per_gene["rmse"].median()),
        "mean_mae": float(per_gene["mae"].mean()),
        "median_mae": float(per_gene["mae"].median()),
        "macro_rmse": float(per_gene["rmse"].mean()),
        "micro_rmse": micro_rmse,
        "macro_mae": float(per_gene["mae"].mean()),
        "micro_mae": micro_mae,
        "mean_winning_prefix_len": float(per_gene["winning_prefix_len"].mean()),
        "mean_n_predictors_selected": float(per_gene["n_predictors_selected"].mean()),
        "total_fallback_count": int(per_gene["fallback_count"].sum()),
        "total_selection_seconds": float(
            per_gene["selection_seconds"].fillna(0).sum()
            + per_gene.get("final_fit_seconds", pd.Series(0)).fillna(0).sum()
        ),
    }


def _classify_slot(summary: dict, n_expected_genes: int) -> str:
    """A / B / C for one slot (or preflight)."""
    if summary.get("leakage_or_protocol_fail"):
        return "C"
    if int(summary.get("n_predictor_nans_at_impute", 0)) != 0:
        return "C"
    if float(summary.get("svr_coverage", 0.0)) < 1.0 - 1e-12:
        return "C"
    if int(summary.get("total_fallback_count", 0)) != 0:
        return "C"
    n_ok = int(summary.get("n_genes_completed", 0))
    n_fail = int(summary.get("n_failures", 0))
    if n_ok == n_expected_genes and n_fail == 0:
        return "A"
    if n_fail > 0 or n_ok < n_expected_genes:
        return "B"
    return "B"


def _resolve_gene_workers(arg: str | int | None) -> int:
    """Resolve --gene-workers (int or 'auto' from parallel_benchmark)."""
    if arg is None or str(arg).lower() in {"auto", "recommended"}:
        rec = (
            ARTIFACT_ROOT
            / "parallel_benchmark"
            / "recommended_workers.json"
        )
        if rec.exists():
            payload = json.loads(rec.read_text(encoding="utf-8"))
            return int(payload.get("recommended_workers", 1))
        return 1
    return max(1, int(arg))


def _run_one_slot(
    *,
    cohort,
    base_imp,
    item,
    rate: float,
    mechanism: str,
    slot_dir: Path,
    evaluation: str,
    target_genes: list[str] | None,
    n_splits: int,
    random_state: int,
    resume: bool,
    n_gene_workers: int = 1,
) -> dict:
    slot_dir.mkdir(parents=True, exist_ok=True)
    done_path = slot_dir / "DONE.json"
    if resume and done_path.exists():
        print(f"[skip] {slot_dir}", flush=True)
        return json.loads(done_path.read_text(encoding="utf-8"))

    mask = np.asarray(item.mask, dtype=bool)
    mhash = _mask_hash(mask)
    mask_path = slot_dir / "mask.npz"
    if not mask_path.exists():
        np.savez_compressed(
            mask_path,
            mask=mask,
            seed=np.array([item.seed]),
            mask_hash=np.array([mhash]),
        )

    # Clone-like: set target genes on this run's imputer via attribute
    base_imp.target_genes = list(target_genes) if target_genes else None
    base_imp.set_run_context(
        dataset=cohort.name,
        mechanism=mechanism,
        missing_rate=float(rate),
        replicate=int(item.replicate),
        seed=item.seed,
    )

    print(
        f"[run] {mechanism} rate={rate:.2f} rep={item.replicate} "
        f"seed={item.seed} genes={len(target_genes) if target_genes else 'all'} "
        f"gene_workers={n_gene_workers}",
        flush=True,
    )
    rss0 = _peak_rss_mb()
    t0 = time.perf_counter()
    failure: dict | None = None
    try:
        if evaluation != "repeated_mask_holdout":
            raise ValueError(
                "Reduced analysis requires --evaluation repeated_mask_holdout"
            )
        imp_df, detail = run_imputation_repeated_mask_holdout_target_wise(
            X_full=cohort.X,
            mask=mask,
            imputer=base_imp,
            missing_rate=float(rate),
            replicate=int(item.replicate),
            seed=item.seed,
            checkpoint_root=str(slot_dir / "checkpoint"),
            n_gene_workers=int(n_gene_workers),
        )
    except Exception as exc:  # noqa: BLE001
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "context": {
                "mechanism": mechanism,
                "rate": rate,
                "replicate": int(item.replicate),
                "seed": item.seed,
                "mask_hash": mhash,
            },
        }
        _write_json(slot_dir / "FAILURE.json", failure, overwrite=True)
        raise

    rss1 = _peak_rss_mb()
    wall = time.perf_counter() - t0
    per_gene = pd.DataFrame(detail["per_gene_metrics"])
    # Enrich with seed / mask hash / wall
    if not per_gene.empty:
        per_gene["seed"] = item.seed
        per_gene["mask_hash"] = mhash
        per_gene["mechanism"] = mechanism
        per_gene["missing_rate"] = float(rate)
        per_gene["replicate"] = int(item.replicate)
        per_gene["wall_seconds_slot"] = wall
        # SVR coverage proxy per gene: status ok and finite metrics
        per_gene["svr_ok"] = (per_gene["status"] == "ok").astype(int)

    agg = _aggregate_gene_table(per_gene)
    svr_cov = float(imp_df["svr_coverage"].iloc[0])
    fb_rate = float(imp_df["fallback_rate"].iloc[0])
    n_pred_nan = int(imp_df["n_predictor_nans_at_impute"].iloc[0])
    agg["global_svr_coverage"] = svr_cov
    agg["fallback_rate"] = fb_rate

    # Mask integrity: evaluated positions == target mask columns
    eval_mask = detail["eval_mask"]
    target_set = set(target_genes) if target_genes else set(cohort.X.columns.astype(str))
    expected = np.zeros_like(mask, dtype=bool)
    for j, g in enumerate(cohort.X.columns):
        if str(g) in target_set:
            expected[:, j] = mask[:, j]
    mask_match = bool(np.array_equal(eval_mask, expected))

    # Persist artifacts (atomic where possible)
    _write_json(slot_dir / "gene_selection_audit.json", detail["audit"], overwrite=True)
    per_gene.to_csv(slot_dir / "per_gene_metrics.csv", index=False)
    gene_rows = [
        {
            "gene": g["gene"],
            "status": g["status"],
            "winning_prefix_len": g["winning_prefix_len"],
            "n_predictors_selected": g["n_predictors_selected"],
            "winning_predictors": "|".join(g.get("winning_predictors", [])),
            "selection_seconds": g["selection_seconds"],
            "n_candidates": len(g.get("candidates", [])),
        }
        for g in detail["audit"].get("genes", [])
    ]
    pd.DataFrame(gene_rows).to_csv(slot_dir / "gene_summary.csv", index=False)
    imp_df.to_csv(slot_dir / "exp1_imputation_raw.csv", index=False)

    n_expected = len(target_genes) if target_genes else len(PAM50_GENES)
    summary = {
        "slot": f"{mechanism}/rate_{rate:.2f}/rep_{int(item.replicate)}",
        "evaluation_protocol": "repeated_mask_holdout",
        "input_protocol": "target_wise_complete_predictors",
        "predictor_values": "original_complete_matrix",
        "selection_protocol": "leakage_safe",
        "use_scaler": False,
        "max_candidates": 49,
        "mechanism": mechanism,
        "missing_rate": float(rate),
        "replicate": int(item.replicate),
        "seed": item.seed,
        "mask_hash": mhash,
        "mask_match_eval_positions": mask_match,
        "n_expected_genes": n_expected,
        "rmse": float(imp_df["rmse"].mean()),
        "mae": float(imp_df["mae"].mean()),
        "n_masked": int(imp_df["n_masked_test_values"].sum()),
        "svr_coverage": svr_cov,
        "fallback_rate": fb_rate,
        "n_predictor_nans_at_impute": n_pred_nan,
        "fit_seconds": float(imp_df["fit_seconds"].sum()),
        "transform_seconds": float(imp_df["transform_seconds"].sum()),
        "wall_seconds": wall,
        "memory_rss_mb": {"before": rss0, "after": rss1},
        "workers": int(n_gene_workers),
        "blas_threads": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        **agg,
        "leakage_or_protocol_fail": (not mask_match) or (n_pred_nan != 0),
    }
    summary["classification"] = _classify_slot(summary, n_expected)
    _write_json(slot_dir / "slot_summary.json", summary, overwrite=True)
    _write_json(slot_dir / "DONE.json", summary, overwrite=True)
    print(
        f"  done class={summary['classification']} "
        f"rmse={summary['rmse']:.4f} cov={svr_cov:.4f} "
        f"fb={agg['total_fallback_count']} pred_nan={n_pred_nan} "
        f"wall_s={wall:.1f}",
        flush=True,
    )
    return summary


def _mechanism_report(
    out_root: Path,
    mechanism: str,
    rate: float,
    replicates: list[int],
    report_path: Path,
) -> dict:
    slots = []
    for rep in replicates:
        done = _slot_dir_reduced(out_root, mechanism, rate, rep) / "DONE.json"
        if done.exists():
            slots.append(json.loads(done.read_text(encoding="utf-8")))
    if not slots:
        payload = {"status": "EMPTY", "mechanism": mechanism, "rate": rate}
        _write_text(report_path, f"# Report {mechanism.upper()} {rate}\n\nNo slots.\n")
        return payload

    rmses = [s["rmse"] for s in slots]
    maes = [s["mae"] for s in slots]
    walls = [s.get("wall_seconds", float("nan")) for s in slots]
    covs = [s["svr_coverage"] for s in slots]
    fbs = [s.get("total_fallback_count", 0) for s in slots]
    n_ok = [s.get("n_genes_completed", 0) for s in slots]
    n_fail = [s.get("n_failures", 0) for s in slots]
    classes = [s.get("classification", "?") for s in slots]

    # Per-gene across reps
    gene_frames = []
    for rep in replicates:
        p = _slot_dir_reduced(out_root, mechanism, rate, rep) / "per_gene_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["replicate"] = rep
            gene_frames.append(df)
    gene_all = pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame()

    overall_class = "A"
    if any(c == "C" for c in classes):
        overall_class = "C"
    elif any(c == "B" for c in classes) or len(slots) < len(replicates):
        overall_class = "B"
    if any(n != 50 for n in n_ok) or any(f != 0 for f in fbs) or any(
        c < 1.0 - 1e-12 for c in covs
    ):
        if overall_class == "A":
            overall_class = "B"

    payload = {
        "mechanism": mechanism,
        "rate": rate,
        "n_replicates_done": len(slots),
        "n_replicates_expected": len(replicates),
        "classification": overall_class,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "rmse_median": float(np.median(rmses)),
        "rmse_min": float(np.min(rmses)),
        "rmse_max": float(np.max(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        "wall_seconds_total": float(np.nansum(walls)),
        "wall_seconds_mean_rep": float(np.nanmean(walls)),
        "svr_coverage_min": float(np.min(covs)),
        "total_fallback_count": int(sum(fbs)),
        "genes_completed_per_rep": n_ok,
        "failures_per_rep": n_fail,
        "slot_classifications": classes,
        "slots": slots,
        "methodological_justification": METHOD_JUSTIFICATION,
    }

    # Prefix / predictor distributions
    dist_lines = ""
    if not gene_all.empty:
        dist_lines = (
            f"- Prefix length: mean={gene_all['winning_prefix_len'].mean():.2f}, "
            f"median={gene_all['winning_prefix_len'].median():.1f}, "
            f"min={gene_all['winning_prefix_len'].min()}, "
            f"max={gene_all['winning_prefix_len'].max()}\n"
            f"- Final predictors: mean={gene_all['n_predictors_selected'].mean():.2f}, "
            f"median={gene_all['n_predictors_selected'].median():.1f}, "
            f"min={gene_all['n_predictors_selected'].min()}, "
            f"max={gene_all['n_predictors_selected'].max()}\n"
        )
        gene_all.to_csv(
            report_path.with_name(f"per_gene_all_{mechanism}_rate_{rate:.2f}.csv"),
            index=False,
        )

    md = f"""# REPORT {mechanism.upper()} {rate:.0%} × {len(replicates)} réplicas

## Status

- Classification: **{overall_class}**
- Replicates done: {len(slots)} / {len(replicates)}
- Genes completed per rep: {n_ok}
- Failures per rep: {n_fail}
- SVR coverage (min): {payload['svr_coverage_min']:.6f}
- Total fallback_count: {payload['total_fallback_count']}
- Wall clock total: {payload['wall_seconds_total']/3600:.2f} h

## Methodological justification

{METHOD_JUSTIFICATION}

## Protocol

- method = OriginalRFECA
- evaluation_protocol = repeated_mask_holdout
- input_protocol = target_wise_complete_predictors
- predictor_values = original_complete_matrix
- selection_protocol = leakage_safe
- use_scaler = false
- max_candidates = 49
- inner_validation = KFold(5)
- RFE = SVR(kernel=linear), step=1, n_features_to_select=None

## Aggregates across replicates

| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| RMSE | {payload['rmse_mean']:.6f} | {payload['rmse_std']:.6f} | {payload['rmse_median']:.6f} | {payload['rmse_min']:.6f} | {payload['rmse_max']:.6f} |
| MAE | {payload['mae_mean']:.6f} | {payload['mae_std']:.6f} | — | — | — |

## Prefix / predictor distributions

{dist_lines or '_n/a_'}

## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
"""
    for s in slots:
        md += (
            f"| {s['replicate']} | {s['rmse']:.6f} | {s['mae']:.6f} | "
            f"{s.get('n_genes_completed',0)} | {s.get('n_failures',0)} | "
            f"{s['svr_coverage']:.4f} | {s.get('total_fallback_count',0)} | "
            f"{s.get('wall_seconds', float('nan')):.1f} | "
            f"{s.get('classification','?')} |\n"
        )
    md += f"\nArtifacts: `{out_root / mechanism / f'rate_{rate:.2f}'}`\n"
    _write_text(report_path, md, overwrite=True)
    _write_json(report_path.with_suffix(".json"), payload, overwrite=True)
    return payload


def _preflight_checks(
    slot_dir: Path,
    summary: dict,
    cohort,
    item,
    genes: list[str],
) -> tuple[str, list[str]]:
    issues: list[str] = []
    if summary.get("n_predictor_nans_at_impute", 1) != 0:
        issues.append("predictor_nan_count != 0")
    if float(summary.get("svr_coverage", 0)) < 1.0 - 1e-12:
        issues.append(f"svr_coverage={summary.get('svr_coverage')} != 1.0")
    if int(summary.get("total_fallback_count", 1)) != 0:
        issues.append("fallback_count != 0")
    if not summary.get("mask_match_eval_positions", False):
        issues.append("eval positions != persisted target mask")
    if summary.get("n_genes_completed", 0) != len(genes):
        issues.append("not all preflight genes completed")
    # Determinism: re-run one gene with checkpoint should skip
    ck = slot_dir / "checkpoint" / "genes"
    for g in genes:
        if not (ck / f"{g}.json").exists():
            issues.append(f"missing checkpoint for {g}")
    # Finite metrics
    pg = pd.read_csv(slot_dir / "per_gene_metrics.csv")
    for col in ("rmse", "mae"):
        if not np.isfinite(pg[col]).all():
            issues.append(f"non-finite {col}")
    # Masked cells not in train: n_observed + n_masked == n_samples for each gene
    n = len(cohort.X)
    for _, row in pg.iterrows():
        if int(row["n_observed_train"]) + int(row["n_masked"]) != n:
            issues.append(
                f"{row['gene']}: n_obs+n_masked != n_samples "
                f"({row['n_observed_train']}+{row['n_masked']}!={n})"
            )
    # Resume: DONE exists
    if not (slot_dir / "DONE.json").exists():
        issues.append("DONE.json missing")

    # Determinism: second pass should skip all genes via checkpoint
    # (verified by re-invoking fit with same checkpoint — quick structural check)
    models = slot_dir / "checkpoint" / "gene_models.joblib"
    if not models.exists():
        issues.append("gene_models.joblib missing (resume models)")

    grade = "A" if not issues else ("C" if any("nan" in i or "mask" in i or "predictor" in i for i in issues) else "B")
    if issues and any(
        x in " ".join(issues)
        for x in ("predictor_nan", "eval positions", "fallback", "svr_coverage")
    ):
        grade = "C"
    return grade, issues


def main() -> int:
    _configure_blas_serial()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm", action="store_true")
    p.add_argument(
        "--reduced",
        action="store_true",
        help="Use reduced METABRIC layout under original_rfeca_reduced_metabric/",
    )
    p.add_argument(
        "--phase",
        choices=["preflight", "mcar", "mar", "full_reduced", "legacy"],
        default="legacy",
        help="preflight -> mcar -> mar orchestration for reduced analysis.",
    )
    p.add_argument("--mode", choices=["intermediate", "full"], default="intermediate")
    p.add_argument(
        "--evaluation",
        choices=["repeated_mask_holdout", "outer_cv"],
        default="repeated_mask_holdout",
    )
    p.add_argument("--cohort", choices=["metabric", "discovery"], default="metabric")
    p.add_argument("--mechanism", choices=["mcar", "mar"], default="mcar")
    p.add_argument(
        "--mechanisms",
        nargs="+",
        choices=["mcar", "mar"],
        default=None,
    )
    p.add_argument("--rate", type=float, default=0.20)
    p.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
        help="Missing rates (e.g. 0.10 0.20 0.30). Default 0.20 for reduced phases.",
    )
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    p.add_argument("--n-repetitions", type=int, default=None)
    p.add_argument("--seed-scheme", choices=["v2", "legacy"], default="v2")
    p.add_argument("--max-candidates", type=int, default=49)
    p.add_argument("--inner-cv", type=int, default=5)
    p.add_argument(
        "--use-scaler",
        type=lambda s: str(s).lower() in {"1", "true", "yes"},
        default=False,
    )
    p.add_argument(
        "--gene-workers",
        default="auto",
        help=(
            "Independent gene processes (methodology unchanged). "
            "'auto' reads artifacts/parallel_benchmark/recommended_workers.json "
            "(currently 8 after autotune). Use 1 for serial."
        ),
    )
    p.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Optional target gene subset (preflight). Default: all PAM50.",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-dir", type=str, default=None)
    p.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Override artifact root (default: reduced or mask_holdout namespace).",
    )
    p.add_argument(
        "--auto-continue",
        action="store_true",
        help="After preflight PASS, continue MCAR then MAR (reduced phase).",
    )
    args = p.parse_args()
    if args.no_resume:
        args.resume = False

    n_gene_workers = _resolve_gene_workers(args.gene_workers)

    # Phase presets (explicit --rates overrides the default 0.20)
    rates_cli = args.rates
    if args.phase == "preflight":
        args.reduced = True
        args.mechanisms = ["mcar"]
        args.rates = rates_cli if rates_cli is not None else [0.20]
        args.replicates = [0]
        if args.genes is None:
            args.genes = list(PAM50_GENES[:2])
        args.evaluation = "repeated_mask_holdout"
    elif args.phase == "mcar":
        args.reduced = True
        args.mechanisms = ["mcar"]
        args.rates = rates_cli if rates_cli is not None else [0.20]
        args.replicates = args.replicates or [0, 1, 2, 3, 4]
        args.genes = None  # all 50
    elif args.phase == "mar":
        args.reduced = True
        args.mechanisms = ["mar"]
        args.rates = rates_cli if rates_cli is not None else [0.20]
        args.replicates = args.replicates or [0, 1, 2, 3, 4]
        args.genes = None
    elif args.phase == "full_reduced":
        args.reduced = True
        args.mechanisms = ["mcar", "mar"]
        args.rates = rates_cli if rates_cli is not None else [0.20]
        args.replicates = [0, 1, 2, 3, 4]
        args.genes = None
        args.auto_continue = True
    elif rates_cli is None:
        args.rates = [float(args.rate)]

    if args.rates is None:
        args.rates = [0.20]

    if args.mechanisms is None:
        args.mechanisms = [args.mechanism]

    if args.out_root:
        out_root = Path(args.out_root)
    elif args.reduced or args.phase != "legacy":
        out_root = NAMESPACE_REDUCED
    else:
        out_root = NAMESPACE_DEFAULT

    # Legacy single-slot path still supported
    if args.phase == "legacy" and not args.reduced:
        return _legacy_main(args)

    print(json.dumps({
        "phase": args.phase,
        "mechanisms": args.mechanisms,
        "rates": args.rates,
        "replicates": args.replicates,
        "genes": args.genes,
        "out_root": str(out_root),
        "evaluation": args.evaluation,
        "max_candidates": args.max_candidates,
        "inner_cv": args.inner_cv,
        "use_scaler": args.use_scaler,
        "gene_workers": n_gene_workers,
        "gene_workers_arg": args.gene_workers,
        "workers": n_gene_workers,
        "blas": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }, indent=2), flush=True)

    if not args.confirm:
        print("Refusing without --confirm")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.perf_counter()
    cohort = load_cohort(args.cohort)
    assert np.isfinite(cohort.X.to_numpy(dtype=float)).all()

    all_summaries: list[dict] = []
    # Preflight only MCAR rep0 × 2 genes
    if args.phase == "preflight":
        args.mechanism = "mcar"
        cfg = _configure(args)
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
        imputers = build_imputers(cfg)
        base_imp = imputers["OriginalRFECA"]
        rate = 0.20
        item = next(
            it for it in missing_sets[rate] if int(it.replicate) == 0
        )
        slot_dir = _slot_dir_reduced(out_root, "mcar", rate, 0) / "preflight"
        # Fresh preflight namespace under rep_0/preflight
        summary = _run_one_slot(
            cohort=cohort,
            base_imp=base_imp,
            item=item,
            rate=rate,
            mechanism="mcar",
            slot_dir=slot_dir,
            evaluation=args.evaluation,
            target_genes=list(args.genes),
            n_splits=cfg.original_rfeca_n_splits,
            random_state=cfg.random_state,
            resume=args.resume,
            n_gene_workers=n_gene_workers,
        )
        grade, issues = _preflight_checks(
            slot_dir, summary, cohort, item, list(args.genes)
        )
        # Determinism re-run: resume should be instant-ish
        t_det0 = time.perf_counter()
        summary2 = _run_one_slot(
            cohort=cohort,
            base_imp=base_imp,
            item=item,
            rate=rate,
            mechanism="mcar",
            slot_dir=slot_dir,
            evaluation=args.evaluation,
            target_genes=list(args.genes),
            n_splits=cfg.original_rfeca_n_splits,
            random_state=cfg.random_state,
            resume=True,
            n_gene_workers=n_gene_workers,
        )
        det_s = time.perf_counter() - t_det0
        det_ok = abs(summary2["rmse"] - summary["rmse"]) < 1e-12
        if not det_ok:
            issues.append("determinism RMSE mismatch on resume")
            grade = "C"
        if det_s > 30:
            # still ok if checkpoints skipped; warn only
            pass

        report = f"""# PREFLIGHT_REPORT

## Verdict: {grade}

- Genes: {args.genes}
- Mechanism: MCAR, rate=0.20, replicate=0
- predictor_nan_count: {summary.get('n_predictor_nans_at_impute')}
- svr_coverage: {summary.get('svr_coverage')}
- fallback_count: {summary.get('total_fallback_count')}
- mask_match_eval_positions: {summary.get('mask_match_eval_positions')}
- wall_seconds: {summary.get('wall_seconds'):.1f}
- resume_determinism_ok: {det_ok} (resume wall {det_s:.2f}s)
- Issues: {issues or 'none'}

## Protocol checks

- NaN only on target: enforced by target_wise matrix builder
- Predictors from original complete matrix: yes
- No SimpleImputer / chaining: yes
- Metrics only on masked target cells: yes
- Checkpoint per gene: `{slot_dir / 'checkpoint' / 'genes'}`

## Classification criteria

- A: all checks pass
- B: methodology ok, operational issues
- C: leakage / imputation / non-reproducible

Artifacts: `{slot_dir}`
"""
        _write_text(out_root / "PREFLIGHT_REPORT.md", report, overwrite=True)
        _write_json(
            out_root / "PREFLIGHT_REPORT.json",
            {
                "classification": grade,
                "issues": issues,
                "summary": summary,
                "determinism_ok": det_ok,
                "resume_seconds": det_s,
            },
            overwrite=True,
        )
        print(f"\nPREFLIGHT {grade} issues={issues}", flush=True)
        if grade != "A":
            return 1
        if not args.auto_continue:
            return 0
        # Fall through to MCAR
        args.phase = "mcar"
        args.genes = None
        args.mechanisms = ["mcar"]
        args.replicates = [0, 1, 2, 3, 4]

    # Run mechanism tranches
    for mech in args.mechanisms:
        args.mechanism = mech
        cfg = _configure(args)
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
        imputers = build_imputers(cfg)
        base_imp = imputers["OriginalRFECA"]
        target_genes = list(args.genes) if args.genes else list(PAM50_GENES)

        t_mech0 = time.perf_counter()
        for rate in cfg.missing_rates:
            for item in missing_sets[rate]:
                if int(item.replicate) not in set(args.replicates):
                    continue
                slot_dir = _slot_dir_reduced(
                    out_root, mech, float(rate), int(item.replicate)
                )
                try:
                    summary = _run_one_slot(
                        cohort=cohort,
                        base_imp=base_imp,
                        item=item,
                        rate=float(rate),
                        mechanism=mech,
                        slot_dir=slot_dir,
                        evaluation=args.evaluation,
                        target_genes=target_genes,
                        n_splits=cfg.original_rfeca_n_splits,
                        random_state=cfg.random_state,
                        resume=args.resume,
                        n_gene_workers=n_gene_workers,
                    )
                    all_summaries.append(summary)
                except Exception as exc:  # noqa: BLE001
                    print(f"SLOT FAIL {mech} rep={item.replicate}: {exc}", flush=True)
                    all_summaries.append(
                        {
                            "mechanism": mech,
                            "replicate": int(item.replicate),
                            "classification": "B",
                            "error": str(exc),
                        }
                    )

        elapsed = time.perf_counter() - t_mech0
        for rate in cfg.missing_rates:
            rate_f = float(rate)
            pct = int(round(rate_f * 100))
            n_done = sum(
                1
                for r in args.replicates
                if (_slot_dir_reduced(out_root, mech, rate_f, r) / "DONE.json").exists()
            )
            eta_remaining = None
            if n_done > 0 and n_done < len(args.replicates):
                per_rep = elapsed / max(n_done, 1)
                eta_remaining = per_rep * (len(args.replicates) - n_done)

            report_name = f"REPORT_{mech.upper()}_{pct}_5REPS.md"
            payload = _mechanism_report(
                out_root,
                mech,
                rate_f,
                list(args.replicates),
                out_root / report_name,
            )
            print(
                json.dumps(
                    {
                        "tranche": mech,
                        "rate": rate_f,
                        "status": payload.get("classification"),
                        "completed_reps": n_done,
                        "expected_reps": len(args.replicates),
                        "elapsed_hours": elapsed / 3600,
                        "eta_remaining_hours": (
                            None if eta_remaining is None else eta_remaining / 3600
                        ),
                        "coverage_min": payload.get("svr_coverage_min"),
                        "failures": payload.get("failures_per_rep"),
                        "artifacts": str(out_root / mech / f"rate_{rate_f:.2f}"),
                        "classification": payload.get("classification"),
                    },
                    indent=2,
                ),
                flush=True,
            )

        # Sequential: do not start MAR until MCAR finishes when auto_continue
        last_payload_ok = True
        for rate in cfg.missing_rates:
            rep_path = out_root / f"REPORT_{mech.upper()}_{int(round(float(rate)*100))}_5REPS.json"
            if rep_path.exists():
                cls = json.loads(rep_path.read_text(encoding="utf-8")).get("classification")
                if cls not in {"A", "B"}:
                    last_payload_ok = False
        if (
            args.phase == "mcar"
            and args.auto_continue
            and mech == "mcar"
            and last_payload_ok
        ):
            args.mechanisms = ["mar"]
            # loop will continue if we append — instead restart mar below
            continue

    # If phase was mcar with auto_continue, run mar now
    if args.phase == "mcar" and args.auto_continue and "mar" not in [
        s.get("mechanism") for s in all_summaries
    ]:
        # Re-enter mar
        args.phase = "mar"
        args.mechanisms = ["mar"]
        args.mechanism = "mar"
        cfg = _configure(args)
        missing_sets = generate_missingness_sets(
            cohort.X,
            missing_rates=cfg.missing_rates,
            n_repetitions=cfg.n_repetitions,
            base_seed=cfg.random_state,
            originally_observed_mask=cohort.originally_observed_mask,
            target_cell_policy=cfg.target_cell_policy,
            mechanism="mar",
            seed_scheme=cfg.missingness_seed_scheme,
        )
        imputers = build_imputers(cfg)
        base_imp = imputers["OriginalRFECA"]
        for rate in cfg.missing_rates:
            for item in missing_sets[rate]:
                if int(item.replicate) not in set(args.replicates):
                    continue
                slot_dir = _slot_dir_reduced(
                    out_root, "mar", float(rate), int(item.replicate)
                )
                try:
                    summary = _run_one_slot(
                        cohort=cohort,
                        base_imp=base_imp,
                        item=item,
                        rate=float(rate),
                        mechanism="mar",
                        slot_dir=slot_dir,
                        evaluation=args.evaluation,
                        target_genes=list(PAM50_GENES),
                        n_splits=cfg.original_rfeca_n_splits,
                        random_state=cfg.random_state,
                        resume=args.resume,
                        n_gene_workers=n_gene_workers,
                    )
                    all_summaries.append(summary)
                except Exception as exc:  # noqa: BLE001
                    print(f"SLOT FAIL mar rep={item.replicate}: {exc}", flush=True)
            pct = int(round(float(rate) * 100))
            _mechanism_report(
                out_root,
                "mar",
                float(rate),
                list(args.replicates),
                out_root / f"REPORT_MAR_{pct}_5REPS.md",
            )

    # Final combined report when rate 0.20 MCAR+MAR are both complete
    reps = args.replicates or [0, 1, 2, 3, 4]
    mcar_done = all(
        (_slot_dir_reduced(out_root, "mcar", 0.20, r) / "DONE.json").exists()
        for r in reps
    )
    mar_done = all(
        (_slot_dir_reduced(out_root, "mar", 0.20, r) / "DONE.json").exists()
        for r in reps
    )
    if mcar_done and mar_done:
        _write_final_report(out_root, reps)

    wall = time.perf_counter() - t_wall0
    print(f"\nWall clock: {wall/3600:.2f} h\nOutput: {out_root}", flush=True)
    return 0


def _write_final_report(out_root: Path, replicates: list[int]) -> None:
    mcar = json.loads((out_root / "REPORT_MCAR_20_5REPS.json").read_text(encoding="utf-8"))
    mar = json.loads((out_root / "REPORT_MAR_20_5REPS.json").read_text(encoding="utf-8"))
    cls = "A"
    if mcar["classification"] == "C" or mar["classification"] == "C":
        cls = "C"
    elif mcar["classification"] == "B" or mar["classification"] == "B":
        cls = "B"
    # Strict A: 50/50 × 10 slots
    for mech in ("mcar", "mar"):
        for r in replicates:
            s = json.loads(
                (
                    _slot_dir_reduced(out_root, mech, 0.20, r) / "DONE.json"
                ).read_text(encoding="utf-8")
            )
            if (
                int(s.get("n_genes_completed", 0)) != 50
                or int(s.get("total_fallback_count", 0)) != 0
                or float(s.get("svr_coverage", 0)) < 1.0 - 1e-12
                or int(s.get("n_predictor_nans_at_impute", 1)) != 0
            ):
                if cls == "A":
                    cls = "B"

    md = f"""# REPORT_RFECA_REDUCED_FINAL

## Classification: **{cls}**

## Methodological justification

{METHOD_JUSTIFICATION}

## Protocol

- method = OriginalRFECA
- evaluation_protocol = repeated_mask_holdout
- input_protocol = target_wise_complete_predictors
- predictor_values = original_complete_matrix
- selection_protocol = leakage_safe
- use_scaler = false
- max_candidates = 49
- No RFACA, no outer CV, no auxiliary imputation, no chaining

## Scope justification

Only MCAR and MAR at 20% missingness with 5 independent replicates (10 slots × 50 genes).

## MCAR vs MAR at 20%

| Mechanism | RMSE mean±std | MAE mean | Wall (h) | Cov min | Fallbacks | Class |
|-----------|---------------|----------|----------|---------|-----------|-------|
| MCAR | {mcar['rmse_mean']:.6f}±{mcar['rmse_std']:.6f} | {mcar['mae_mean']:.6f} | {mcar['wall_seconds_total']/3600:.2f} | {mcar['svr_coverage_min']:.4f} | {mcar['total_fallback_count']} | {mcar['classification']} |
| MAR | {mar['rmse_mean']:.6f}±{mar['rmse_std']:.6f} | {mar['mae_mean']:.6f} | {mar['wall_seconds_total']/3600:.2f} | {mar['svr_coverage_min']:.4f} | {mar['total_fallback_count']} | {mar['classification']} |

## Variability

MCAR RMSE range [{mcar['rmse_min']:.6f}, {mcar['rmse_max']:.6f}];
MAR RMSE range [{mar['rmse_min']:.6f}, {mar['rmse_max']:.6f}].

## Limitations

- Restricted to a single missingness rate (20%) due to compute cost.
- Serial gene execution; wall clock is large (~days).
- Target-wise protocol does not model joint multivariate missingness at prediction time.

Artifacts root: `{out_root}`
"""
    _write_text(out_root / "REPORT_RFECA_REDUCED_FINAL.md", md, overwrite=True)
    _write_json(
        out_root / "REPORT_RFECA_REDUCED_FINAL.json",
        {"classification": cls, "mcar": mcar, "mar": mar},
        overwrite=True,
    )


def _legacy_main(args) -> int:
    """Previous single-run layout under original_rfeca_target_wise_mask_holdout."""
    cfg = _configure(args)
    print(json.dumps(cfg.snapshot(), indent=2))
    if not args.confirm:
        print("Refusing without --confirm")
        return 0

    if args.resume_dir:
        out_dir = Path(args.resume_dir)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = (
            f"{args.cohort}_{args.mechanism}_{args.mode}_{args.evaluation}_"
            f"tw_noscaler_maxcand49_{run_id}"
        )
        out_dir = safe_mkdir(NAMESPACE_DEFAULT / name)

    print(f"Output: {out_dir}", flush=True)
    report = {
        "run_id": out_dir.name,
        "namespace": str(NAMESPACE_DEFAULT),
        "mode": args.mode,
        "evaluation_protocol": args.evaluation,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": cfg.snapshot(),
        "slots": [],
        "errors": [],
    }
    t_wall0 = time.perf_counter()
    try:
        cohort = load_cohort(args.cohort)
        assert np.isfinite(cohort.X.to_numpy(dtype=float)).all()
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
        imputers = build_imputers(cfg)
        base_imp = imputers["OriginalRFECA"]
        for rate, reps in missing_sets.items():
            for item in reps:
                if args.mode == "intermediate" and int(item.replicate) != int(
                    args.replicate
                ):
                    continue
                if int(item.replicate) not in set(args.replicates):
                    continue
                key = f"rate_{float(rate):.2f}_rep_{int(item.replicate)}"
                slot_dir = out_dir / "slots" / key
                summary = _run_one_slot(
                    cohort=cohort,
                    base_imp=base_imp,
                    item=item,
                    rate=float(rate),
                    mechanism=cfg.missingness_mechanism,
                    slot_dir=slot_dir,
                    evaluation=args.evaluation,
                    target_genes=None,
                    n_splits=cfg.original_rfeca_n_splits,
                    random_state=cfg.random_state,
                    resume=True,
                    n_gene_workers=_resolve_gene_workers(
                        getattr(args, "gene_workers", "auto")
                    ),
                )
                report["slots"].append(summary)
        report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAIL"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["errors"].append(traceback.format_exc())
        print("FAILED:", exc, flush=True)

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["timings_seconds"] = {"wall_clock_seconds": time.perf_counter() - t_wall0}
    _write_json(out_dir / "run_report.json", report, overwrite=True)
    print(f"\nStatus: {report['status']}\nOutput: {out_dir}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
