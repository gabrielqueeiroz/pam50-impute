"""Gene-level parallelism for TARGET-WISE OriginalRFECA (methodology-preserving).

Only the outer gene loop is parallelized. Each worker runs the full single-gene
pipeline with OMP/MKL/OPENBLAS/NUMEXPR pinned to 1 thread. RFE, folds, prefixes,
SVR, masks, and seeds are unchanged.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

# Module-level state for ProcessPoolExecutor workers (spawn-safe).
_WORKER_STATE: dict[str, Any] | None = None


class EarlyAbort(Exception):
    """Raised when a parallel gene run is stopped early by an external guard."""

    def __init__(
        self,
        reason: str,
        *,
        partial_results: list[dict[str, Any]] | None = None,
        elapsed_seconds: float | None = None,
    ):
        super().__init__(reason)
        self.reason = reason
        self.partial_results = list(partial_results or [])
        self.elapsed_seconds = elapsed_seconds


def pin_blas_threads(n: int = 1) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(int(n))


def _worker_init(
    X_values: np.ndarray,
    columns: list[str],
    index: list[str],
    mask: np.ndarray,
    *,
    seed: int,
    mechanism: str,
    missing_rate: float,
    replicate: int,
    dataset: str,
    imputer_kwargs: dict[str, Any],
    checkpoint_dir: str | None = None,
) -> None:
    """Initializer: pin threads and cache shared arrays once per process."""
    pin_blas_threads(1)
    global _WORKER_STATE
    _WORKER_STATE = {
        "X_values": np.asarray(X_values, dtype=float),
        "columns": list(columns),
        "index": list(index),
        "mask": np.asarray(mask, dtype=bool),
        "seed": int(seed),
        "mechanism": str(mechanism),
        "missing_rate": float(missing_rate),
        "replicate": int(replicate),
        "dataset": str(dataset),
        "imputer_kwargs": dict(imputer_kwargs),
        "checkpoint_dir": checkpoint_dir,
    }


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:  # noqa: BLE001
        return None


def fit_predict_one_gene(gene: str) -> dict[str, Any]:
    """Fit+transform+metrics for a single target gene (worker entrypoint)."""
    pin_blas_threads(1)
    if _WORKER_STATE is None:
        raise RuntimeError("Worker state not initialized.")

    from pathlib import Path

    import joblib

    from bcimpute.imputation_original.rfeca import OriginalRFECAImputer

    st = _WORKER_STATE
    t0 = time.perf_counter()
    t_start_unix = time.time()
    rss0 = _rss_mb()
    try:
        X = pd.DataFrame(
            st["X_values"],
            index=st["index"],
            columns=st["columns"],
        )
        mask = st["mask"]
        kwargs = dict(st["imputer_kwargs"])
        kwargs["target_genes"] = [gene]
        # Per-gene checkpoint dir (optional): avoid cross-process joblib races.
        ckpt_root = st.get("checkpoint_dir")
        gene_ckpt = None
        if ckpt_root:
            gene_ckpt = Path(ckpt_root) / "genes"
            gene_ckpt.mkdir(parents=True, exist_ok=True)
            kwargs["checkpoint_dir"] = str(Path(ckpt_root) / f"worker_{gene}")
        else:
            kwargs["checkpoint_dir"] = None
        imp = OriginalRFECAImputer(**kwargs)
        imp.set_run_context(
            dataset=st["dataset"],
            mechanism=st["mechanism"],
            missing_rate=st["missing_rate"],
            replicate=st["replicate"],
            seed=st["seed"],
        )
        imp.set_target_wise_context(X, mask, for_transform=False)
        imp.fit(X)
        imp.set_target_wise_context(X, mask, for_transform=True)
        X_imp = imp.transform(X)

        j = list(X.columns).index(gene)
        m = mask[:, j]
        tv = X.to_numpy(dtype=float)[m, j]
        pv = X_imp.to_numpy(dtype=float)[m, j]
        rmse = float(np.sqrt(np.mean((pv - tv) ** 2))) if tv.size else float("nan")
        mae = float(np.mean(np.abs(pv - tv))) if tv.size else float("nan")

        ga = imp.get_audit_dict()
        rec = next((g for g in ga.get("genes", []) if g["gene"] == gene), {})
        packed = imp._gene_models_.get(gene)
        predictors = list(packed[1]) if packed is not None else []

        # Persist model artifact for parent merge (atomic replace).
        if ckpt_root and packed is not None:
            models_dir = Path(ckpt_root) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            tmp = models_dir / f"{gene}.joblib.tmp"
            final = models_dir / f"{gene}.joblib"
            joblib.dump(packed, tmp)
            tmp.replace(final)
            # Also write gene audit JSON if not already via imputer checkpoint
            # Mirror gene audit JSON into the shared checkpoint tree.
            audit_path = Path(ckpt_root) / "genes" / f"{gene}.json"
            if not audit_path.exists() and rec:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_a = audit_path.with_suffix(".json.tmp")
                tmp_a.write_text(
                    __import__("json").dumps(rec, indent=2, default=str),
                    encoding="utf-8",
                )
                tmp_a.replace(audit_path)

        wall = time.perf_counter() - t0
        rss1 = _rss_mb()
        return {
            "gene": gene,
            "ok": True,
            "error": None,
            "rmse": rmse,
            "mae": mae,
            "n_masked": int(m.sum()),
            "n_observed_train": int(rec.get("n_observed", int((~m).sum()))),
            "winning_prefix_len": int(rec.get("winning_prefix_len", 0)),
            "n_predictors_selected": int(rec.get("n_predictors_selected", 0)),
            "winning_predictors": list(rec.get("winning_predictors", [])),
            "winning_prefix_genes": list(rec.get("winning_prefix_genes", [])),
            "status": str(rec.get("status", "unknown")),
            "selection_seconds": float(rec.get("selection_seconds", 0.0)),
            "final_fit_seconds": float(rec.get("final_fit_seconds", 0.0)),
            "wall_seconds": wall,
            "t_start_unix": t_start_unix,
            "t_end_unix": time.time(),
            "seed": int(st["seed"]),
            "svr_coverage": ga.get("svr_coverage"),
            "fallback_rate": ga.get("fallback_rate"),
            "n_rfe_fits": ga.get("n_rfe_fits_total"),
            "n_svr_fits": ga.get("n_svr_fits_total"),
            "fallback_events": ga.get("fallback_events", []),
            "audit_gene": rec,
            "model_predictors": predictors,
            "rss_mb_before": rss0,
            "rss_mb_after": rss1,
            "fingerprint": {
                "gene": gene,
                "rmse": round(rmse, 12) if np.isfinite(rmse) else None,
                "mae": round(mae, 12) if np.isfinite(mae) else None,
                "winning_prefix_len": int(rec.get("winning_prefix_len", 0)),
                "n_predictors_selected": int(rec.get("n_predictors_selected", 0)),
                "winning_predictors": tuple(rec.get("winning_predictors", [])),
                "seed": int(st["seed"]),
                "status": str(rec.get("status", "")),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "gene": gene,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - t0,
            "t_start_unix": t_start_unix,
            "t_end_unix": time.time(),
            "rss_mb_before": rss0,
            "rss_mb_after": _rss_mb(),
            "fingerprint": None,
        }


@dataclass
class ParallelGeneRunResult:
    n_workers: int
    genes: list[str]
    results: list[dict[str, Any]]
    wall_seconds: float
    fingerprints: dict[str, dict]
    early_aborted: bool = False
    abort_reason: str | None = None
    n_completed: int = 0


def _terminate_pool_processes(ex: ProcessPoolExecutor) -> None:
    """Best-effort kill of still-running pool worker processes (Windows-safe)."""
    procs = []
    try:
        # CPython private map: {pid: multiprocessing.Process}
        procs = list(getattr(ex, "_processes", {}) or {}).values()
    except Exception:  # noqa: BLE001
        procs = []
    for proc in procs:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    t_join = time.time() + 3.0
    for proc in procs:
        try:
            remaining = max(0.0, t_join - time.time())
            proc.join(timeout=remaining)
            if proc.is_alive():
                proc.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        try:
            ex.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def run_genes_parallel(
    *,
    X: pd.DataFrame,
    mask: np.ndarray,
    genes: list[str],
    n_workers: int,
    seed: int,
    mechanism: str,
    missing_rate: float,
    replicate: int,
    dataset: str,
    imputer_kwargs: dict[str, Any],
    checkpoint_dir: str | None = None,
    should_abort: Callable[[], str | None] | None = None,
    on_gene_done: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
    poll_interval_s: float = 1.0,
) -> ParallelGeneRunResult:
    """
    Run independent gene jobs with ``n_workers`` processes.

    ``n_workers=1`` uses an in-process loop (no pool) for a clean serial baseline.

    Optional ``should_abort`` is polled between gene completions (and every
    ``poll_interval_s`` while waiting). If it returns a non-empty reason string,
    pending work is cancelled and :class:`EarlyAbort` is raised.
    """
    pin_blas_threads(1)
    genes = list(genes)
    n_workers = max(1, int(n_workers))

    initargs = (
        X.to_numpy(dtype=float),
        [str(c) for c in X.columns],
        [str(i) for i in X.index],
        np.asarray(mask, dtype=bool),
    )
    initkwargs = dict(
        seed=seed,
        mechanism=mechanism,
        missing_rate=missing_rate,
        replicate=replicate,
        dataset=dataset,
        imputer_kwargs=imputer_kwargs,
        checkpoint_dir=checkpoint_dir,
    )

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    def _raise_if_abort(partial: list[dict[str, Any]]) -> None:
        if should_abort is None:
            return
        reason = should_abort()
        if reason:
            raise EarlyAbort(
                reason,
                partial_results=partial,
                elapsed_seconds=time.perf_counter() - t0,
            )

    if n_workers == 1:
        _worker_init(*initargs, **initkwargs)
        for g in genes:
            _raise_if_abort(results)
            rec = fit_predict_one_gene(g)
            results.append(rec)
            if on_gene_done is not None:
                on_gene_done(rec, list(results))
            _raise_if_abort(results)
    else:
        init_positional = initargs + (
            seed,
            mechanism,
            missing_rate,
            replicate,
            dataset,
            imputer_kwargs,
            checkpoint_dir,
        )
        by_gene: dict[str, dict] = {}
        ex = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init_positional,
            initargs=init_positional,
        )
        aborted = False
        try:
            futs = {ex.submit(fit_predict_one_gene, g): g for g in genes}
            pending = set(futs)
            while pending:
                _raise_if_abort(list(by_gene.values()))
                done, pending = wait(
                    pending,
                    timeout=float(poll_interval_s),
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    gene = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = {
                            "gene": gene,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "wall_seconds": None,
                            "fingerprint": None,
                        }
                    by_gene[gene] = rec
                    partial = [by_gene[g] for g in genes if g in by_gene]
                    if on_gene_done is not None:
                        on_gene_done(rec, partial)
                    _raise_if_abort(partial)
            results = [by_gene[g] for g in genes]
        except EarlyAbort:
            aborted = True
            _terminate_pool_processes(ex)
            raise
        finally:
            try:
                ex.shutdown(wait=not aborted, cancel_futures=True)
            except TypeError:
                try:
                    ex.shutdown(wait=not aborted)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass

    wall = time.perf_counter() - t0
    fps = {
        r["gene"]: r["fingerprint"]
        for r in results
        if r.get("ok") and r.get("fingerprint") is not None
    }
    return ParallelGeneRunResult(
        n_workers=n_workers,
        genes=genes,
        results=results,
        wall_seconds=wall,
        fingerprints=fps,
        n_completed=len(results),
    )


def _worker_init_positional(
    X_values,
    columns,
    index,
    mask,
    seed,
    mechanism,
    missing_rate,
    replicate,
    dataset,
    imputer_kwargs,
    checkpoint_dir=None,
) -> None:
    _worker_init(
        X_values,
        columns,
        index,
        mask,
        seed=seed,
        mechanism=mechanism,
        missing_rate=missing_rate,
        replicate=replicate,
        dataset=dataset,
        imputer_kwargs=imputer_kwargs,
        checkpoint_dir=checkpoint_dir,
    )


def fingerprints_equal(a: dict, b: dict, *, rtol: float = 0.0, atol: float = 1e-10) -> bool:
    """Compare gene fingerprints for parallel-vs-serial validation."""
    if a.keys() != b.keys():
        return False
    for gene in a:
        fa, fb = a[gene], b[gene]
        if fa is None or fb is None:
            return False
        if fa["winning_prefix_len"] != fb["winning_prefix_len"]:
            return False
        if fa["n_predictors_selected"] != fb["n_predictors_selected"]:
            return False
        if tuple(fa["winning_predictors"]) != tuple(fb["winning_predictors"]):
            return False
        if fa["seed"] != fb["seed"]:
            return False
        if fa["status"] != fb["status"]:
            return False
        for key in ("rmse", "mae"):
            va, vb = fa.get(key), fb.get(key)
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                return False
            if not np.isclose(va, vb, rtol=rtol, atol=atol):
                return False
    return True


def default_imputer_kwargs() -> dict[str, Any]:
    from bcimpute.config import PAM50_GENES

    return {
        "validation_strategy": "kfold",
        "n_splits": 5,
        "random_state": 42,
        "kernel": "linear",
        "C": 1.0,
        "epsilon": 0.1,
        "use_scaler": False,
        "min_train_samples": 10,
        "max_candidates": 49,
        "feature_names": list(PAM50_GENES),
        "input_protocol": "target_wise_complete_predictors",
        "selection_protocol": "leakage_safe",
        "candidate_rule": "full_matrix",
    }
