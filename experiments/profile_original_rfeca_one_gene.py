#!/usr/bin/env python3
"""
Profile OriginalRFECA TARGET-WISE for a single gene.

Produces:
  - stage timers (correlation / RFE / SVR / OOF / final refit)
  - cProfile top functions (tottime + cumtime)
  - JSON + markdown report under artifacts/profiling/
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from bcimpute.config import PAM50_GENES, apply_original_rfeca_only, full_benchmark_config  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.imputers import build_imputers  # noqa: E402
from bcimpute.missingness import generate_missingness_sets  # noqa: E402
from bcimpute import imputation_original as io_pkg  # noqa: E402
from bcimpute.imputation_original import base as base_mod  # noqa: E402
from bcimpute.imputation_original import correlation as corr_mod  # noqa: E402
from bcimpute.imputation_original import selection as sel_mod  # noqa: E402
from bcimpute.imputation_original import svr_model as svr_mod  # noqa: E402


class StageTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._stack: list[tuple[str, float]] = []

    def start(self, name: str) -> None:
        self._stack.append((name, time.perf_counter()))

    def stop(self, name: str | None = None) -> None:
        if not self._stack:
            return
        n, t0 = self._stack.pop()
        if name is not None and name != n:
            # mismatch — still attribute to popped name
            pass
        dt = time.perf_counter() - t0
        self.totals[n] += dt
        self.counts[n] += 1

    def summary(self) -> dict:
        total = sum(self.totals.values()) or 1.0
        rows = []
        for k in sorted(self.totals, key=lambda x: -self.totals[x]):
            rows.append(
                {
                    "stage": k,
                    "seconds": self.totals[k],
                    "calls": self.counts[k],
                    "pct_of_instrumented": 100.0 * self.totals[k] / total,
                    "ms_per_call": 1000.0 * self.totals[k] / max(self.counts[k], 1),
                }
            )
        return {
            "instrumented_seconds": sum(self.totals.values()),
            "stages": rows,
        }


TIMERS = StageTimer()


def _install_hooks() -> None:
    _rank = base_mod.BaseOriginalCorrelationImputer._rank_on_rows
    _select = base_mod.BaseOriginalCorrelationImputer._select_on_train
    _oof = base_mod.BaseOriginalCorrelationImputer._oof_for_prefix_len
    _fit_gene = base_mod.BaseOriginalCorrelationImputer._fit_one_gene
    _pearson = corr_mod.pearson_abs_ranking
    _rfe = sel_mod.select_features_rfe
    _make_pipe = svr_mod.make_svr_pipeline
    _fit_final = svr_mod.fit_final_svr

    def rank_wrap(self, *a, **kw):
        TIMERS.start("correlation_rank")
        try:
            return _rank(self, *a, **kw)
        finally:
            TIMERS.stop("correlation_rank")

    def select_wrap(self, *a, **kw):
        TIMERS.start("rfe_select")
        try:
            return _select(self, *a, **kw)
        finally:
            TIMERS.stop("rfe_select")

    def oof_wrap(self, *a, **kw):
        TIMERS.start("oof_prefix_eval")
        try:
            return _oof(self, *a, **kw)
        finally:
            TIMERS.stop("oof_prefix_eval")

    def fit_gene_wrap(self, *a, **kw):
        TIMERS.start("fit_one_gene_total")
        try:
            return _fit_gene(self, *a, **kw)
        finally:
            TIMERS.stop("fit_one_gene_total")

    def pearson_wrap(*a, **kw):
        TIMERS.start("pearson_abs_ranking")
        try:
            return _pearson(*a, **kw)
        finally:
            TIMERS.stop("pearson_abs_ranking")

    def rfe_wrap(*a, **kw):
        TIMERS.start("sklearn_RFE_fit")
        try:
            return _rfe(*a, **kw)
        finally:
            TIMERS.stop("sklearn_RFE_fit")

    def make_pipe_wrap(*a, **kw):
        pipe = _make_pipe(*a, **kw)
        orig_fit = pipe.fit
        orig_predict = pipe.predict

        def fit_w(*fa, **fkw):
            TIMERS.start("svr_pipeline_fit")
            try:
                return orig_fit(*fa, **fkw)
            finally:
                TIMERS.stop("svr_pipeline_fit")

        def pred_w(*pa, **pkw):
            TIMERS.start("svr_pipeline_predict")
            try:
                return orig_predict(*pa, **pkw)
            finally:
                TIMERS.stop("svr_pipeline_predict")

        pipe.fit = fit_w  # type: ignore[method-assign]
        pipe.predict = pred_w  # type: ignore[method-assign]
        return pipe

    def fit_final_wrap(*a, **kw):
        TIMERS.start("final_svr_refit")
        try:
            return _fit_final(*a, **kw)
        finally:
            TIMERS.stop("final_svr_refit")

    base_mod.BaseOriginalCorrelationImputer._rank_on_rows = rank_wrap  # type: ignore[method-assign]
    base_mod.BaseOriginalCorrelationImputer._select_on_train = select_wrap  # type: ignore[method-assign]
    base_mod.BaseOriginalCorrelationImputer._oof_for_prefix_len = oof_wrap  # type: ignore[method-assign]
    base_mod.BaseOriginalCorrelationImputer._fit_one_gene = fit_gene_wrap  # type: ignore[method-assign]
    corr_mod.pearson_abs_ranking = pearson_wrap  # type: ignore[assignment]
    sel_mod.select_features_rfe = rfe_wrap  # type: ignore[assignment]
    # selection.select_features calls select_features_rfe by name in module
    sel_mod.select_features = (  # type: ignore[assignment]
        lambda X, y, kind, **kw: (
            rfe_wrap(X, y)
            if kind == "RFE"
            else sel_mod.select_features_rfa(X, y, **kw)
        )
    )
    # base imports select_features at module load — patch there too
    base_mod.select_features = sel_mod.select_features  # type: ignore[attr-defined]
    svr_mod.make_svr_pipeline = make_pipe_wrap  # type: ignore[assignment]
    svr_mod.fit_final_svr = fit_final_wrap  # type: ignore[assignment]
    base_mod.make_svr_pipeline = make_pipe_wrap  # type: ignore[attr-defined]
    base_mod.fit_final_svr = fit_final_wrap  # type: ignore[attr-defined]


def _cprofile_report(pr: cProfile.Profile, top_n: int = 40) -> dict:
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats("tottime")
    stats.print_stats(top_n)
    by_tot = s.getvalue()

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("cumtime").print_stats(top_n)
    by_cum = s2.getvalue()

    # Structured top entries
    entries_tot = []
    entries_cum = []
    for sort_key, bucket in (("tottime", entries_tot), ("cumtime", entries_cum)):
        st = pstats.Stats(pr).sort_stats(sort_key)
        for func, (cc, nc, tt, ct, callers) in list(st.stats.items())[:0]:
            pass
        # Use internal ordering
        items = sorted(
            st.stats.items(),
            key=lambda kv: kv[1][2] if sort_key == "tottime" else kv[1][3],
            reverse=True,
        )[:top_n]
        for (filename, line, funcname), (cc, nc, tt, ct, _callers) in items:
            bucket.append(
                {
                    "file": Path(filename).name,
                    "line": line,
                    "func": funcname,
                    "ncalls": nc,
                    "tottime": tt,
                    "cumtime": ct,
                }
            )
    return {
        "top_tottime_text": by_tot,
        "top_cumtime_text": by_cum,
        "top_tottime": entries_tot,
        "top_cumtime": entries_cum,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gene", default="ACTR3B")
    p.add_argument("--mechanism", choices=["mcar", "mar"], default="mcar")
    p.add_argument("--rate", type=float, default=0.20)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--cohort", default="metabric")
    args = p.parse_args()

    if args.gene not in PAM50_GENES:
        raise SystemExit(f"Unknown gene {args.gene}; not in PAM50_GENES")

    out_dir = ROOT / "artifacts" / "profiling" / (
        f"{args.cohort}_{args.mechanism}_r{args.rate:.2f}_rep{args.replicate}_{args.gene}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _install_hooks()

    cfg = apply_original_rfeca_only(
        full_benchmark_config(args.cohort, missingness_mechanism=args.mechanism)
    )
    cfg.n_jobs = 1
    cfg.missing_rates = [float(args.rate)]
    cfg.n_repetitions = max(5, args.replicate + 1)
    cfg.original_rfeca_use_scaler = False
    cfg.original_rfeca_max_candidates = 49
    cfg.original_rfeca_n_splits = 5
    cfg.original_rfeca_validation = "kfold"

    cohort = load_cohort(args.cohort)
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
    item = next(
        it
        for it in missing_sets[float(args.rate)]
        if int(it.replicate) == int(args.replicate)
    )
    imp = build_imputers(cfg)["OriginalRFECA"]
    imp.target_genes = [args.gene]
    imp.set_run_context(
        dataset=cohort.name,
        mechanism=args.mechanism,
        missing_rate=float(args.rate),
        replicate=int(args.replicate),
        seed=item.seed,
    )
    # No checkpoint: wrapped SVR fit/predict closures are not picklable.
    imp.checkpoint_dir = None

    print(
        f"Profiling gene={args.gene} mech={args.mechanism} rate={args.rate} "
        f"rep={args.replicate} seed={item.seed}",
        flush=True,
    )

    pr = cProfile.Profile()
    wall0 = time.perf_counter()
    pr.enable()
    imp.set_target_wise_context(cohort.X, item.mask, for_transform=False)
    imp.fit(cohort.X)
    imp.set_target_wise_context(cohort.X, item.mask, for_transform=True)
    _ = imp.transform(cohort.X)
    pr.disable()
    wall = time.perf_counter() - wall0

    stage = TIMERS.summary()
    cprof = _cprofile_report(pr, top_n=40)

    # Save raw cProfile
    prof_path = out_dir / "cprofile.pstats"
    pr.dump_stats(str(prof_path))

    audit = imp.get_audit_dict()
    gene_rec = next((g for g in audit.get("genes", []) if g["gene"] == args.gene), {})

    # Nested accounting note: oof_prefix_eval includes rfe+svr+corr inside
    nested_note = (
        "oof_prefix_eval encompasses per-fold correlation_rank, rfe_select/sklearn_RFE_fit, "
        "and svr_pipeline_fit/predict. fit_one_gene_total encompasses everything including "
        "final refit. Percentages of leaf stages (pearson, RFE, SVR) are the best cost drivers."
    )

    leaf_keys = [
        "pearson_abs_ranking",
        "sklearn_RFE_fit",
        "svr_pipeline_fit",
        "svr_pipeline_predict",
        "final_svr_refit",
    ]
    leaf_total = sum(TIMERS.totals.get(k, 0.0) for k in leaf_keys) or 1.0
    leaf_rows = []
    for k in leaf_keys:
        sec = TIMERS.totals.get(k, 0.0)
        leaf_rows.append(
            {
                "stage": k,
                "seconds": sec,
                "calls": TIMERS.counts.get(k, 0),
                "pct_of_leaf_total": 100.0 * sec / leaf_total,
                "ms_per_call": 1000.0 * sec / max(TIMERS.counts.get(k, 1), 1),
            }
        )
    leaf_rows.sort(key=lambda r: -r["seconds"])

    payload = {
        "gene": args.gene,
        "mechanism": args.mechanism,
        "rate": args.rate,
        "replicate": args.replicate,
        "seed": item.seed,
        "wall_seconds": wall,
        "n_rfe_fits": audit.get("n_rfe_fits_total"),
        "n_svr_fits": audit.get("n_svr_fits_total"),
        "winning_prefix_len": gene_rec.get("winning_prefix_len"),
        "n_predictors_selected": gene_rec.get("n_predictors_selected"),
        "selection_seconds_audit": gene_rec.get("selection_seconds"),
        "final_fit_seconds_audit": gene_rec.get("final_fit_seconds"),
        "nested_accounting_note": nested_note,
        "stages_all": stage,
        "stages_leaf": {"leaf_seconds": leaf_total, "stages": leaf_rows},
        "cprofile_top_tottime": cprof["top_tottime"],
        "cprofile_top_cumtime": cprof["top_cumtime"],
        "blas_threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "concurrent_mcar_paused": True,
    }

    (out_dir / "profile_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "cprofile_tottime.txt").write_text(cprof["top_tottime_text"], encoding="utf-8")
    (out_dir / "cprofile_cumtime.txt").write_text(cprof["top_cumtime_text"], encoding="utf-8")

    md = [
        f"# Profile: OriginalRFECA / `{args.gene}`",
        "",
        f"- mechanism={args.mechanism} rate={args.rate} rep={args.replicate} seed={item.seed}",
        f"- wall_seconds={wall:.1f}",
        f"- winning_prefix_len={gene_rec.get('winning_prefix_len')} "
        f"n_predictors={gene_rec.get('n_predictors_selected')}",
        f"- n_rfe_fits={audit.get('n_rfe_fits_total')} n_svr_fits={audit.get('n_svr_fits_total')}",
        "",
        "## Leaf stage breakdown (non-nested)",
        "",
        "| Stage | Seconds | Calls | % leaf | ms/call |",
        "|-------|---------|-------|--------|---------|",
    ]
    for r in leaf_rows:
        md.append(
            f"| {r['stage']} | {r['seconds']:.2f} | {r['calls']} | "
            f"{r['pct_of_leaf_total']:.1f}% | {r['ms_per_call']:.1f} |"
        )
    md += [
        "",
        f"*Leaf total = {leaf_total:.1f}s (sum of instrumented leaf stages).*",
        "",
        f"Note: {nested_note}",
        "",
        "## All instrumented stages (includes nesting)",
        "",
        "| Stage | Seconds | Calls | % instrumented |",
        "|-------|---------|-------|----------------|",
    ]
    for r in stage["stages"]:
        md.append(
            f"| {r['stage']} | {r['seconds']:.2f} | {r['calls']} | "
            f"{r['pct_of_instrumented']:.1f}% |"
        )
    md += [
        "",
        "## cProfile top 15 by tottime (exclusive)",
        "",
        "| Func | File | Calls | tottime | cumtime |",
        "|------|------|-------|---------|---------|",
    ]
    for r in cprof["top_tottime"][:15]:
        md.append(
            f"| `{r['func']}` | {r['file']}:{r['line']} | {r['ncalls']} | "
            f"{r['tottime']:.2f} | {r['cumtime']:.2f} |"
        )
    md += [
        "",
        f"Artifacts: `{out_dir}`",
        "",
    ]
    (out_dir / "PROFILE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md), flush=True)
    print(f"\nWrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
