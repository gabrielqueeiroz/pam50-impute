#!/usr/bin/env python3
"""
Autotune gene-level parallelism for OriginalRFECA TARGET-WISE.

Runs the same 8 genes × MCAR 20% rep 0 under workers={1,2,4,6,8}, validates
fingerprints against serial (workers=1), chooses the best worker count, and
writes artifacts/parallel_benchmark/.

Does NOT start the reduced full benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Pin before importing numpy/sklearn-heavy stacks in parent too.
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

from bcimpute.config import PAM50_GENES  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.imputation_original.parallel_genes import (  # noqa: E402
    default_imputer_kwargs,
    fingerprints_equal,
    pin_blas_threads,
    run_genes_parallel,
)
from bcimpute.missingness import generate_missingness_sets  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "parallel_benchmark"
RECOMMENDED_PATH = OUT_DIR / "recommended_workers.json"

# 8 consecutive genes not yet finished in the paused MCAR rep0 (sorted order
# after CEP55): keeps autotune independent of existing checkpoints.
DEFAULT_GENES = [
    "CXXC5",
    "EGFR",
    "ERBB2",
    "ESR1",
    "EXO1",
    "FGFR4",
    "FOXA1",
    "FOXC1",
]


class SystemMonitor:
    """Background sampler for CPU / RAM / disk / paging / temperature."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def _loop(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        # Prime cpu_percent
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            try:
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()
                disk = psutil.disk_io_counters()
                temps = None
                try:
                    tmap = psutil.sensors_temperatures()
                    if tmap:
                        vals = []
                        for entries in tmap.values():
                            for e in entries:
                                if e.current is not None:
                                    vals.append(float(e.current))
                        temps = max(vals) if vals else None
                except Exception:  # noqa: BLE001
                    temps = None
                self.samples.append(
                    {
                        "t": time.time(),
                        "cpu_pct": float(psutil.cpu_percent(interval=None)),
                        "ram_used_mb": float(vm.used / (1024 ** 2)),
                        "ram_pct": float(vm.percent),
                        "ram_available_mb": float(vm.available / (1024 ** 2)),
                        "swap_pct": float(swap.percent),
                        "swap_used_mb": float(swap.used / (1024 ** 2)),
                        "disk_read_mb": float((disk.read_bytes if disk else 0) / (1024 ** 2)),
                        "disk_write_mb": float(
                            (disk.write_bytes if disk else 0) / (1024 ** 2)
                        ),
                        "cpu_temp_c": temps,
                        "n_procs": len(psutil.pids()),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "cpu_mean": None,
                "cpu_max": None,
                "ram_peak_mb": None,
                "ram_mean_mb": None,
                "ram_pct_peak": None,
                "ram_pct_mean": None,
                "swap_pct_peak": None,
                "cpu_temp_max": None,
                "throttling_suspected": False,
                "n_samples": 0,
            }
        cpu = [s["cpu_pct"] for s in self.samples]
        ram = [s["ram_used_mb"] for s in self.samples]
        ram_pct = [s["ram_pct"] for s in self.samples]
        swap = [s["swap_pct"] for s in self.samples]
        temps = [s["cpu_temp_c"] for s in self.samples if s["cpu_temp_c"] is not None]
        # crude throttle heuristic: sustained CPU < expected and temp high
        throttling = False
        if temps and max(temps) >= 95:
            throttling = True
        return {
            "cpu_mean": float(np.mean(cpu)),
            "cpu_max": float(np.max(cpu)),
            "ram_peak_mb": float(np.max(ram)),
            "ram_mean_mb": float(np.mean(ram)),
            "ram_pct_peak": float(np.max(ram_pct)),
            "ram_pct_mean": float(np.mean(ram_pct)),
            "swap_pct_peak": float(np.max(swap)),
            "cpu_temp_max": float(max(temps)) if temps else None,
            "throttling_suspected": throttling,
            "n_samples": len(self.samples),
            "ram_available_min_mb": float(
                min(s["ram_available_mb"] for s in self.samples)
            ),
        }


def _process_rss_sum_mb() -> float | None:
    try:
        import psutil

        total = 0.0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                if p.info["name"] and "python" in p.info["name"].lower():
                    total += p.info["memory_info"].rss / (1024 ** 2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return float(total)
    except Exception:  # noqa: BLE001
        return None


def _cpu_affinity_count() -> int | None:
    try:
        import psutil

        aff = psutil.Process().cpu_affinity()
        return len(aff) if aff else None
    except Exception:  # noqa: BLE001
        return None


def choose_workers(rows: list[dict], ram_limit_frac: float = 0.80) -> dict:
    """
    Select best workers:
      1) valid fingerprints
      2) RAM peak < ~80% available (use ram_pct_peak < 80)
      3) no oversubscription flag / throttle
      4) minimal wall time
      5) if 6 only <5% faster than 4, prefer 4
    """
    valid = [r for r in rows if r.get("results_valid")]
    if not valid:
        return {
            "recommended_workers": 1,
            "reason": "No valid parallel configuration; defaulting to serial.",
            "candidates": [],
        }

    eligible = []
    for r in valid:
        if r.get("ram_pct_peak") is not None and r["ram_pct_peak"] > ram_limit_frac * 100:
            r = dict(r)
            r["rejected_reason"] = f"RAM peak {r['ram_pct_peak']:.1f}% > {ram_limit_frac*100:.0f}%"
            continue
        if r.get("throttling_suspected"):
            continue
        if r.get("oversubscription_suspected"):
            continue
        eligible.append(r)

    if not eligible:
        eligible = valid  # fall back to valid even if RAM high

    # Sort by wall time ascending
    eligible = sorted(eligible, key=lambda r: r["wall_seconds"])
    best = eligible[0]

    # Prefer 4 over 6 if marginal
    by_w = {int(r["workers"]): r for r in eligible}
    if 6 in by_w and 4 in by_w:
        w4, w6 = by_w[4], by_w[6]
        gain = (w4["wall_seconds"] - w6["wall_seconds"]) / w4["wall_seconds"]
        if gain < 0.05 and best["workers"] == 6:
            best = w4
            reason_extra = (
                f" workers=6 only {gain*100:.1f}% faster than 4 (<5%); prefer 4."
            )
        else:
            reason_extra = ""
    else:
        reason_extra = ""

    # Justify if 8 worse than 6/4
    notes = []
    if 8 in {int(r["workers"]) for r in rows}:
        r8 = next(r for r in rows if int(r["workers"]) == 8)
        for ref_w in (6, 4):
            ref = next((r for r in rows if int(r["workers"]) == ref_w), None)
            if ref and r8["wall_seconds"] > ref["wall_seconds"]:
                notes.append(
                    f"workers=8 slower than {ref_w} "
                    f"({r8['wall_seconds']:.0f}s vs {ref['wall_seconds']:.0f}s)."
                )

    return {
        "recommended_workers": int(best["workers"]),
        "reason": (
            f"Lowest valid wall time among eligible configs "
            f"(wall={best['wall_seconds']:.1f}s, speedup={best.get('speedup', float('nan')):.2f}x)."
            + reason_extra
        ),
        "notes": notes,
        "best_row": best,
    }


def _plot(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ws = [r["workers"] for r in rows]
    walls = [r["wall_seconds"] / 60.0 for r in rows]
    speed = [r.get("speedup") or float("nan") for r in rows]
    eff = [r.get("efficiency") or float("nan") for r in rows]
    cpu = [r.get("cpu_mean") or float("nan") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(ws, walls, "o-", color="#1f4e79")
    ax.set_xlabel("workers")
    ax.set_ylabel("wall time (min)")
    ax.set_title("Wall time")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ws, speed, "o-", color="#2e7d32")
    ax.plot(ws, ws, "--", color="gray", alpha=0.5, label="ideal")
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs 1")
    ax.set_title("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ws, eff, "o-", color="#c62828")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("workers")
    ax.set_ylabel("efficiency (speedup/workers)")
    ax.set_title("Parallel efficiency")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ws, cpu, "o-", color="#6a1b9a")
    ax.set_xlabel("workers")
    ax.set_ylabel("CPU mean %")
    ax.set_title("CPU utilization")
    ax.grid(True, alpha=0.3)

    fig.suptitle("OriginalRFECA gene-level parallel autotune (MCAR 20% rep0, 8 genes)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    pin_blas_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    p.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
    )
    p.add_argument("--mechanism", default="mcar")
    p.add_argument("--rate", type=float, default=0.20)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()

    genes = list(args.genes)
    for g in genes:
        if g not in PAM50_GENES:
            raise SystemExit(f"Unknown gene: {g}")

    print(
        json.dumps(
            {
                "genes": genes,
                "workers_grid": args.workers,
                "mechanism": args.mechanism,
                "rate": args.rate,
                "replicate": args.replicate,
                "logical_cpus": os.cpu_count(),
                "affinity_count": _cpu_affinity_count(),
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.confirm:
        print("Refusing without --confirm")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort("metabric")
    missing_sets = generate_missingness_sets(
        cohort.X,
        missing_rates=[float(args.rate)],
        n_repetitions=max(5, args.replicate + 1),
        base_seed=42,
        originally_observed_mask=cohort.originally_observed_mask,
        target_cell_policy="originally_observed_only",
        mechanism=args.mechanism,
        seed_scheme="v2",
    )
    item = next(
        it
        for it in missing_sets[float(args.rate)]
        if int(it.replicate) == int(args.replicate)
    )
    kwargs = default_imputer_kwargs()

    import psutil

    vm0 = psutil.virtual_memory()
    host = {
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": float(vm0.total / (1024 ** 2)),
        "ram_available_mb_start": float(vm0.available / (1024 ** 2)),
        "affinity_count": _cpu_affinity_count(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    serial_fps = None
    serial_wall = None
    rows: list[dict] = []

    for nw in args.workers:
        print(f"\n=== workers={nw} ===", flush=True)
        mon = SystemMonitor(interval_s=1.0)
        mon.start()
        cpu_t0 = time.process_time()
        rss_py_before = _process_rss_sum_mb()
        run = run_genes_parallel(
            X=cohort.X,
            mask=item.mask,
            genes=genes,
            n_workers=int(nw),
            seed=int(item.seed),
            mechanism=args.mechanism,
            missing_rate=float(args.rate),
            replicate=int(args.replicate),
            dataset=cohort.name,
            imputer_kwargs=kwargs,
        )
        cpu_time = time.process_time() - cpu_t0
        sys_s = mon.stop()
        rss_py_after = _process_rss_sum_mb()

        ok_all = all(r.get("ok") for r in run.results)
        if nw == 1:
            serial_fps = run.fingerprints
            serial_wall = run.wall_seconds
            valid = ok_all
        else:
            valid = ok_all and fingerprints_equal(serial_fps or {}, run.fingerprints)

        speedup = (
            float(serial_wall / run.wall_seconds)
            if serial_wall and run.wall_seconds > 0
            else (1.0 if nw == 1 else float("nan"))
        )
        efficiency = speedup / nw if nw > 0 else float("nan")
        mean_gene = float(np.mean([r["wall_seconds"] for r in run.results]))
        throughput = len(genes) / (run.wall_seconds / 60.0)

        # Oversubscription heuristic: efficiency << 0.5 at high worker counts
        # with CPU mean already near 100% and wall not improving.
        oversub = False
        if nw >= 6 and efficiency < 0.45:
            oversub = True
        if (
            sys_s.get("cpu_mean") is not None
            and sys_s["cpu_mean"] > 95
            and efficiency < 0.5
            and nw >= 8
        ):
            oversub = True

        row = {
            "workers": int(nw),
            "n_genes": len(genes),
            "wall_seconds": run.wall_seconds,
            "cpu_time_parent_seconds": cpu_time,
            "seconds_per_gene": mean_gene,
            "throughput_genes_per_min": throughput,
            "speedup": speedup,
            "efficiency": efficiency,
            "cpu_mean": sys_s.get("cpu_mean"),
            "cpu_max": sys_s.get("cpu_max"),
            "ram_peak_mb": sys_s.get("ram_peak_mb"),
            "ram_mean_mb": sys_s.get("ram_mean_mb"),
            "ram_pct_peak": sys_s.get("ram_pct_peak"),
            "ram_pct_mean": sys_s.get("ram_pct_mean"),
            "ram_available_min_mb": sys_s.get("ram_available_min_mb"),
            "python_rss_sum_mb_after": rss_py_after,
            "python_rss_sum_mb_before": rss_py_before,
            "mem_per_process_est_mb": (
                None
                if rss_py_after is None
                else float(rss_py_after / max(nw, 1))
            ),
            "swap_pct_peak": sys_s.get("swap_pct_peak"),
            "cpu_temp_max": sys_s.get("cpu_temp_max"),
            "throttling_suspected": sys_s.get("throttling_suspected"),
            "oversubscription_suspected": oversub,
            "n_processes_configured": int(nw),
            "cpu_affinity_count": _cpu_affinity_count(),
            "results_valid": bool(valid),
            "all_genes_ok": bool(ok_all),
            "seed": int(item.seed),
            "errors": [r.get("error") for r in run.results if not r.get("ok")],
            "gene_walls": {r["gene"]: r["wall_seconds"] for r in run.results},
            "fingerprints": run.fingerprints,
        }
        rows.append(row)
        print(
            f"  wall={run.wall_seconds:.1f}s speedup={speedup:.2f}x "
            f"eff={efficiency:.2f} cpu_mean={sys_s.get('cpu_mean')} "
            f"valid={valid}",
            flush=True,
        )
        # Persist incremental progress
        (OUT_DIR / "benchmark_workers_partial.json").write_text(
            json.dumps({"host": host, "rows": rows}, indent=2, default=str),
            encoding="utf-8",
        )

    decision = choose_workers(rows)
    recommended = int(decision["recommended_workers"])

    # Pairwise relative speedups
    by_w = {int(r["workers"]): r for r in rows}
    pairwise = {}
    for a, b in [(1, 2), (2, 4), (4, 6), (6, 8)]:
        if a in by_w and b in by_w and by_w[b]["wall_seconds"] > 0:
            pairwise[f"{a}->{b}"] = float(
                by_w[a]["wall_seconds"] / by_w[b]["wall_seconds"]
            )

    # Updated ETA for reduced benchmark (50 genes × 5 reps × 2 mechs)
    serial_per_gene = (
        by_w[1]["wall_seconds"] / len(genes) if 1 in by_w else 677.0
    )
    n_gene_slots = 50 * 5  # per mechanism
    eta_serial_mcar_h = serial_per_gene * n_gene_slots / 3600.0
    eta_par_mcar_h = eta_serial_mcar_h / max(
        by_w.get(recommended, {}).get("speedup", 1.0), 1e-9
    )
    eta = {
        "seconds_per_gene_serial": serial_per_gene,
        "mcar_20_5reps_serial_hours": eta_serial_mcar_h,
        "mcar_20_5reps_parallel_hours": eta_par_mcar_h,
        "mar_20_5reps_parallel_hours": eta_par_mcar_h,
        "reduced_full_parallel_hours": 2 * eta_par_mcar_h,
        "recommended_workers": recommended,
        "assumed_speedup": by_w.get(recommended, {}).get("speedup"),
    }

    payload = {
        "host": host,
        "protocol": {
            "method": "OriginalRFECA",
            "evaluation_protocol": "repeated_mask_holdout",
            "input_protocol": "target_wise_complete_predictors",
            "parallelism": "gene_level_only",
            "blas_threads": 1,
            "mechanism": args.mechanism,
            "rate": args.rate,
            "replicate": args.replicate,
            "seed": item.seed,
            "genes": genes,
        },
        "rows": rows,
        "pairwise_speedup": pairwise,
        "decision": decision,
        "eta": eta,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # CSV
    csv_cols = [
        "workers",
        "wall_seconds",
        "seconds_per_gene",
        "throughput_genes_per_min",
        "speedup",
        "efficiency",
        "cpu_mean",
        "cpu_max",
        "ram_peak_mb",
        "ram_pct_peak",
        "mem_per_process_est_mb",
        "results_valid",
        "oversubscription_suspected",
        "throttling_suspected",
    ]
    pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in rows]).to_csv(
        OUT_DIR / "benchmark_workers.csv", index=False
    )
    (OUT_DIR / "benchmark_workers.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _plot(rows, OUT_DIR / "benchmark_workers.png")

    # Markdown report
    md = [
        "# Gene-level parallel autotune — OriginalRFECA",
        "",
        f"- Hardware: Ryzen-class ({host['cpu_count_physical']}C/"
        f"{host['cpu_count_logical']}T), RAM {host['ram_total_mb']:.0f} MB",
        f"- Genes ({len(genes)}): {', '.join(genes)}",
        f"- Mask: MCAR 20% rep {args.replicate} seed={item.seed}",
        "- Parallelism: genes only; BLAS threads=1; methodology unchanged",
        "",
        f"## Recommended workers: **{recommended}**",
        "",
        decision["reason"],
        "",
    ]
    if decision.get("notes"):
        md.append("Notes:")
        for n in decision["notes"]:
            md.append(f"- {n}")
        md.append("")

    md += [
        "## Comparative table",
        "",
        "| workers | wall (s) | s/gene | speedup | efficiency | CPU mean % | RAM peak % | valid |",
        "|---------|----------|--------|---------|------------|------------|------------|-------|",
    ]
    for r in rows:
        md.append(
            f"| {r['workers']} | {r['wall_seconds']:.1f} | {r['seconds_per_gene']:.1f} | "
            f"{r['speedup']:.2f} | {r['efficiency']:.2f} | "
            f"{(r['cpu_mean'] if r['cpu_mean'] is not None else float('nan')):.1f} | "
            f"{(r['ram_pct_peak'] if r['ram_pct_peak'] is not None else float('nan')):.1f} | "
            f"{r['results_valid']} |"
        )

    md += [
        "",
        "## Pairwise relative speedup",
        "",
    ]
    for k, v in pairwise.items():
        md.append(f"- {k}: **{v:.2f}×**")

    md += [
        "",
        "## Updated ETA (reduced benchmark)",
        "",
        f"- s/gene (serial): {serial_per_gene:.1f}",
        f"- MCAR 20% × 5 reps @ {recommended} workers: **{eta['mcar_20_5reps_parallel_hours']:.1f} h**",
        f"- MAR 20% × 5 reps @ {recommended} workers: **{eta['mar_20_5reps_parallel_hours']:.1f} h**",
        f"- Reduced full (MCAR+MAR): **{eta['reduced_full_parallel_hours']:.1f} h**",
        "",
        f"(Serial MCAR-only estimate was ~{eta_serial_mcar_h:.1f} h.)",
        "",
        "## Remaining bottleneck",
        "",
        "RFE with linear SVR inside each gene (~97% of gene time). Gene-level "
        "parallelism raises CPU utilization; it does not reduce per-gene RFE cost.",
        "",
    ]
    (OUT_DIR / "benchmark_workers.md").write_text("\n".join(md), encoding="utf-8")

    rec_payload = {
        "recommended_workers": recommended,
        "decision": decision,
        "eta": eta,
        "source": str(OUT_DIR / "benchmark_workers.json"),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    RECOMMENDED_PATH.write_text(
        json.dumps(rec_payload, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "\n".join(md), flush=True)
    print(f"\nWrote {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    # Windows spawn requires freeze_support for some contexts
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
