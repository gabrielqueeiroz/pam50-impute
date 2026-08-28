#!/usr/bin/env python3
"""
Extended gene-level parallel autotune for OriginalRFECA (16 genes × 8/10/12/16).

Isolated under artifacts/parallel_benchmark_extended/ — does NOT touch the
paused MCAR tranche checkpoints and does NOT resume the full benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
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
    EarlyAbort,
    default_imputer_kwargs,
    fingerprints_equal,
    pin_blas_threads,
    run_genes_parallel,
)
from bcimpute.missingness import generate_missingness_sets  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "parallel_benchmark_extended"
MAIN_REC = ROOT / "artifacts" / "parallel_benchmark" / "recommended_workers.json"
MCAR_PROGRESS = (
    ROOT
    / "artifacts"
    / "original_rfeca_reduced_metabric"
    / "mcar"
    / "rate_0.20"
    / "rep_0"
    / "checkpoint"
    / "progress.jsonl"
)

# Sorted PAM50 after the 13 genes already checkpointed in MCAR rep0.
# Mix includes previously profiled genes (CXXC5–FOXC1) plus 8 more.
DEFAULT_16 = [
    "CXXC5",
    "EGFR",
    "ERBB2",
    "ESR1",
    "EXO1",
    "FGFR4",
    "FOXA1",
    "FOXC1",
    "GPR160",
    "GRB7",
    "KIF2C",
    "KRT14",
    "KRT17",
    "KRT5",
    "MAPT",
    "MDM2",
]

FP_ATOL = 1e-10
FP_RTOL = 0.0
MIN_GAIN_VS_8 = 0.05  # 5% wall reduction required to beat baseline 8
EARLY_STOP_WINDOW_S = 120.0  # continuous window before abort
SWAP_PCT_ABORT = 5.0
SWAP_USED_DELTA_MB = 128.0
MIN_GENES_FOR_WALL_PROJECTION = 3
# High CPU alone must NEVER trigger early stop.
CPU_BUSY_FOR_THROTTLE = 70.0


def _sensor_temp_limits() -> tuple[float | None, float | None, float | None]:
    """Return (current_max, high_limit, critical_limit) from psutil sensors."""
    try:
        import psutil

        tmap = psutil.sensors_temperatures()
    except Exception:  # noqa: BLE001
        return None, None, None
    if not tmap:
        return None, None, None
    currents: list[float] = []
    highs: list[float] = []
    crits: list[float] = []
    for entries in tmap.values():
        for e in entries:
            if e.current is not None:
                currents.append(float(e.current))
            if getattr(e, "high", None) is not None:
                highs.append(float(e.high))
            if getattr(e, "critical", None) is not None:
                crits.append(float(e.critical))
    return (
        max(currents) if currents else None,
        min(highs) if highs else None,
        min(crits) if crits else None,
    )


class EarlyStopGuard:
    """Abort a config when a stop condition holds continuously for ≥2 minutes.

    Explicit non-trigger: CPU at 90%/100% alone never aborts.
    """

    def __init__(
        self,
        *,
        n_genes: int,
        n_workers: int,
        best_wall_seconds: float | None = None,
        expected_workers: int | None = None,
        window_s: float = EARLY_STOP_WINDOW_S,
    ):
        self.n_genes = int(n_genes)
        self.n_workers = int(n_workers)
        self.expected_workers = int(
            expected_workers if expected_workers is not None else n_workers
        )
        self.best_wall_seconds = (
            float(best_wall_seconds) if best_wall_seconds is not None else None
        )
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._condition_since: dict[str, float] = {}
        self._abort_reason: str | None = None
        self._abort_details: dict = {}
        self.t0_perf = time.perf_counter()
        self.completed: list[dict] = []
        self.projected_wall: float | None = None
        self._swap0_used_mb: float | None = None
        self._swap0_pct: float | None = None
        self._freq_peak_mhz: float | None = None
        self._temp_high: float | None = None
        self._temp_critical: float | None = None
        cur, high, crit = _sensor_temp_limits()
        self._temp_high = high
        self._temp_critical = crit
        # Fallback critical if sensors omit limits (common on Windows).
        if self._temp_critical is None and self._temp_high is not None:
            self._temp_critical = self._temp_high
        if self._temp_critical is None:
            self._temp_critical = 95.0
        self._exception_timestamps: list[float] = []
        self._n_fail = 0
        self._last_flags: dict[str, bool] = {}

    @property
    def abort_reason(self) -> str | None:
        with self._lock:
            return self._abort_reason

    @property
    def abort_details(self) -> dict:
        with self._lock:
            return dict(self._abort_details)

    def note_gene_done(self, result: dict, _partial: list[dict] | None = None) -> None:
        with self._lock:
            self.completed.append(result)
            if not result.get("ok", False):
                self._n_fail += 1
                self._exception_timestamps.append(time.time())
                # Keep last 10 minutes of failures
                cutoff = time.time() - 600.0
                self._exception_timestamps = [
                    t for t in self._exception_timestamps if t >= cutoff
                ]
            self._update_projection_unlocked()

    def _update_projection_unlocked(self) -> None:
        n_done = len(self.completed)
        if n_done < MIN_GENES_FOR_WALL_PROJECTION:
            self.projected_wall = None
            return
        walls = [
            float(r["wall_seconds"])
            for r in self.completed
            if r.get("wall_seconds") is not None and np.isfinite(r["wall_seconds"])
        ]
        if not walls:
            self.projected_wall = None
            return
        mean_gene = float(np.mean(walls))
        elapsed = time.perf_counter() - self.t0_perf
        remaining = max(0, self.n_genes - n_done)
        # Makespan-style remainder: unfinished work / concurrency.
        rem_wall = (remaining * mean_gene) / max(self.n_workers, 1)
        self.projected_wall = float(elapsed + rem_wall)

    def _set_abort(self, reason: str, details: dict) -> None:
        if self._abort_reason is None:
            self._abort_reason = reason
            self._abort_details = details
            print(f"EARLY_STOP: {reason}", flush=True)

    def _track_continuous(self, name: str, active: bool, now: float) -> bool:
        """Return True if ``name`` has been continuously active for ≥ window_s."""
        if active:
            if name not in self._condition_since:
                self._condition_since[name] = now
            return (now - self._condition_since[name]) >= self.window_s
        self._condition_since.pop(name, None)
        return False

    def observe_sample(self, sample: dict) -> str | None:
        """Update hardware-driven conditions from one monitor sample."""
        with self._lock:
            if self._abort_reason is not None:
                return self._abort_reason
            now = float(sample.get("t") or time.time())
            if self._swap0_used_mb is None:
                self._swap0_used_mb = float(sample.get("swap_used_mb") or 0.0)
                self._swap0_pct = float(sample.get("swap_pct") or 0.0)
            freq = sample.get("cpu_freq_mhz")
            if freq is not None:
                f = float(freq)
                self._freq_peak_mhz = (
                    f
                    if self._freq_peak_mhz is None
                    else max(self._freq_peak_mhz, f)
                )

            swap_pct = float(sample.get("swap_pct") or 0.0)
            swap_used = float(sample.get("swap_used_mb") or 0.0)
            swap_delta = swap_used - float(self._swap0_used_mb or 0.0)
            swap_pressure = (swap_pct >= SWAP_PCT_ABORT) or (
                swap_delta >= SWAP_USED_DELTA_MB
            )

            temp = sample.get("cpu_temp_c")
            temp_c = float(temp) if temp is not None else None
            crit = float(self._temp_critical) if self._temp_critical else None
            high = float(self._temp_high) if self._temp_high else None
            temp_critical = bool(
                temp_c is not None and crit is not None and temp_c >= crit
            )

            cpu = float(sample.get("cpu_pct") or 0.0)
            throttle = False
            if temp_c is not None and high is not None and temp_c >= high:
                throttle = True
            if (
                freq is not None
                and self._freq_peak_mhz
                and self._freq_peak_mhz > 0
                and cpu >= CPU_BUSY_FOR_THROTTLE
                and float(freq) < 0.7 * self._freq_peak_mhz
            ):
                throttle = True
            # Confirmed: critical breach also counts as thermal pressure.
            if temp_critical:
                throttle = True

            # Instability: ≥2 failures in the last 5 minutes, or process collapse.
            recent_fail = sum(
                1 for t in self._exception_timestamps if now - t <= 300.0
            )
            n_py = sample.get("n_python_procs")
            elapsed = time.perf_counter() - self.t0_perf
            proc_collapse = bool(
                elapsed >= 60.0
                and n_py is not None
                and int(n_py) < max(2, self.expected_workers // 2)
                and cpu >= CPU_BUSY_FOR_THROTTLE
            )
            instability = (recent_fail >= 2) or proc_collapse

            # Projected wall vs best completed config.
            self._update_projection_unlocked()
            wall_worse = bool(
                self.best_wall_seconds is not None
                and self.projected_wall is not None
                and self.projected_wall > self.best_wall_seconds
            )

            flags = {
                "swap_paging": swap_pressure,
                "thermal_throttling": throttle,
                "temp_critical": temp_critical,
                "projected_wall_worse": wall_worse,
                "process_instability": instability,
            }
            # CPU saturation is recorded but NEVER an abort trigger.
            flags_meta = {
                **flags,
                "cpu_saturated": cpu >= 90.0,
                "cpu_pct": cpu,
            }
            self._last_flags = flags_meta

            for name, active in flags.items():
                if self._track_continuous(name, active, now):
                    details = {
                        "condition": name,
                        "window_s": self.window_s,
                        "sample": {
                            k: sample.get(k)
                            for k in (
                                "cpu_pct",
                                "ram_pct",
                                "swap_pct",
                                "swap_used_mb",
                                "cpu_temp_c",
                                "cpu_freq_mhz",
                                "n_python_procs",
                            )
                        },
                        "projected_wall": self.projected_wall,
                        "best_wall_seconds": self.best_wall_seconds,
                        "n_fail": self._n_fail,
                        "temp_critical_limit_c": self._temp_critical,
                        "temp_high_limit_c": self._temp_high,
                        "flags": flags_meta,
                    }
                    reason = (
                        f"{name} sustained >={self.window_s:.0f}s "
                        f"(projected_wall={self.projected_wall}, "
                        f"best_wall={self.best_wall_seconds}, "
                        f"swap_pct={swap_pct:.1f}, temp={temp_c})"
                    )
                    self._set_abort(reason, details)
                    return self._abort_reason
            return None

    def should_abort(self) -> str | None:
        with self._lock:
            # Re-check projection even between samples (gene completions).
            if self._abort_reason is not None:
                return self._abort_reason
            self._update_projection_unlocked()
            now = time.time()
            wall_worse = bool(
                self.best_wall_seconds is not None
                and self.projected_wall is not None
                and self.projected_wall > self.best_wall_seconds
            )
            if self._track_continuous("projected_wall_worse", wall_worse, now):
                details = {
                    "condition": "projected_wall_worse",
                    "projected_wall": self.projected_wall,
                    "best_wall_seconds": self.best_wall_seconds,
                    "n_completed": len(self.completed),
                }
                reason = (
                    f"projected_wall_worse sustained >={self.window_s:.0f}s "
                    f"(projected={self.projected_wall:.1f}s > "
                    f"best={self.best_wall_seconds:.1f}s)"
                )
                self._set_abort(reason, details)
            return self._abort_reason


class SystemMonitor:
    def __init__(
        self,
        interval_s: float = 1.0,
        early_stop: EarlyStopGuard | None = None,
    ):
        self.interval_s = interval_s
        self.early_stop = early_stop
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
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            try:
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()
                disk = psutil.disk_io_counters()
                temps = None
                temp_high = None
                temp_crit = None
                try:
                    cur, high, crit = _sensor_temp_limits()
                    temps = cur
                    temp_high = high
                    temp_crit = crit
                except Exception:  # noqa: BLE001
                    temps = None
                freq = None
                try:
                    f = psutil.cpu_freq()
                    freq = float(f.current) if f else None
                except Exception:  # noqa: BLE001
                    freq = None
                ctx = None
                try:
                    ctx = int(psutil.cpu_stats().ctx_switches)
                except Exception:  # noqa: BLE001
                    ctx = None
                n_py = 0
                py_rss = 0.0
                try:
                    for p in psutil.process_iter(["name", "memory_info"]):
                        name = (p.info.get("name") or "").lower()
                        if "python" in name:
                            n_py += 1
                            if p.info.get("memory_info"):
                                py_rss += p.info["memory_info"].rss / (1024 ** 2)
                except Exception:  # noqa: BLE001
                    pass
                sample = {
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
                    "cpu_temp_high_c": temp_high,
                    "cpu_temp_critical_c": temp_crit,
                    "cpu_freq_mhz": freq,
                    "ctx_switches": ctx,
                    "n_python_procs": n_py,
                    "python_rss_sum_mb": py_rss,
                }
                self.samples.append(sample)
                if self.early_stop is not None:
                    self.early_stop.observe_sample(sample)
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
                "cpu_temp_mean": None,
                "cpu_freq_mean": None,
                "throttling_suspected": False,
                "n_samples": 0,
                "samples": [],
            }
        cpu = [s["cpu_pct"] for s in self.samples]
        ram = [s["ram_used_mb"] for s in self.samples]
        ram_pct = [s["ram_pct"] for s in self.samples]
        swap = [s["swap_pct"] for s in self.samples]
        temps = [s["cpu_temp_c"] for s in self.samples if s["cpu_temp_c"] is not None]
        freqs = [s["cpu_freq_mhz"] for s in self.samples if s["cpu_freq_mhz"] is not None]
        py_rss = [s["python_rss_sum_mb"] for s in self.samples]
        n_py = [s["n_python_procs"] for s in self.samples]
        throttling = bool(temps and max(temps) >= 95)
        # Also flag if freq collapses while CPU busy (weak heuristic)
        if freqs and len(freqs) > 10 and cpu:
            if np.mean(cpu[-10:]) > 70 and np.mean(freqs[-10:]) < 0.7 * max(freqs):
                throttling = True
        return {
            "cpu_mean": float(np.mean(cpu)),
            "cpu_max": float(np.max(cpu)),
            "ram_peak_mb": float(np.max(ram)),
            "ram_mean_mb": float(np.mean(ram)),
            "ram_pct_peak": float(np.max(ram_pct)),
            "ram_pct_mean": float(np.mean(ram_pct)),
            "ram_available_min_mb": float(min(s["ram_available_mb"] for s in self.samples)),
            "swap_pct_peak": float(np.max(swap)),
            "cpu_temp_max": float(max(temps)) if temps else None,
            "cpu_temp_mean": float(np.mean(temps)) if temps else None,
            "cpu_freq_mean": float(np.mean(freqs)) if freqs else None,
            "python_rss_peak_mb": float(np.max(py_rss)) if py_rss else None,
            "python_rss_mean_mb": float(np.mean(py_rss)) if py_rss else None,
            "n_python_procs_max": int(np.max(n_py)) if n_py else None,
            "throttling_suspected": throttling,
            "n_samples": len(self.samples),
            "samples": self.samples,
        }


def _select_genes() -> list[str]:
    done = set()
    if MCAR_PROGRESS.exists():
        for line in MCAR_PROGRESS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            done.add(json.loads(line)["gene"])
    # Prefer DEFAULT_16 if all unused; otherwise fill from remaining sorted PAM50
    genes = [g for g in DEFAULT_16 if g not in done]
    if len(genes) < 16:
        for g in sorted(PAM50_GENES):
            if g in done or g in genes:
                continue
            genes.append(g)
            if len(genes) >= 16:
                break
    genes = genes[:16]
    if len(genes) < 16:
        raise RuntimeError(f"Only found {len(genes)} unused genes; need 16.")
    for g in genes:
        if g not in PAM50_GENES:
            raise ValueError(g)
    return genes


def _tail_metrics(results: list[dict], n_workers: int) -> dict:
    """Estimate load-imbalance / tail from per-gene start/end timestamps."""
    starts = [r["t_start_unix"] for r in results if r.get("t_start_unix") is not None]
    ends = [r["t_end_unix"] for r in results if r.get("t_end_unix") is not None]
    walls = [r["wall_seconds"] for r in results]
    if not starts or not ends:
        return {
            "slowest_gene_seconds": float(max(walls)) if walls else None,
            "tail_idle_fraction": None,
            "mean_active_workers": None,
            "wait_due_to_slowest_seconds": None,
        }
    t0 = min(starts)
    t1 = max(ends)
    duration = max(t1 - t0, 1e-9)
    # Sample activity every 1s
    grid = np.arange(t0, t1 + 1e-6, 1.0)
    active = []
    for t in grid:
        n = sum(
            1
            for r in results
            if r.get("t_start_unix") is not None
            and r.get("t_end_unix") is not None
            and r["t_start_unix"] <= t <= r["t_end_unix"]
        )
        active.append(n)
    active_arr = np.asarray(active, dtype=float)
    # Tail: last 20% of wall where active < n_workers/2
    cut = int(0.8 * len(active_arr))
    tail = active_arr[cut:] if cut < len(active_arr) else active_arr
    tail_idle_frac = float(np.mean(tail < max(1, n_workers / 2)))
    slowest = float(max(walls))
    # Wait caused by slowest ≈ wall - (sum gene walls / workers) ideal lower bound gap
    ideal = float(sum(walls) / max(n_workers, 1))
    wall = float(t1 - t0)
    return {
        "slowest_gene_seconds": slowest,
        "tail_idle_fraction": tail_idle_frac,
        "mean_active_workers": float(np.mean(active_arr)),
        "wait_due_to_slowest_seconds": max(0.0, wall - ideal),
        "span_seconds": wall,
        "activity_series": {
            "t0_unix": t0,
            "active_per_second": active_arr.tolist(),
        },
    }


def choose_extended(rows: list[dict], baseline_workers: int = 8) -> dict:
    by_w = {int(r["workers"]): r for r in rows if not r.get("early_stopped")}
    base = by_w.get(baseline_workers)
    if base is None:
        return {
            "recommended_workers": baseline_workers,
            "conservative_workers": baseline_workers,
            "reason": (
                f"Baseline {baseline_workers} missing or early-stopped; keep "
                f"{baseline_workers}."
            ),
            "update_main_recommendation": False,
        }

    candidates = []
    for r in rows:
        w = int(r["workers"])
        if w == baseline_workers:
            continue
        if r.get("early_stopped"):
            continue
        if not r.get("results_valid"):
            continue
        gain = (base["wall_seconds"] - r["wall_seconds"]) / base["wall_seconds"]
        if gain < MIN_GAIN_VS_8:
            continue
        if r.get("ram_pct_peak") is not None and r["ram_pct_peak"] > 80:
            continue
        if r.get("swap_pct_peak") is not None and r["swap_pct_peak"] > 5:
            continue
        if r.get("throttling_suspected"):
            continue
        if r.get("oversubscription_severe"):
            continue
        candidates.append({**r, "gain_vs_8": gain})

    if not candidates:
        return {
            "recommended_workers": baseline_workers,
            "conservative_workers": baseline_workers,
            "reason": (
                f"No configuration beat {baseline_workers} workers by ≥"
                f"{MIN_GAIN_VS_8*100:.0f}% with safety constraints."
            ),
            "update_main_recommendation": False,
            "candidates": [],
        }

    # Prefer higher gain; if within 5% wall of each other, prefer fewer workers
    candidates = sorted(
        candidates,
        key=lambda r: (r["wall_seconds"], int(r["workers"])),
    )
    best = candidates[0]
    # Collapse near-ties: among those within 5% of best wall, pick fewest workers
    near = [
        c
        for c in candidates
        if (c["wall_seconds"] - best["wall_seconds"]) / best["wall_seconds"] < 0.05
    ]
    near = sorted(near, key=lambda r: (int(r["workers"]), r["wall_seconds"]))
    chosen = near[0]

    # Conservative night: among candidates with gain>=5% and efficiency>=0.5, fewest workers
    cons = [
        c
        for c in candidates
        if (c.get("efficiency_vs_8") or 0) >= 0.5 or c["workers"] <= 12
    ]
    if not cons:
        cons = candidates
    conservative = sorted(cons, key=lambda r: (int(r["workers"]), r["wall_seconds"]))[0]

    return {
        "recommended_workers": int(chosen["workers"]),
        "conservative_workers": int(conservative["workers"]),
        "reason": (
            f"workers={chosen['workers']} reduces wall by "
            f"{chosen['gain_vs_8']*100:.1f}% vs 8 "
            f"({base['wall_seconds']:.0f}s -> {chosen['wall_seconds']:.0f}s) "
            f"with valid fingerprints and safety checks."
        ),
        "update_main_recommendation": int(chosen["workers"]) != baseline_workers,
        "best_row": chosen,
        "candidates": [
            {
                "workers": int(c["workers"]),
                "gain_vs_8": c["gain_vs_8"],
                "wall_seconds": c["wall_seconds"],
            }
            for c in candidates
        ],
    }


def _plot_summary(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ws = [r["workers"] for r in rows]
    walls = [r["wall_seconds"] / 60 for r in rows]
    speed = [r.get("speedup_vs_8") for r in rows]
    cpu = [r.get("cpu_mean") for r in rows]
    ram = [r.get("ram_pct_peak") for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(ws, walls, "o-", color="#1f4e79")
    axes[0, 0].set_title("Wall time")
    axes[0, 0].set_ylabel("min")
    axes[0, 1].plot(ws, speed, "o-", color="#2e7d32")
    axes[0, 1].axhline(1.0, color="gray", ls="--", alpha=0.5)
    axes[0, 1].set_title("Speedup vs 8 workers")
    axes[1, 0].plot(ws, cpu, "o-", color="#6a1b9a")
    axes[1, 0].set_title("CPU mean %")
    axes[1, 1].plot(ws, ram, "o-", color="#c62828")
    axes[1, 1].axhline(80, color="gray", ls="--", alpha=0.5)
    axes[1, 1].set_title("RAM peak %")
    for ax in axes.ravel():
        ax.set_xlabel("workers")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Extended gene-parallel autotune (16 genes)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_activity(row: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    act = (row.get("tail") or {}).get("activity_series")
    if not act:
        return
    y = act["active_per_second"]
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(x, y, alpha=0.4, color="#1f4e79")
    ax.plot(x, y, color="#1f4e79", lw=1)
    ax.set_xlabel("seconds since first gene start")
    ax.set_ylabel("active workers")
    ax.set_title(f"Worker activity timeline (workers={row['workers']})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_cpu_mem(row: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    samples = row.get("monitor_samples") or []
    if len(samples) < 2:
        return
    t0 = samples[0]["t"]
    xs = [s["t"] - t0 for s in samples]
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(xs, [s["cpu_pct"] for s in samples], color="#6a1b9a", label="CPU %")
    ax1.set_ylabel("CPU %", color="#6a1b9a")
    ax2 = ax1.twinx()
    ax2.plot(xs, [s["ram_pct"] for s in samples], color="#c62828", label="RAM %")
    ax2.set_ylabel("RAM %", color="#c62828")
    ax1.set_xlabel("seconds")
    ax1.set_title(f"CPU / RAM timeline (workers={row['workers']})")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _eta_block(seconds_per_gene: float, speedup_vs_serial_approx: float) -> dict:
    """ETA ranges using ±15% band around point estimate."""
    # serial s/gene from prior autotune ~565; use current / speedup_vs_8 * prior_8_speedup
    # Here we estimate parallel s/gene effective = wall/16 for recommended config
    def band(hours: float) -> dict:
        return {
            "point_hours": hours,
            "low_hours": hours * 0.85,
            "high_hours": hours * 1.15,
        }

    one_rep = seconds_per_gene * 50 / 3600.0
    mcar5 = one_rep * 5
    mar5 = mcar5
    reduced = mcar5 + mar5
    # Expanded: rates 0.10,0.20,0.30 × MCAR+MAR × 5 reps = 6 slots of 5-rep = 30 rep-slots
    # vs reduced 10 rep-slots → ×3
    expanded = reduced * 3
    return {
        "seconds_per_gene_effective": seconds_per_gene,
        "one_replica_50_genes": band(one_rep),
        "mcar_20_5reps": band(mcar5),
        "mar_20_5reps": band(mar5),
        "reduced_full_mcar_mar_20": band(reduced),
        "expanded_10_20_30_mcar_mar_5reps": band(expanded),
        "note": (
            "Effective s/gene = wall_time / 16 for the recommended worker count. "
            "Bands are ±15%. Expanded assumes similar cost at 10%/30% as 20%."
        ),
    }


def main() -> int:
    pin_blas_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--workers", type=int, nargs="+", default=[8, 10, 12, 16])
    p.add_argument("--order-seed", type=int, default=20260731)
    p.add_argument("--cooldown-seconds", type=float, default=20.0)
    p.add_argument("--genes", nargs="+", default=None)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    genes_path = OUT_DIR / "selected_genes.json"
    if args.genes:
        genes = list(args.genes)
    elif genes_path.exists():
        genes = json.loads(genes_path.read_text(encoding="utf-8"))["genes"]
    else:
        genes = _select_genes()
        genes_path.write_text(
            json.dumps(
                {
                    "genes": genes,
                    "n": len(genes),
                    "selection_rule": (
                        "16 unused genes after MCAR rep0 checkpoints; "
                        "DEFAULT_16 prefix then sorted PAM50 fill"
                    ),
                    "mcar_done_excluded": (
                        [
                            json.loads(l)["gene"]
                            for l in MCAR_PROGRESS.read_text(encoding="utf-8").splitlines()
                            if l.strip()
                        ]
                        if MCAR_PROGRESS.exists()
                        else []
                    ),
                    "written_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Randomize config order with fixed seed (baseline 8 still included)
    worker_order = list(args.workers)
    rng = random.Random(int(args.order_seed))
    rng.shuffle(worker_order)

    meta = {
        "genes": genes,
        "workers_grid": args.workers,
        "execution_order": worker_order,
        "order_seed": args.order_seed,
        "cooldown_seconds": args.cooldown_seconds,
        "mechanism": "mcar",
        "rate": 0.20,
        "replicate": 0,
        "fp_atol": FP_ATOL,
        "fp_rtol": FP_RTOL,
        "min_gain_vs_8": MIN_GAIN_VS_8,
        "early_stop_window_s": EARLY_STOP_WINDOW_S,
        "early_stop_conditions": [
            "swap_paging",
            "thermal_throttling",
            "temp_critical",
            "projected_wall_worse",
            "process_instability",
        ],
        "early_stop_excludes": ["cpu_pct_alone"],
        "logical_cpus": os.cpu_count(),
        "isolation": str(OUT_DIR),
        "does_not_touch_mcar_checkpoints": True,
    }
    print(json.dumps(meta, indent=2), flush=True)
    if not args.confirm:
        print("Refusing without --confirm")
        return 0

    import psutil

    vm0 = psutil.virtual_memory()
    host = {
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": float(vm0.total / (1024 ** 2)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    cohort = load_cohort("metabric")
    missing_sets = generate_missingness_sets(
        cohort.X,
        missing_rates=[0.20],
        n_repetitions=5,
        base_seed=42,
        originally_observed_mask=cohort.originally_observed_mask,
        target_cell_policy="originally_observed_only",
        mechanism="mcar",
        seed_scheme="v2",
    )
    item = next(it for it in missing_sets[0.20] if int(it.replicate) == 0)
    kwargs = default_imputer_kwargs()

    rows: list[dict] = []
    baseline_fps = None
    baseline_wall = None
    best_wall_so_far: float | None = None
    errors: list[dict] = []

    for nw in worker_order:
        print(f"\n=== workers={nw} (order position) ===", flush=True)
        # Isolated checkpoint dir per config — never the main MCAR tree
        ckpt = OUT_DIR / "runs" / f"workers_{nw}" / "checkpoint"
        if ckpt.exists():
            # fresh run artifacts under timestamped subdir to avoid resume mixing
            ckpt = (
                OUT_DIR
                / "runs"
                / f"workers_{nw}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
                / "checkpoint"
            )
        ckpt.mkdir(parents=True, exist_ok=True)

        guard = EarlyStopGuard(
            n_genes=len(genes),
            n_workers=int(nw),
            best_wall_seconds=best_wall_so_far,
            expected_workers=int(nw),
        )
        mon = SystemMonitor(interval_s=1.0, early_stop=guard)
        mon.start()
        t_cpu0 = time.process_time()
        early_stopped = False
        abort_reason = None
        abort_details: dict = {}
        partial_results: list[dict] = []
        run = None
        try:
            run = run_genes_parallel(
                X=cohort.X,
                mask=item.mask,
                genes=genes,
                n_workers=int(nw),
                seed=int(item.seed),
                mechanism="mcar",
                missing_rate=0.20,
                replicate=0,
                dataset=cohort.name,
                imputer_kwargs=kwargs,
                checkpoint_dir=str(ckpt),
                should_abort=guard.should_abort,
                on_gene_done=guard.note_gene_done,
                poll_interval_s=1.0,
            )
        except EarlyAbort as exc:
            early_stopped = True
            abort_reason = exc.reason
            abort_details = guard.abort_details
            partial_results = list(exc.partial_results)
            print(f"ABORTED workers={nw}: {abort_reason}", flush=True)
        except Exception as exc:  # noqa: BLE001
            mon.stop()
            err = {
                "workers": nw,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"FAILED workers={nw}: {exc}", flush=True)
            (OUT_DIR / "benchmark_extended_partial.json").write_text(
                json.dumps({"rows": rows, "errors": errors}, indent=2, default=str),
                encoding="utf-8",
            )
            continue
        cpu_parent = time.process_time() - t_cpu0
        sys_s = mon.stop()

        if early_stopped:
            gene_results = partial_results
            wall_seconds = time.perf_counter() - guard.t0_perf
            fps = {
                r["gene"]: r["fingerprint"]
                for r in gene_results
                if r.get("ok") and r.get("fingerprint") is not None
            }
            walls = [
                r["wall_seconds"]
                for r in gene_results
                if r.get("wall_seconds") is not None
            ]
            n_done = len(gene_results)
            row = {
                "workers": int(nw),
                "n_genes": len(genes),
                "n_genes_completed": n_done,
                "wall_seconds": wall_seconds,
                "cpu_time_parent_seconds": cpu_parent,
                "seconds_per_gene_mean": float(np.mean(walls)) if walls else None,
                "seconds_per_gene_median": float(np.median(walls)) if walls else None,
                "seconds_per_gene_min": float(np.min(walls)) if walls else None,
                "seconds_per_gene_max": float(np.max(walls)) if walls else None,
                "throughput_genes_per_min": (
                    n_done / (wall_seconds / 60.0) if wall_seconds > 0 else None
                ),
                "speedup_vs_8": float("nan"),
                "gain_vs_8": float("nan"),
                "efficiency_vs_8": float("nan"),
                "cpu_mean": sys_s.get("cpu_mean"),
                "cpu_max": sys_s.get("cpu_max"),
                "ram_peak_mb": sys_s.get("ram_peak_mb"),
                "ram_mean_mb": sys_s.get("ram_mean_mb"),
                "ram_pct_peak": sys_s.get("ram_pct_peak"),
                "ram_pct_mean": sys_s.get("ram_pct_mean"),
                "python_rss_peak_mb": sys_s.get("python_rss_peak_mb"),
                "python_rss_mean_mb": sys_s.get("python_rss_mean_mb"),
                "mem_per_process_est_mb": (
                    None
                    if not sys_s.get("python_rss_peak_mb")
                    else float(sys_s["python_rss_peak_mb"] / max(nw, 1))
                ),
                "swap_pct_peak": sys_s.get("swap_pct_peak"),
                "cpu_temp_max": sys_s.get("cpu_temp_max"),
                "cpu_temp_mean": sys_s.get("cpu_temp_mean"),
                "cpu_freq_mean": sys_s.get("cpu_freq_mean"),
                "throttling_suspected": sys_s.get("throttling_suspected"),
                "oversubscription_severe": False,
                "n_python_procs_max": sys_s.get("n_python_procs_max"),
                "results_valid": False,
                "validity_deferred": False,
                "all_genes_ok": False,
                "early_stopped": True,
                "abort_reason": abort_reason,
                "abort_details": abort_details,
                "projected_wall_at_abort": guard.projected_wall,
                "best_wall_at_abort": best_wall_so_far,
                "seed": int(item.seed),
                "mask_seed": int(item.seed),
                "checkpoint_dir": str(ckpt),
                "gene_walls": {
                    r["gene"]: r.get("wall_seconds") for r in gene_results
                },
                "fingerprints": fps,
                "tail": {},
                "activity_series": None,
                "monitor_samples": sys_s.get("samples"),
            }
            rows.append(row)
            errors.append(
                {
                    "workers": nw,
                    "error": f"EarlyAbort: {abort_reason}",
                    "early_stopped": True,
                    "abort_details": abort_details,
                }
            )
            print(
                f"  early_stopped wall={wall_seconds:.1f}s "
                f"completed={n_done}/{len(genes)} reason={abort_reason}",
                flush=True,
            )
            (OUT_DIR / "benchmark_extended_partial.json").write_text(
                json.dumps(
                    {"host": host, "meta": meta, "rows": rows, "errors": errors},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            if args.cooldown_seconds > 0:
                time.sleep(float(args.cooldown_seconds))
            continue

        assert run is not None
        ok_all = all(r.get("ok") for r in run.results)
        if int(nw) == 8:
            baseline_fps = run.fingerprints
            baseline_wall = run.wall_seconds

        if baseline_fps is None:
            # Defer validity until baseline 8 is known
            valid = ok_all
            deferred = True
        else:
            valid = ok_all and fingerprints_equal(
                baseline_fps, run.fingerprints, rtol=FP_RTOL, atol=FP_ATOL
            )
            deferred = False

        walls = [r["wall_seconds"] for r in run.results]
        tail = _tail_metrics(run.results, int(nw))
        speedup_vs_8 = (
            float(baseline_wall / run.wall_seconds)
            if baseline_wall and run.wall_seconds > 0
            else (1.0 if nw == 8 else float("nan"))
        )
        gain_vs_8 = (
            float((baseline_wall - run.wall_seconds) / baseline_wall)
            if baseline_wall and baseline_wall > 0
            else (0.0 if nw == 8 else float("nan"))
        )
        # Efficiency relative to 8: speedup_vs_8 / (nw/8)
        efficiency_vs_8 = (
            speedup_vs_8 / (nw / 8.0) if nw > 0 and np.isfinite(speedup_vs_8) else float("nan")
        )
        oversub_severe = bool(
            nw >= 12
            and np.isfinite(efficiency_vs_8)
            and efficiency_vs_8 < 0.45
            and (sys_s.get("cpu_mean") or 0) > 90
        )

        row = {
            "workers": int(nw),
            "n_genes": len(genes),
            "n_genes_completed": len(run.results),
            "wall_seconds": run.wall_seconds,
            "cpu_time_parent_seconds": cpu_parent,
            "seconds_per_gene_mean": float(np.mean(walls)),
            "seconds_per_gene_median": float(np.median(walls)),
            "seconds_per_gene_min": float(np.min(walls)),
            "seconds_per_gene_max": float(np.max(walls)),
            "throughput_genes_per_min": len(genes) / (run.wall_seconds / 60.0),
            "speedup_vs_8": speedup_vs_8,
            "gain_vs_8": gain_vs_8,
            "efficiency_vs_8": efficiency_vs_8,
            "cpu_mean": sys_s.get("cpu_mean"),
            "cpu_max": sys_s.get("cpu_max"),
            "ram_peak_mb": sys_s.get("ram_peak_mb"),
            "ram_mean_mb": sys_s.get("ram_mean_mb"),
            "ram_pct_peak": sys_s.get("ram_pct_peak"),
            "ram_pct_mean": sys_s.get("ram_pct_mean"),
            "python_rss_peak_mb": sys_s.get("python_rss_peak_mb"),
            "python_rss_mean_mb": sys_s.get("python_rss_mean_mb"),
            "mem_per_process_est_mb": (
                None
                if not sys_s.get("python_rss_peak_mb")
                else float(sys_s["python_rss_peak_mb"] / max(nw, 1))
            ),
            "swap_pct_peak": sys_s.get("swap_pct_peak"),
            "cpu_temp_max": sys_s.get("cpu_temp_max"),
            "cpu_temp_mean": sys_s.get("cpu_temp_mean"),
            "cpu_freq_mean": sys_s.get("cpu_freq_mean"),
            "throttling_suspected": sys_s.get("throttling_suspected"),
            "oversubscription_severe": oversub_severe,
            "n_python_procs_max": sys_s.get("n_python_procs_max"),
            "results_valid": bool(valid),
            "validity_deferred": deferred,
            "all_genes_ok": bool(ok_all),
            "early_stopped": False,
            "abort_reason": None,
            "seed": int(item.seed),
            "mask_seed": int(item.seed),
            "checkpoint_dir": str(ckpt),
            "gene_walls": {r["gene"]: r["wall_seconds"] for r in run.results},
            "fingerprints": run.fingerprints,
            "tail": {
                k: v
                for k, v in tail.items()
                if k != "activity_series"
            },
            "activity_series": tail.get("activity_series"),
            "monitor_samples": sys_s.get("samples"),
        }
        rows.append(row)
        if ok_all and (
            best_wall_so_far is None or run.wall_seconds < best_wall_so_far
        ):
            best_wall_so_far = float(run.wall_seconds)
        print(
            f"  wall={run.wall_seconds:.1f}s speedup_vs_8={speedup_vs_8:.3f} "
            f"gain={gain_vs_8*100 if np.isfinite(gain_vs_8) else float('nan'):.1f}% "
            f"cpu={sys_s.get('cpu_mean')} valid={valid} "
            f"best_wall_so_far={best_wall_so_far}",
            flush=True,
        )
        (OUT_DIR / "benchmark_extended_partial.json").write_text(
            json.dumps(
                {"host": host, "meta": meta, "rows": rows, "errors": errors},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        if args.cooldown_seconds > 0:
            time.sleep(float(args.cooldown_seconds))

    # Recompute validity/speedups now that baseline 8 is known
    by_w = {int(r["workers"]): r for r in rows if not r.get("early_stopped")}
    if 8 in by_w and not by_w[8].get("early_stopped"):
        baseline_fps = by_w[8]["fingerprints"]
        baseline_wall = by_w[8]["wall_seconds"]
        for r in rows:
            if r.get("early_stopped"):
                r["results_valid"] = False
                r["validity_deferred"] = False
                continue
            r["speedup_vs_8"] = float(baseline_wall / r["wall_seconds"])
            r["gain_vs_8"] = float((baseline_wall - r["wall_seconds"]) / baseline_wall)
            r["efficiency_vs_8"] = r["speedup_vs_8"] / (r["workers"] / 8.0)
            r["results_valid"] = bool(
                r["all_genes_ok"]
                and fingerprints_equal(
                    baseline_fps, r["fingerprints"], rtol=FP_RTOL, atol=FP_ATOL
                )
            )
            r["validity_deferred"] = False

    # Include early-stopped rows in lookup for reporting only
    by_w_all = {int(r["workers"]): r for r in rows}
    by_w = {w: r for w, r in by_w_all.items() if not r.get("early_stopped")} or by_w_all

    decision = choose_extended(rows, baseline_workers=8)
    recommended = int(decision["recommended_workers"])
    cons = int(decision["conservative_workers"])

    # ETA from recommended row effective s/gene
    rec_row = by_w.get(recommended) or by_w.get(8)
    if rec_row is None:
        # Fall back to any completed (non-aborted) row
        completed = [r for r in rows if not r.get("early_stopped")]
        rec_row = completed[0] if completed else rows[0]
    eff_s_gene = rec_row["wall_seconds"] / max(len(genes), 1)
    eta = _eta_block(eff_s_gene, rec_row.get("speedup_vs_8") or 1.0)

    # Plots (completed configs only)
    completed_rows = [r for r in rows if not r.get("early_stopped")]
    if completed_rows:
        _plot_summary(completed_rows, OUT_DIR / "benchmark_extended.png")
    # Prefer activity/cpu plots for recommended and for 8
    for label, w in [("baseline8", 8), ("recommended", recommended)]:
        if w in by_w and not by_w[w].get("early_stopped"):
            _plot_activity(
                by_w[w], OUT_DIR / f"worker_activity_timeline_{label}_w{w}.png"
            )
            _plot_cpu_mem(by_w[w], OUT_DIR / f"cpu_memory_timeline_{label}_w{w}.png")
    # Canonical names pointing at recommended
    if recommended in by_w and not by_w[recommended].get("early_stopped"):
        _plot_activity(by_w[recommended], OUT_DIR / "worker_activity_timeline.png")
        _plot_cpu_mem(by_w[recommended], OUT_DIR / "cpu_memory_timeline.png")

    # Strip heavy samples from JSON rows for main file (keep in separate files)
    rows_light = []
    for r in rows:
        rl = dict(r)
        samples = rl.pop("monitor_samples", None)
        activity = rl.pop("activity_series", None)
        rows_light.append(rl)
        w = r["workers"]
        if samples is not None:
            (OUT_DIR / f"monitor_samples_w{w}.json").write_text(
                json.dumps(samples, indent=2, default=str), encoding="utf-8"
            )
        if activity is not None:
            (OUT_DIR / f"activity_w{w}.json").write_text(
                json.dumps(activity, indent=2, default=str), encoding="utf-8"
            )

    payload = {
        "host": host,
        "meta": meta,
        "rows": rows_light,
        "errors": errors,
        "decision": decision,
        "eta": eta,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_command_suggestion": (
            "python experiments/run_original_rfeca_targetwise.py --confirm "
            "--phase mcar --replicates 0 1 2 3 4 --resume --auto-continue "
            f"--gene-workers {recommended}"
        ),
    }

    csv_cols = [
        "workers",
        "wall_seconds",
        "seconds_per_gene_mean",
        "seconds_per_gene_median",
        "seconds_per_gene_min",
        "seconds_per_gene_max",
        "throughput_genes_per_min",
        "speedup_vs_8",
        "gain_vs_8",
        "efficiency_vs_8",
        "cpu_mean",
        "cpu_max",
        "ram_pct_peak",
        "ram_pct_mean",
        "swap_pct_peak",
        "throttling_suspected",
        "oversubscription_severe",
        "results_valid",
        "early_stopped",
        "abort_reason",
    ]
    pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in rows_light]).to_csv(
        OUT_DIR / "benchmark_extended.csv", index=False
    )
    (OUT_DIR / "benchmark_extended.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # Markdown
    md = [
        "# Extended gene-level parallel autotune",
        "",
        "## 1. Hardware",
        "",
        f"- Logical CPUs: {host['cpu_count_logical']}",
        f"- Physical cores: {host['cpu_count_physical']}",
        f"- RAM: {host['ram_total_mb']:.0f} MB",
        "",
        "## 2. Methodology",
        "",
        "- OriginalRFECA TARGET-WISE + repeated_mask_holdout",
        "- MCAR 20% replicate 0, seed from v2 schedule",
        "- max_candidates=49, use_scaler=false, KFold=5, BLAS threads=1",
        "- Parallelism: genes only",
        f"- Fingerprint tolerance: atol={FP_ATOL}, rtol={FP_RTOL}",
        f"- Promotion rule vs 8 workers: ≥{MIN_GAIN_VS_8*100:.0f}% wall reduction + safety",
        f"- Early stop window: {EARLY_STOP_WINDOW_S:.0f}s continuous "
        "(swap/paging, thermal throttle, temp critical, projected wall worse, instability); "
        "CPU% alone never aborts",
        "- Isolated under `artifacts/parallel_benchmark_extended/` (MCAR untouched)",
        "",
        "## 3. Genes (16)",
        "",
        ", ".join(genes),
        "",
        "## 4. Execution order",
        "",
        f"Seed={args.order_seed}; order: {worker_order}; cooldown={args.cooldown_seconds}s",
        "",
        "## 5. Comparative table",
        "",
        "| workers | wall (s) | mean s/gene | speedup vs 8 | gain vs 8 | eff vs 8 | CPU mean | RAM peak % | valid | early_stop |",
        "|---------|----------|-------------|--------------|-----------|----------|----------|------------|-------|------------|",
    ]
    for r in sorted(rows_light, key=lambda x: x["workers"]):
        def _f(key: str, default: float = float("nan")) -> float:
            v = r.get(key, default)
            try:
                return float(v) if v is not None and np.isfinite(float(v)) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        md.append(
            f"| {r['workers']} | {_f('wall_seconds'):.1f} | "
            f"{_f('seconds_per_gene_mean'):.1f} | "
            f"{_f('speedup_vs_8'):.3f} | {_f('gain_vs_8')*100:.1f}% | "
            f"{_f('efficiency_vs_8'):.3f} | "
            f"{_f('cpu_mean'):.1f} | {_f('ram_pct_peak'):.1f} | "
            f"{r.get('results_valid')} | {r.get('early_stopped', False)} |"
        )

    md += [
        "",
        "## 6–9. Speedup / CPU-RAM / validity / tail",
        "",
    ]
    for r in sorted(rows_light, key=lambda x: x["workers"]):
        tail = r.get("tail") or {}
        if r.get("early_stopped"):
            md.append(
                f"- **w={r['workers']}**: EARLY_STOPPED — {r.get('abort_reason')}; "
                f"completed={r.get('n_genes_completed')}/{r.get('n_genes')}, "
                f"projected={r.get('projected_wall_at_abort')}, "
                f"best_at_abort={r.get('best_wall_at_abort')}"
            )
            continue
        md.append(
            f"- **w={r['workers']}**: valid={r['results_valid']}, "
            f"CPU {r.get('cpu_mean'):.1f}% (max {r.get('cpu_max')}), "
            f"RAM peak {r.get('ram_pct_peak')}%, swap peak {r.get('swap_pct_peak')}%, "
            f"throttle={r.get('throttling_suspected')}, "
            f"mean_active={tail.get('mean_active_workers')}, "
            f"slowest_gene={tail.get('slowest_gene_seconds')}, "
            f"tail_idle_frac={tail.get('tail_idle_fraction')}"
        )

    md += [
        "",
        "## 10. Recommendation",
        "",
        f"- **Primary: {recommended} workers**",
        f"- **Conservative (overnight): {cons} workers**",
        f"- Update main `--gene-workers auto`: **{decision.get('update_main_recommendation')}**",
        "",
        decision.get("reason", ""),
        "",
        "## 11. Updated ETA (point ±15%)",
        "",
        f"- Effective s/gene @ {recommended}w: {eff_s_gene:.1f}",
        f"- 1 replica (50 genes): {eta['one_replica_50_genes']['low_hours']:.1f}–"
        f"{eta['one_replica_50_genes']['high_hours']:.1f} h "
        f"(point {eta['one_replica_50_genes']['point_hours']:.1f})",
        f"- MCAR 20% × 5: {eta['mcar_20_5reps']['low_hours']:.1f}–"
        f"{eta['mcar_20_5reps']['high_hours']:.1f} h",
        f"- MAR 20% × 5: {eta['mar_20_5reps']['low_hours']:.1f}–"
        f"{eta['mar_20_5reps']['high_hours']:.1f} h",
        f"- Reduced full: {eta['reduced_full_mcar_mar_20']['low_hours']:.1f}–"
        f"{eta['reduced_full_mcar_mar_20']['high_hours']:.1f} h",
        f"- Expanded 10/20/30 MCAR+MAR×5: "
        f"{eta['expanded_10_20_30_mcar_mar_5reps']['low_hours']:.1f}–"
        f"{eta['expanded_10_20_30_mcar_mar_5reps']['high_hours']:.1f} h",
        "",
        "## Resume command (NOT executed)",
        "",
        "```bash",
        payload["resume_command_suggestion"],
        "```",
        "",
    ]
    if errors:
        md += ["## Errors", ""]
        for e in errors:
            md.append(f"- workers={e.get('workers')}: {e.get('error')}")

    (OUT_DIR / "benchmark_extended.md").write_text("\n".join(md), encoding="utf-8")

    rec_payload = {
        "recommended_workers": recommended,
        "conservative_workers": cons,
        "baseline_workers": 8,
        "decision": decision,
        "eta": eta,
        "source": str(OUT_DIR / "benchmark_extended.json"),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "update_main_recommendation": bool(decision.get("update_main_recommendation")),
    }
    (OUT_DIR / "recommended_workers_extended.json").write_text(
        json.dumps(rec_payload, indent=2, default=str), encoding="utf-8"
    )

    # Update main recommendation only if criteria met
    if decision.get("update_main_recommendation") and recommended != 8:
        hist = []
        if MAIN_REC.exists():
            old = json.loads(MAIN_REC.read_text(encoding="utf-8"))
            hist_path = MAIN_REC.with_name("recommended_workers_history.jsonl")
            with hist_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
                            "previous": old,
                        },
                        default=str,
                    )
                    + "\n"
                )
            # keep copy
            MAIN_REC.with_name(
                f"recommended_workers_before_extended_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            ).write_text(json.dumps(old, indent=2, default=str), encoding="utf-8")
        MAIN_REC.parent.mkdir(parents=True, exist_ok=True)
        MAIN_REC.write_text(
            json.dumps(
                {
                    "recommended_workers": recommended,
                    "conservative_workers": cons,
                    "source": "parallel_benchmark_extended",
                    "previous_recommended_workers": 8,
                    "decision": decision,
                    "eta": eta,
                    "written_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Updated {MAIN_REC} → {recommended}", flush=True)
    else:
        print(
            f"Keeping main recommendation at 8 "
            f"(update_main={decision.get('update_main_recommendation')})",
            flush=True,
        )

    print("\n" + "\n".join(md), flush=True)
    print(f"\nWrote {OUT_DIR}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
