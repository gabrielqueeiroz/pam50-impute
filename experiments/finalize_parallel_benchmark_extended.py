#!/usr/bin/env python3
"""Finalize extended parallel benchmark from partial JSON (post Unicode crash)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bcimpute.imputation_original.parallel_genes import fingerprints_equal  # noqa: E402

# Import helpers from extended script
sys.path.insert(0, str(ROOT / "experiments"))
import benchmark_gene_parallelism_extended as ext  # noqa: E402

OUT = ROOT / "artifacts" / "parallel_benchmark_extended"
MAIN_REC = ROOT / "artifacts" / "parallel_benchmark" / "recommended_workers.json"
PARTIAL = OUT / "benchmark_extended_partial.json"
FP_ATOL = 1e-10
FP_RTOL = 0.0


def main() -> int:
    raw = PARTIAL.read_text(encoding="utf-8")
    # JSON may contain NaN literals from Python json.dumps default=str sometimes
    raw = raw.replace(": NaN", ": null")
    data = json.loads(raw)
    rows = data["rows"]
    host = data["host"]
    meta = data["meta"]
    errors = data.get("errors", [])
    genes = meta["genes"]

    by_w = {int(r["workers"]): r for r in rows}
    if 8 not in by_w:
        raise SystemExit("Baseline workers=8 missing from partial results")
    baseline_fps = by_w[8]["fingerprints"]
    # Convert winning_predictors lists back to tuples for comparison
    for fps in [baseline_fps] + [r["fingerprints"] for r in rows]:
        for g, fp in fps.items():
            if fp and isinstance(fp.get("winning_predictors"), list):
                fp["winning_predictors"] = tuple(fp["winning_predictors"])

    baseline_wall = by_w[8]["wall_seconds"]
    for r in rows:
        r["speedup_vs_8"] = float(baseline_wall / r["wall_seconds"])
        r["gain_vs_8"] = float((baseline_wall - r["wall_seconds"]) / baseline_wall)
        r["efficiency_vs_8"] = r["speedup_vs_8"] / (r["workers"] / 8.0)
        r["results_valid"] = bool(
            r.get("all_genes_ok", True)
            and fingerprints_equal(
                baseline_fps, r["fingerprints"], rtol=FP_RTOL, atol=FP_ATOL
            )
        )
        r["validity_deferred"] = False

    decision = ext.choose_extended(rows, baseline_workers=8)
    recommended = int(decision["recommended_workers"])
    cons = int(decision["conservative_workers"])
    rec_row = by_w.get(recommended) or by_w[8]
    eff_s_gene = rec_row["wall_seconds"] / len(genes)
    eta = ext._eta_block(eff_s_gene, rec_row.get("speedup_vs_8") or 1.0)

    # Plots (monitor samples may be embedded in rows)
    ext._plot_summary(rows, OUT / "benchmark_extended.png")
    for label, w in [("baseline8", 8), ("recommended", recommended)]:
        if w in by_w:
            # activity may be under activity_series key or separate file
            if by_w[w].get("activity_series") is None:
                ap = OUT / f"activity_w{w}.json"
                if ap.exists():
                    by_w[w]["tail"] = by_w[w].get("tail") or {}
                    by_w[w]["activity_series"] = json.loads(
                        ap.read_text(encoding="utf-8")
                    )
            # rebuild activity for plot helper
            if by_w[w].get("activity_series") and "tail" in by_w[w]:
                by_w[w]["tail"] = dict(by_w[w]["tail"])
                by_w[w]["tail"]["activity_series"] = by_w[w]["activity_series"]
            ext._plot_activity(
                {
                    **by_w[w],
                    "tail": {
                        **(by_w[w].get("tail") or {}),
                        "activity_series": by_w[w].get("activity_series"),
                    },
                },
                OUT / f"worker_activity_timeline_{label}_w{w}.png",
            )
            samples = by_w[w].get("monitor_samples")
            if samples is None:
                sp = OUT / f"monitor_samples_w{w}.json"
                if sp.exists():
                    samples = json.loads(sp.read_text(encoding="utf-8"))
            if samples:
                ext._plot_cpu_mem(
                    {**by_w[w], "monitor_samples": samples},
                    OUT / f"cpu_memory_timeline_{label}_w{w}.png",
                )
    if recommended in by_w:
        r = by_w[recommended]
        ext._plot_activity(
            {
                **r,
                "tail": {
                    **(r.get("tail") or {}),
                    "activity_series": r.get("activity_series"),
                },
            },
            OUT / "worker_activity_timeline.png",
        )
        samples = r.get("monitor_samples")
        if samples is None:
            sp = OUT / f"monitor_samples_w{recommended}.json"
            if sp.exists():
                samples = json.loads(sp.read_text(encoding="utf-8"))
        if samples:
            ext._plot_cpu_mem(
                {**r, "monitor_samples": samples},
                OUT / "cpu_memory_timeline.png",
            )

    rows_light = []
    for r in rows:
        rl = {k: v for k, v in r.items() if k not in {"monitor_samples", "activity_series"}}
        rows_light.append(rl)

    resume_cmd = (
        "python experiments/run_original_rfeca_targetwise.py --confirm "
        "--phase mcar --replicates 0 1 2 3 4 --resume --auto-continue "
        f"--gene-workers {recommended}"
    )
    payload = {
        "host": host,
        "meta": meta,
        "rows": rows_light,
        "errors": errors,
        "decision": decision,
        "eta": eta,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_command_suggestion": resume_cmd,
        "finalized_from_partial": True,
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
    ]
    pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in rows_light]).to_csv(
        OUT / "benchmark_extended.csv", index=False
    )
    (OUT / "benchmark_extended.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

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
        "- MCAR 20% replicate 0; BLAS threads=1; genes only",
        f"- Fingerprint atol={FP_ATOL}, rtol={FP_RTOL}",
        "- Promotion vs 8 workers requires >=5% wall reduction + safety",
        "- Isolated under artifacts/parallel_benchmark_extended/",
        "",
        "## 3. Genes (16)",
        "",
        ", ".join(genes),
        "",
        "## 4. Execution order",
        "",
        f"Seed={meta['order_seed']}; order: {meta['execution_order']}; "
        f"cooldown={meta['cooldown_seconds']}s",
        "",
        "## 5. Comparative table",
        "",
        "| workers | wall (s) | mean s/gene | speedup vs 8 | gain vs 8 | eff vs 8 | CPU mean | RAM peak % | valid |",
        "|---------|----------|-------------|--------------|-----------|----------|----------|------------|-------|",
    ]
    for r in sorted(rows_light, key=lambda x: x["workers"]):
        md.append(
            f"| {r['workers']} | {r['wall_seconds']:.1f} | {r['seconds_per_gene_mean']:.1f} | "
            f"{r['speedup_vs_8']:.3f} | {r['gain_vs_8']*100:.1f}% | {r['efficiency_vs_8']:.3f} | "
            f"{(r.get('cpu_mean') or float('nan')):.1f} | {(r.get('ram_pct_peak') or float('nan')):.1f} | "
            f"{r['results_valid']} |"
        )

    md += ["", "## 6-9. Speedup / CPU-RAM / validity / tail", ""]
    for r in sorted(rows_light, key=lambda x: x["workers"]):
        tail = r.get("tail") or {}
        md.append(
            f"- **w={r['workers']}**: valid={r['results_valid']}, "
            f"CPU {(r.get('cpu_mean') or float('nan')):.1f}% "
            f"(max {r.get('cpu_max')}), "
            f"RAM peak {r.get('ram_pct_peak')}%, swap {r.get('swap_pct_peak')}%, "
            f"throttle={r.get('throttling_suspected')}, "
            f"mean_active={tail.get('mean_active_workers')}, "
            f"slowest={tail.get('slowest_gene_seconds')}, "
            f"tail_idle={tail.get('tail_idle_fraction')}"
        )

    # ASCII-only reason (avoid Windows console issues)
    reason = str(decision.get("reason", "")).replace("\u2192", "->")
    md += [
        "",
        "## 10. Recommendation",
        "",
        f"- **Primary: {recommended} workers**",
        f"- **Conservative (overnight): {cons} workers**",
        f"- Update main --gene-workers auto: **{decision.get('update_main_recommendation')}**",
        "",
        reason,
        "",
        "## 11. Updated ETA (point +/-15%)",
        "",
        f"- Effective s/gene @ {recommended}w: {eff_s_gene:.1f}",
        f"- 1 replica (50 genes): {eta['one_replica_50_genes']['low_hours']:.1f}-"
        f"{eta['one_replica_50_genes']['high_hours']:.1f} h "
        f"(point {eta['one_replica_50_genes']['point_hours']:.1f})",
        f"- MCAR 20% x 5: {eta['mcar_20_5reps']['low_hours']:.1f}-"
        f"{eta['mcar_20_5reps']['high_hours']:.1f} h",
        f"- MAR 20% x 5: {eta['mar_20_5reps']['low_hours']:.1f}-"
        f"{eta['mar_20_5reps']['high_hours']:.1f} h",
        f"- Reduced full: {eta['reduced_full_mcar_mar_20']['low_hours']:.1f}-"
        f"{eta['reduced_full_mcar_mar_20']['high_hours']:.1f} h",
        f"- Expanded 10/20/30 MCAR+MAR x5: "
        f"{eta['expanded_10_20_30_mcar_mar_5reps']['low_hours']:.1f}-"
        f"{eta['expanded_10_20_30_mcar_mar_5reps']['high_hours']:.1f} h",
        "",
        "## Resume command (NOT executed)",
        "",
        "```bash",
        resume_cmd,
        "```",
        "",
    ]
    (OUT / "benchmark_extended.md").write_text("\n".join(md), encoding="utf-8")

    rec_payload = {
        "recommended_workers": recommended,
        "conservative_workers": cons,
        "baseline_workers": 8,
        "decision": decision,
        "eta": eta,
        "source": str(OUT / "benchmark_extended.json"),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "update_main_recommendation": bool(decision.get("update_main_recommendation")),
    }
    # Sanitize decision reason for JSON consumers on Windows
    if isinstance(rec_payload["decision"].get("reason"), str):
        rec_payload["decision"]["reason"] = rec_payload["decision"]["reason"].replace(
            "\u2192", "->"
        )
    (OUT / "recommended_workers_extended.json").write_text(
        json.dumps(rec_payload, indent=2, default=str), encoding="utf-8"
    )

    if decision.get("update_main_recommendation") and recommended != 8:
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
            MAIN_REC.with_name(
                f"recommended_workers_before_extended_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            ).write_text(json.dumps(old, indent=2, default=str), encoding="utf-8")
        MAIN_REC.write_text(
            json.dumps(
                {
                    "recommended_workers": recommended,
                    "conservative_workers": cons,
                    "source": "parallel_benchmark_extended",
                    "previous_recommended_workers": 8,
                    "decision": rec_payload["decision"],
                    "eta": eta,
                    "written_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Updated main recommendation -> {recommended}")
    else:
        print("Keeping main recommendation at 8")

    print("\n".join(md))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
