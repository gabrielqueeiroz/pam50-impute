#!/usr/bin/env python3
"""
Orchestrate reduced OriginalRFECA resume: MCAR 20% x5 then MAR 20% x5.

Does NOT run 10%/30%. Writes status + triggers consolidation after success.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "original_rfeca_reduced"
STATUS = OUT / "execution_status.json"
REDUCED_ROOT = ROOT / "artifacts" / "original_rfeca_reduced_metabric"


def _write_status(**kwargs) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cur = {}
    if STATUS.exists():
        try:
            cur = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(kwargs)
    cur["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Validate checkpoint
    print("=== Pre-resume validation ===", flush=True)
    rc = subprocess.call(
        [sys.executable, str(ROOT / "experiments" / "validate_reduced_resume.py")],
        cwd=str(ROOT),
    )
    summary = json.loads((OUT / "checkpoint_summary.json").read_text(encoding="utf-8"))
    if rc != 0 or not summary.get("compatible"):
        _write_status(
            phase="failed",
            reason="checkpoint incompatible",
            summary=summary,
        )
        print("ABORT: checkpoint incompatible", flush=True)
        return 2

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_original_rfeca_targetwise.py"),
        "--confirm",
        "--phase",
        "mcar",
        "--replicates",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--resume",
        "--auto-continue",
        "--gene-workers",
        "16",
        "--rates",
        "0.20",
        "--evaluation",
        "repeated_mask_holdout",
    ]
    cmd_str = " ".join(cmd)
    print("=== Resume command ===", flush=True)
    print(cmd_str, flush=True)

    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[k] = "1"

    _write_status(
        phase="running",
        mechanism="mcar",
        rate=0.20,
        gene_workers=16,
        resume=True,
        n_completed_rep0=summary["n_completed"],
        n_pending_rep0=summary["n_pending"],
        command=cmd_str,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_root=str(REDUCED_ROOT),
        monitor_stop=False,
    )

    # Start monitor
    mon_log = OUT / "system_monitoring.csv"
    mon = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "experiments" / "monitor_reduced_run.py"),
            "--out",
            str(mon_log),
            "--status",
            str(STATUS),
            "--interval",
            "20",
        ],
        cwd=str(ROOT),
        env=env,
    )

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    exit_code = None
    try:
        while True:
            # Safety stop requested by monitor
            st = json.loads(STATUS.read_text(encoding="utf-8"))
            if st.get("monitor_stop") and st.get("stop_reason"):
                print(f"SAFETY STOP: {st['stop_reason']}", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _write_status(
                    phase="stopped",
                    reason=st["stop_reason"],
                    elapsed_hours=(time.time() - t0) / 3600,
                    resume_command=cmd_str,
                )
                exit_code = 3
                break
            rc = proc.poll()
            if rc is not None:
                exit_code = rc
                break
            time.sleep(10)
    finally:
        # Stop monitor
        try:
            st = json.loads(STATUS.read_text(encoding="utf-8"))
            st["monitor_stop"] = True
            if exit_code == 0:
                st["phase"] = "consolidating"
            STATUS.write_text(json.dumps(st, indent=2), encoding="utf-8")
        except Exception:
            pass
        try:
            mon.wait(timeout=30)
        except Exception:
            mon.kill()

    elapsed_h = (time.time() - t0) / 3600
    if exit_code == 0:
        print("=== Consolidating reduced results ===", flush=True)
        cons_rc = subprocess.call(
            [
                sys.executable,
                str(ROOT / "experiments" / "consolidate_reduced_rfeca.py"),
                "--elapsed-hours",
                f"{elapsed_h:.4f}",
            ],
            cwd=str(ROOT),
            env=env,
        )
        _write_status(
            phase="done" if cons_rc == 0 else "done_with_consolidation_warnings",
            runner_exit_code=exit_code,
            consolidation_exit_code=cons_rc,
            elapsed_hours=elapsed_h,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        return cons_rc
    else:
        _write_status(
            phase="failed" if exit_code not in {3} else "stopped",
            runner_exit_code=exit_code,
            elapsed_hours=elapsed_h,
            resume_command=cmd_str,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        return int(exit_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
