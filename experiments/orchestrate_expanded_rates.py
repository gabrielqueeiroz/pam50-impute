#!/usr/bin/env python3
"""
Wait for reduced OriginalRFECA @20% (MCAR+MAR ×5) to finish, then run 10% and 30%
under the same protocol:

  target-wise OriginalRFECA, repeated_mask_holdout, seed_scheme=v2,
  gene_workers=16, reps 0–4, MCAR then MAR (auto-continue).

Does not start until all MAR 0.20 slots have DONE.json and the 20% runner
process has exited (avoids competing for CPU with the active job).
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
REDUCED_ROOT = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
OUT = ROOT / "artifacts" / "original_rfeca_expanded_rates"
STATUS = OUT / "execution_status.json"
RATES = [0.10, 0.30]
POLL_S = 60


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


def _slot_done(mech: str, rate: float, rep: int) -> bool:
    return (REDUCED_ROOT / mech / f"rate_{rate:.2f}" / f"rep_{rep}" / "DONE.json").exists()


def _twenty_pct_complete() -> bool:
    return all(_slot_done("mcar", 0.20, r) for r in range(5)) and all(
        _slot_done("mar", 0.20, r) for r in range(5)
    )


def _runner_pids() -> list[int]:
    """PIDs of active reduced 20% runner / orchestrator (not this script)."""
    me = os.getpid()
    markers = (
        "run_original_rfeca_targetwise",
        "orchestrate_reduced_rfeca",
    )
    try:
        import psutil  # type: ignore

        pids = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                pid = int(p.info["pid"])
                if pid == me:
                    continue
                cmd = " ".join(p.info.get("cmdline") or [])
                if "orchestrate_expanded_rates" in cmd:
                    continue
                if any(m in cmd for m in markers):
                    pids.append(pid)
            except (psutil.Error, TypeError, ValueError):
                continue
        return pids
    except ImportError:
        pass

    # Windows fallback without psutil
    if sys.platform.startswith("win"):
        ps = (
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            pids = []
            for row in data:
                cmd = str(row.get("CommandLine") or "")
                pid = int(row.get("ProcessId") or 0)
                if pid == me or not cmd:
                    continue
                if "orchestrate_expanded_rates" in cmd:
                    continue
                if any(m in cmd for m in markers):
                    pids.append(pid)
            return pids
        except Exception:
            return []
    return []


def _wait_for_twenty_pct() -> None:
    _write_status(phase="waiting_for_20pct", rates=RATES)
    print("=== Waiting for MCAR+MAR @20% ×5 to finish ===", flush=True)
    while True:
        done20 = _twenty_pct_complete()
        pids = _runner_pids()
        _write_status(
            phase="waiting_for_20pct",
            twenty_pct_slots_complete=done20,
            active_20pct_pids=pids,
        )
        if done20 and not pids:
            print("20% complete and runner idle — starting 10%/30%.", flush=True)
            return
        n_mar = sum(1 for r in range(5) if _slot_done("mar", 0.20, r))
        print(
            f"[wait] mar_0.20_done={n_mar}/5 slots_ok={done20} "
            f"active_pids={pids} sleep={POLL_S}s",
            flush=True,
        )
        time.sleep(POLL_S)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _wait_for_twenty_pct()

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
        "0.10",
        "0.30",
        "--evaluation",
        "repeated_mask_holdout",
    ]
    cmd_str = " ".join(cmd)
    print("=== Expansion command ===", flush=True)
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
        rates=RATES,
        gene_workers=16,
        command=cmd_str,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_root=str(REDUCED_ROOT),
        order="MCAR(0.10→0.30) then MAR(0.10→0.30)",
    )

    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    elapsed_h = (time.time() - t0) / 3600

    # Light consolidation: confirm all 10/30 slots
    summary = {
        "rates": RATES,
        "mechanisms": ["mcar", "mar"],
        "replicates": list(range(5)),
        "slots": {},
    }
    all_ok = True
    for mech in ("mcar", "mar"):
        for rate in RATES:
            for rep in range(5):
                key = f"{mech}/rate_{rate:.2f}/rep_{rep}"
                ok = _slot_done(mech, rate, rep)
                summary["slots"][key] = ok
                all_ok = all_ok and ok

    (OUT / "expansion_slot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_status(
        phase="done" if rc == 0 and all_ok else "failed",
        runner_exit_code=rc,
        all_slots_complete=all_ok,
        elapsed_hours=elapsed_h,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    print(
        json.dumps(
            {
                "exit_code": rc,
                "all_slots_complete": all_ok,
                "elapsed_hours": elapsed_h,
                "status": str(STATUS),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if rc == 0 and all_ok else int(rc or 1)


if __name__ == "__main__":
    raise SystemExit(main())
