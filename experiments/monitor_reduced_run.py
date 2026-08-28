#!/usr/bin/env python3
"""Background system monitor for reduced OriginalRFECA run (safety + CSV log)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    raise SystemExit("psutil required")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--ram-limit-pct", type=float, default=85.0)
    p.add_argument("--ram-sustain-seconds", type=float, default=120.0)
    args = p.parse_args()

    out = Path(args.out)
    status_path = Path(args.status)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "utc",
        "elapsed_s",
        "cpu_pct",
        "ram_pct",
        "ram_used_mb",
        "swap_pct",
        "n_python",
        "cpu_freq_mhz",
        "cpu_temp_c",
        "stop_requested",
        "stop_reason",
    ]
    new_file = not out.exists()
    t0 = time.time()
    high_ram_since = None
    stop_reason = None

    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            w.writeheader()
        while True:
            # Stop if status says finished/stopped
            if status_path.exists():
                try:
                    st = json.loads(status_path.read_text(encoding="utf-8"))
                    if st.get("phase") in {"done", "stopped", "failed"}:
                        break
                    if st.get("monitor_stop"):
                        break
                except Exception:
                    pass

            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            cpu = float(psutil.cpu_percent(interval=1.0))
            n_py = sum(
                1
                for proc in psutil.process_iter(["name"])
                if proc.info.get("name") and "python" in proc.info["name"].lower()
            )
            freq = None
            try:
                f = psutil.cpu_freq()
                freq = float(f.current) if f else None
            except Exception:
                pass
            temp = None
            try:
                tmap = psutil.sensors_temperatures()
                vals = [
                    float(e.current)
                    for entries in (tmap or {}).values()
                    for e in entries
                    if e.current is not None
                ]
                temp = max(vals) if vals else None
            except Exception:
                pass

            if vm.percent >= args.ram_limit_pct:
                if high_ram_since is None:
                    high_ram_since = time.time()
                elif time.time() - high_ram_since >= args.ram_sustain_seconds:
                    stop_reason = (
                        f"RAM {vm.percent:.1f}% >= {args.ram_limit_pct}% "
                        f"for >={args.ram_sustain_seconds}s"
                    )
            else:
                high_ram_since = None

            if temp is not None and temp >= 95:
                stop_reason = f"CPU temp {temp:.1f}C >= 95"

            row = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(time.time() - t0, 1),
                "cpu_pct": cpu,
                "ram_pct": vm.percent,
                "ram_used_mb": round(vm.used / (1024**2), 1),
                "swap_pct": swap.percent,
                "n_python": n_py,
                "cpu_freq_mhz": freq,
                "cpu_temp_c": temp,
                "stop_requested": bool(stop_reason),
                "stop_reason": stop_reason or "",
            }
            w.writerow(row)
            fh.flush()

            if stop_reason:
                # Signal orchestrator via status file
                payload = {}
                if status_path.exists():
                    try:
                        payload = json.loads(status_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                payload["monitor_stop"] = True
                payload["stop_reason"] = stop_reason
                payload["phase"] = payload.get("phase", "running")
                status_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                break

            time.sleep(max(0.0, args.interval - 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
