#!/usr/bin/env python3
"""
Wait for the in-flight MCAR METABRIC full run to finish, then launch MAR fulls.

Usage (from repo root):
  python scripts/run_mar_chain_after_mcar.py

Optional:
  --mcar-dir artifacts/metabric_full_20260724_185916
  --skip-wait   # start MAR immediately (only if MCAR already done)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MCAR = ROOT / "artifacts" / "metabric_full_20260724_185916"


def _mcar_done(mcar_dir: Path) -> bool:
    report = mcar_dir / "full_benchmark_report.json"
    if not report.exists():
        return False
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("status") == "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcar-dir", type=Path, default=DEFAULT_MCAR)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    mcar_dir = args.mcar_dir
    if not args.skip_wait:
        print(f"Waiting for MCAR METABRIC PASS in {mcar_dir} ...", flush=True)
        while not _mcar_done(mcar_dir):
            print(f"  still waiting ({time.strftime('%H:%M:%S')})", flush=True)
            time.sleep(max(15, args.poll_seconds))
        print("MCAR METABRIC PASS detected.", flush=True)
    elif not _mcar_done(mcar_dir):
        print(f"ERROR: --skip-wait but {mcar_dir} is not PASS yet.", file=sys.stderr)
        return 2

    py = sys.executable
    cmds = [
        [py, str(ROOT / "experiments" / "run_full_discovery.py"), "--confirm", "--mechanism", "mar"],
        [py, str(ROOT / "experiments" / "run_full_metabric.py"), "--confirm", "--mechanism", "mar"],
    ]
    for cmd in cmds:
        print("\n>>>", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            print(f"FAILED with exit {proc.returncode}: {cmd}", file=sys.stderr)
            return proc.returncode

    print("\nMAR Discovery + METABRIC full chain completed.", flush=True)
    print("Next: run consolidated stats across MCAR + MAR artifacts.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
