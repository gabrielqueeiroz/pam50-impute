#!/usr/bin/env python3
"""Run METABRIC multi-clf MCAR, then MAR (20%/30%, classification-only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_metabric_multiclf.py"


def main() -> int:
    for mech in ("mcar", "mar"):
        print("=" * 72, flush=True)
        print(f"Starting METABRIC multi-clf: {mech}", flush=True)
        print("=" * 72, flush=True)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--confirm",
            "--mechanism",
            mech,
            "--seed-scheme",
            "legacy",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            print(f"FAILED on {mech} with code {proc.returncode}", flush=True)
            return proc.returncode
        print(f"PASS {mech}", flush=True)
    print("\nBoth MCAR and MAR multi-clf runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
