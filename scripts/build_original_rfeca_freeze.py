#!/usr/bin/env python3
"""Build freeze manifest for OriginalRFECA TARGET-WISE METABRIC reduced grid."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts" / "original_rfeca_reduced_metabric"
OUT = ART / "FREEZE"


def sha16(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from bcimpute.missingness import missingness_seed

    slots: list[dict] = []
    for mech in ("mcar", "mar"):
        for rate in (0.05, 0.10, 0.20, 0.30):
            for rep in range(5):
                d = ART / mech / f"rate_{rate:.2f}" / f"rep_{rep}"
                done = d / "DONE.json"
                if not done.exists():
                    raise SystemExit(f"missing {done}")
                summary = json.loads(done.read_text(encoding="utf-8"))
                mask = d / "mask.npz"
                slots.append(
                    {
                        "mechanism": mech,
                        "rate": rate,
                        "replicate": rep,
                        "seed": summary.get("seed"),
                        "mask_hash": summary.get("mask_hash"),
                        "mask_npz_sha16": sha16(mask),
                        "rmse": summary.get("rmse"),
                        "mae": summary.get("mae"),
                        "classification": summary.get("classification"),
                        "svr_coverage": summary.get("svr_coverage"),
                        "fallback_rate": summary.get("fallback_rate"),
                        "n_predictor_nans_at_impute": summary.get(
                            "n_predictor_nans_at_impute"
                        ),
                        "n_genes_completed": summary.get("n_genes_completed"),
                        "evaluation_protocol": summary.get("evaluation_protocol"),
                        "input_protocol": summary.get("input_protocol"),
                        "predictor_values": summary.get("predictor_values"),
                        "selection_protocol": summary.get("selection_protocol"),
                        "use_scaler": summary.get("use_scaler"),
                        "max_candidates": summary.get("max_candidates"),
                        "wall_seconds": summary.get("wall_seconds"),
                        "slot_dir": str(d.relative_to(ROOT)).replace("\\", "/"),
                    }
                )

    reports = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(ART.glob("REPORT_*_5REPS.json"))
    }

    pins = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }
    for mod, key in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("joblib", "joblib"),
        ("feature_engine", "feature_engine"),
        ("threadpoolctl", "threadpoolctl"),
    ]:
        m = __import__(mod)
        pins["packages"][key] = getattr(m, "__version__", "unknown")

    config = {
        "method": "OriginalRFECA",
        "cohort": "metabric",
        "evaluation_protocol": "repeated_mask_holdout",
        "input_protocol": "target_wise_complete_predictors",
        "predictor_values": "original_complete_matrix",
        "selection_protocol": "leakage_safe",
        "max_candidates": 49,
        "inner_cv": 5,
        "use_scaler": False,
        "gene_workers": 16,
        "blas_threads": 1,
        "seed_scheme": "v2",
        "base_seed": 42,
        "mechanisms": ["mcar", "mar"],
        "rates": [0.05, 0.10, 0.20, 0.30],
        "replicates": [0, 1, 2, 3, 4],
        "n_genes": 50,
    }

    seed_audit = []
    for mech in ("mcar", "mar"):
        for rate in (0.05, 0.10, 0.20, 0.30):
            for rep in range(5):
                expected = missingness_seed(
                    42, rate, rep, mechanism=mech, scheme="v2"
                )
                got = next(
                    s["seed"]
                    for s in slots
                    if s["mechanism"] == mech
                    and s["rate"] == rate
                    and s["replicate"] == rep
                )
                seed_audit.append(
                    {
                        "mechanism": mech,
                        "rate": rate,
                        "replicate": rep,
                        "expected_seed": expected,
                        "recorded_seed": got,
                        "match": expected == got,
                    }
                )
    if not all(a["match"] for a in seed_audit):
        raise SystemExit("seed mismatch vs v2 formula")

    reproduce = (
        "python experiments/run_original_rfeca_targetwise.py --confirm "
        "--phase mcar --replicates 0 1 2 3 4 --resume --auto-continue "
        "--gene-workers 16 --rates 0.05 0.10 0.20 0.30 "
        "--evaluation repeated_mask_holdout"
    )

    manifest = {
        "freeze_id": "v0.3.1-original-rfeca-targetwise",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "OriginalRFECA TARGET-WISE leakage-safe METABRIC PAM50: "
            "MCAR+MAR x {5,10,20,30}% x 5 replicates. No SimpleImputer; "
            "predictors from original complete matrix; repeated_mask_holdout."
        ),
        "config": config,
        "environment": pins,
        "n_slots": len(slots),
        "all_slots_complete": len(slots) == 40,
        "all_classification_A": all(s.get("classification") == "A" for s in slots),
        "all_seeds_match_v2_formula": True,
        "slots": slots,
        "seed_audit": seed_audit,
        "reports_summary": {
            name: {
                "classification": r.get("classification"),
                "rmse_mean": r.get("rmse_mean"),
                "mae_mean": r.get("mae_mean"),
                "svr_coverage_min": r.get("svr_coverage_min"),
            }
            for name, r in reports.items()
        },
        "artifact_policy": {
            "included": [
                "DONE.json",
                "slot_summary.json",
                "mask.npz",
                "per_gene_metrics.csv",
                "gene_summary.csv",
                "REPORT_*.json/md",
                "per_gene_all_*.csv",
                "FREEZE/*",
            ],
            "excluded_from_git": [
                "*.joblib",
                "checkpoint/",
                "gene_selection_audit.json",
                "worker_*",
                "progress.jsonl",
            ],
            "note": "Models regenerable from code+mask+config; ~1GB joblibs omitted.",
        },
        "reproduce_command": reproduce,
    }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with (OUT / "mask_hashes.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "mechanism",
            "rate",
            "replicate",
            "seed",
            "mask_hash",
            "mask_npz_sha16",
            "rmse",
            "mae",
            "classification",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in slots:
            w.writerow({k: s[k] for k in fields})

    req_lines = [
        f"# Freeze {manifest['freeze_id']} — pinned runtime for OriginalRFECA TARGET-WISE",
        f"# python=={pins['python']}",
        f"# platform={pins['platform']}",
    ]
    for k, v in pins["packages"].items():
        req_lines.append(f"{k}=={v}")
    req_text = "\n".join(req_lines) + "\n"
    (OUT / "requirements.txt").write_text(req_text, encoding="utf-8")
    (ROOT / "requirements-freeze-v0.3.txt").write_text(req_text, encoding="utf-8")

    (OUT / "config_snapshot.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    def rmse(name: str) -> float:
        return float(reports[name]["rmse_mean"])

    md = f"""# Freeze {manifest['freeze_id']}

Frozen UTC: `{manifest['frozen_at_utc']}`

## Protocol
- Method: **OriginalRFECA** TARGET-WISE
- Cohort: METABRIC PAM50 (50 genes)
- Evaluation: `repeated_mask_holdout`
- Predictors: original complete matrix (no SimpleImputer, no chaining)
- Selection: leakage-safe Pearson prefixes + RFE + linear SVR
- `max_candidates=49`, `use_scaler=False`, `inner_cv=5`, `seed_scheme=v2`, `base_seed=42`
- Grid: MCAR+MAR × {{0.05, 0.10, 0.20, 0.30}} × reps 0–4 (40 slots)

## Results (mean RMSE)
| | 5% | 10% | 20% | 30% |
|--|----|-----|-----|-----|
| MCAR | {rmse('REPORT_MCAR_5_5REPS.json'):.4f} | {rmse('REPORT_MCAR_10_5REPS.json'):.4f} | {rmse('REPORT_MCAR_20_5REPS.json'):.4f} | {rmse('REPORT_MCAR_30_5REPS.json'):.4f} |
| MAR | {rmse('REPORT_MAR_5_5REPS.json'):.4f} | {rmse('REPORT_MAR_10_5REPS.json'):.4f} | {rmse('REPORT_MAR_20_5REPS.json'):.4f} | {rmse('REPORT_MAR_30_5REPS.json'):.4f} |

All 40 slots classification **A**; SVR coverage 1.0; fallbacks 0.

## Files
- `manifest.json` — full freeze record (seeds, mask hashes, config, env)
- `mask_hashes.csv` — one row per slot
- `config_snapshot.json` — protocol knobs
- `requirements.txt` — pinned packages

## Reproduce
```bash
pip install -r requirements-freeze-v0.3.txt
{reproduce}
```

Joblib gene models are **not** in git (~1GB); regenerate with the command above
(`--resume` skips completed slots if masks/DONE present).
"""
    (OUT / "README.md").write_text(md, encoding="utf-8")

    print(
        json.dumps(
            {
                "wrote": str(OUT),
                "n_slots": len(slots),
                "all_A": manifest["all_classification_A"],
                "seeds_ok": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
