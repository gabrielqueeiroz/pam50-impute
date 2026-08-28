#!/usr/bin/env python3
"""Pre-resume validation for OriginalRFECA reduced MCAR checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bcimpute.config import PAM50_GENES  # noqa: E402
from bcimpute.data import load_cohort  # noqa: E402
from bcimpute.missingness import generate_missingness_sets, missingness_seed  # noqa: E402

SLOT = (
    ROOT
    / "artifacts"
    / "original_rfeca_reduced_metabric"
    / "mcar"
    / "rate_0.20"
    / "rep_0"
)
OUT = ROOT / "artifacts" / "original_rfeca_reduced"
CK = SLOT / "checkpoint"


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _mask_hash(mask: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
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
    expected_seed = missingness_seed(42, 0.20, 0, mechanism="mcar", scheme="v2")
    expected_mask_hash = _mask_hash(item.mask)

    # Persist mask if missing (resume path)
    mask_path = SLOT / "mask.npz"
    if mask_path.exists():
        loaded = np.load(mask_path)
        stored_mask = loaded["mask"]
        stored_seed = int(loaded["seed"][0]) if "seed" in loaded else None
        stored_hash = _mask_hash(stored_mask)
        if stored_seed is not None and stored_seed != expected_seed:
            issues.append(
                f"mask seed mismatch: stored={stored_seed} expected={expected_seed}"
            )
        if not np.array_equal(stored_mask, item.mask):
            issues.append("stored mask.npz differs from regenerated schedule mask")
        if stored_hash != expected_mask_hash:
            issues.append(
                f"mask hash mismatch: stored={stored_hash} expected={expected_mask_hash}"
            )
    else:
        issues.append("mask.npz missing (will be written on resume — not a blocker)")

    models_path = CK / "gene_models.joblib"
    genes_dir = CK / "genes"
    progress_path = CK / "progress.jsonl"

    if not models_path.exists():
        issues.append("CRITICAL: gene_models.joblib missing — cannot resume safely")
        gene_models = {}
    else:
        gene_models = joblib.load(models_path)

    progress_genes = []
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                progress_genes.append(json.loads(line))

    audit_genes = sorted(p.stem for p in genes_dir.glob("*.json")) if genes_dir.exists() else []
    model_genes = sorted(gene_models.keys())

    # Completed = present in both models and audit with status ok
    completed = []
    for g in model_genes:
        ap = genes_dir / f"{g}.json"
        if not ap.exists():
            issues.append(f"model without audit JSON: {g}")
            continue
        payload = json.loads(ap.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            issues.append(f"gene {g} status={payload.get('status')}")
            continue
        completed.append(g)

    pending = [g for g in PAM50_GENES if g not in set(completed)]
    # Also check sorted-order continuity used by serial fit
    sorted_all = sorted(PAM50_GENES)
    expected_serial_prefix = sorted_all[: len(completed)]
    if completed != expected_serial_prefix and set(completed) != set(expected_serial_prefix):
        # not necessarily wrong if parallel mixed — warn only
        pass
    if set(completed) != set(expected_serial_prefix):
        issues.append(
            "completed set differs from first "
            f"{len(completed)} genes in sorted PAM50 order"
        )

    config = {
        "method": "OriginalRFECA",
        "evaluation_protocol": "repeated_mask_holdout",
        "input_protocol": "target_wise_complete_predictors",
        "predictor_values": "original_complete_matrix",
        "selection_protocol": "leakage_safe",
        "max_candidates": 49,
        "use_scaler": False,
        "inner_cv": 5,
        "gene_workers": 16,
        "conservative_workers": 10,
        "mechanism": "mcar",
        "rate": 0.20,
        "replicates": [0, 1, 2, 3, 4],
        "resume": True,
        "cohort": "metabric",
        "seed_scheme": "v2",
        "base_seed": 42,
        "expected_seed_rep0": expected_seed,
        "expected_mask_hash_rep0": expected_mask_hash,
        "blas_threads": 1,
        "out_root": str(ROOT / "artifacts" / "original_rfeca_reduced_metabric"),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    critical = [i for i in issues if i.startswith("CRITICAL") or "mismatch" in i]
    # Compatible resume states for rep0:
    # - partial (any consistent completed set with models+audits)
    # - fully complete (50/50) — later replicates continue
    consistent_counts = (
        len(completed) == len(model_genes) == len(audit_genes)
        or (
            len(completed) == len(model_genes)
            and set(completed).issubset(set(audit_genes))
        )
    )
    compatible = (
        len(critical) == 0
        and len(completed) >= 1
        and len(completed) + len(pending) == 50
        and consistent_counts
    )

    summary = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "compatible": compatible,
        "issues": issues,
        "critical_issues": critical,
        "completed_genes": completed,
        "n_completed": len(completed),
        "pending_genes": pending,
        "n_pending": len(pending),
        "n_expected_total": 50,
        "progress_jsonl_count": len(progress_genes),
        "audit_json_count": len(audit_genes),
        "model_count": len(model_genes),
        "gene_models_sha16": _sha256_file(models_path),
        "mask_npz_sha16": _sha256_file(mask_path) if mask_path.exists() else None,
        "config": config,
        "config_hash": config_hash,
        "slot": str(SLOT),
    }
    (OUT / "checkpoint_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT / "run_config.json").write_text(
        json.dumps({**config, "config_hash": config_hash}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
