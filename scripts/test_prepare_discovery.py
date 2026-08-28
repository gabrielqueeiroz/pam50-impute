#!/usr/bin/env python3
"""Validate discovery preparation artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "discovery"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "discovery" / "df_MI_no_missing.csv"
EXPECTED = {"LumA": 57, "Basal": 29, "LumB": 17, "Her2": 14}
GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]


def main() -> int:
    required = [
        "discovery_inventory.json",
        "discovery_provenance_report.md",
        "discovery_preparation_log.txt",
        "discovery_excluded_samples.csv",
        "discovery_pam50_4class.csv",
        "discovery_fingerprint.json",
    ]
    missing = [f for f in required if not (OUT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(missing)

    final = pd.read_csv(OUT_DIR / "discovery_pam50_4class.csv")
    inv = json.loads((OUT_DIR / "discovery_inventory.json").read_text(encoding="utf-8"))
    fp = json.loads((OUT_DIR / "discovery_fingerprint.json").read_text(encoding="utf-8"))

    assert list(final.columns) == ["sample_id", *GENES, "PAM50"]
    assert len(GENES) == 50
    assert not final["sample_id"].duplicated().any()
    assert set(final["PAM50"]) == set(EXPECTED)
    assert final["PAM50"].value_counts().to_dict() == EXPECTED
    assert not final[GENES].isna().any().any()
    assert inv["cohort_key"] == "discovery"
    assert "CPTAC" not in inv["cohort_key"]

    # values fingerprint
    ordered = final.sort_values("sample_id").reset_index(drop=True)
    values_csv = ordered[GENES + ["PAM50"]].to_csv(index=False).encode()
    digest = hashlib.sha256(values_csv).hexdigest()
    assert digest == fp["values_and_labels_sha256"]

    # raw unchanged + value parity via synthetic order
    raw = pd.read_csv(RAW, index_col=0)
    raw_X = raw.loc[raw["PAM50"].isin(EXPECTED), GENES].astype(float)
    tmp = ordered.copy()
    tmp["_ord"] = tmp["sample_id"].str.replace("discovery_row_", "", regex=False).astype(int) - 1
    tmp = tmp.sort_values("_ord")
    assert np.max(np.abs(tmp[GENES].to_numpy() - raw_X.to_numpy())) == 0.0

    print("VALIDATION OK")
    print(f"n={len(final)}")
    print(f"class_distribution={final['PAM50'].value_counts().to_dict()}")
    print(f"fingerprint={digest}")
    print(f"id_status={inv['sample_ids']['status']}")
    print(f"provenance_confidence={inv['provenance']['confidence']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
