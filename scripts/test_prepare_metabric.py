#!/usr/bin/env python3
"""Validate METABRIC preparation artifacts without reloading the full expression matrix."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "metabric"
TARGET_SUBTYPES = {"LumA", "LumB", "Her2", "Basal"}
EXPECTED_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]


def fingerprint(df: pd.DataFrame) -> str:
    ordered = df.sort_values("sample_id").reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def main() -> int:
    required = [
        "metabric_inventory.json",
        "clinical_field_report.csv",
        "gene_mapping_report.csv",
        "native_missingness_report.csv",
        "metabric_pam50_4class.csv",
        "preparation_log.txt",
    ]
    missing = [f for f in required if not (OUT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")

    final = pd.read_csv(OUT_DIR / "metabric_pam50_4class.csv")
    mapping = pd.read_csv(OUT_DIR / "gene_mapping_report.csv")
    clinical = pd.read_csv(OUT_DIR / "clinical_field_report.csv")
    inv = json.loads((OUT_DIR / "metabric_inventory.json").read_text(encoding="utf-8"))

    gene_cols = [c for c in final.columns if c not in {"sample_id", "PAM50"}]

    assert len(gene_cols) == 50, gene_cols
    assert len(set(gene_cols)) == 50
    assert gene_cols == EXPECTED_GENES
    assert not final["sample_id"].duplicated().any()
    assert not final.columns.duplicated().any()
    assert set(final["PAM50"].unique()) == TARGET_SUBTYPES
    assert not final[gene_cols].isna().any().any()
    assert (mapping["status"] == "mapped").all()
    assert (mapping["match_method"] == "entrez").sum() == 50
    assert "CLAUDIN_SUBTYPE" in set(clinical["column_name"])
    assert inv["expression_profile_documentation"]["datatype"] == "CONTINUOUS"
    assert "raw" not in (inv["expression_profile_documentation"]["datatype"] or "").lower()

    fp = fingerprint(final)
    assert fp == inv["final_matrix"]["content_sha256"], (fp, inv["final_matrix"]["content_sha256"])

    print("VALIDATION OK")
    print(f"n_samples={len(final)}")
    print(f"class_distribution={final['PAM50'].value_counts().to_dict()}")
    print(f"fingerprint={fp}")
    print(f"datatype={inv['expression_profile_documentation']['datatype']}")
    print(f"profile_description={inv['expression_profile_documentation']['profile_description']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
