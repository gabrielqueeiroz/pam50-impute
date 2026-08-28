#!/usr/bin/env python3
"""
Prepare LinkedOmics CPTAC-BRCA prospective RNAseq with the same analysis-ready
schema and filtering rules used by prepare_discovery.py.

Output cohort key: cptac_linkedomics
Does NOT rename or replace the discovery cohort.
Does NOT run benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cptac_linkedomics"
OUT_DIR = ROOT / "data" / "processed" / "cptac_linkedomics"
EXTERNAL_DIR = ROOT / "data" / "external" / "cptac_brca_linkedomics"

CLINICAL_NAME = "HS_CPTAC_BRCA_2018_CLI.tsi"
RNASEQ_NAME = "HS_CPTAC_BRCA_2018_RNA_GENE.cct"
# Cached download may use alternate capitalization.
RNASEQ_ALIASES = [
    "HS_CPTAC_BRCA_2018_RNA_GENE.cct",
    "HS_CPTAC_BRCA_2018_RNA_Gene.cct",
]

CLINICAL_URL = (
    "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_CLI.tsi"
)
RNASEQ_URL = (
    "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_RNA_GENE.cct"
)

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]
TARGET_LABELS = ("LumA", "LumB", "Her2", "Basal")


class PrepLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{stamp}] {msg}"
        self.lines.append(line)
        print(line)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, dest: Path, log: PrepLogger) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "bioinfo-cptac-prep/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    if len(data) < 500 or b"<html" in data[:200].lower():
        raise RuntimeError(f"Download does not look like a data file: {url}")
    dest.write_bytes(data)
    log.log(f"Downloaded {url} -> {dest} ({len(data)} bytes)")


def stage_raw_sources(log: PrepLogger) -> tuple[Path, Path]:
    """Stage immutable copies under data/raw/cptac_linkedomics/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    clin_raw = RAW_DIR / CLINICAL_NAME
    rna_raw = RAW_DIR / RNASEQ_NAME

    # Prefer external cache, then raw, else download.
    clin_src = EXTERNAL_DIR / CLINICAL_NAME
    if not clin_src.exists() and not clin_raw.exists():
        _download(CLINICAL_URL, EXTERNAL_DIR / CLINICAL_NAME, log)
        clin_src = EXTERNAL_DIR / CLINICAL_NAME
    if not clin_raw.exists():
        shutil.copy2(clin_src if clin_src.exists() else EXTERNAL_DIR / CLINICAL_NAME, clin_raw)
        log.log(f"Staged clinical raw: {clin_raw}")
    else:
        log.log(f"Using existing clinical raw: {clin_raw}")

    rna_src = None
    for name in RNASEQ_ALIASES:
        cand = EXTERNAL_DIR / name
        if cand.exists() and cand.stat().st_size > 1_000_000:
            rna_src = cand
            break
    if rna_src is None and not rna_raw.exists():
        _download(RNASEQ_URL, EXTERNAL_DIR / RNASEQ_NAME, log)
        rna_src = EXTERNAL_DIR / RNASEQ_NAME
    if not rna_raw.exists():
        src = rna_src if rna_src is not None else EXTERNAL_DIR / RNASEQ_NAME
        shutil.copy2(src, rna_raw)
        log.log(f"Staged RNAseq raw from {src} -> {rna_raw}")
    else:
        log.log(f"Using existing RNAseq raw: {rna_raw}")

    log.log(f"Clinical sha256={sha256_file(clin_raw)}")
    log.log(f"RNAseq sha256={sha256_file(rna_raw)}")
    return clin_raw, rna_raw


def normalize_subtype(x: object) -> str:
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    mapping = {
        "Luminal A": "LumA",
        "LumA": "LumA",
        "Luminal B": "LumB",
        "LumB": "LumB",
        "Her2": "Her2",
        "HER2": "Her2",
        "Basal": "Basal",
        "Normal-like": "Normal-like",
        "Normal": "Normal-like",
    }
    return mapping.get(s, s)


def build_source_table(clin_path: Path, rna_path: Path, log: PrepLogger) -> pd.DataFrame:
    """
    Build a discovery-like table: rows=samples, columns=PAM50 genes + PAM50.
    Index = native LinkedOmics sample IDs.
    """
    clin = pd.read_csv(clin_path, sep="\t", index_col=0)
    drop_idx = [i for i in clin.index if str(i).upper() in {"IDX", "ATTRIBUTE_TYPE", "TYPE"}]
    if drop_idx:
        clin = clin.drop(index=drop_idx)
    if "PAM50" not in clin.columns:
        raise ValueError("Clinical matrix missing PAM50 column.")

    rna = pd.read_csv(rna_path, sep="\t", index_col=0)
    missing_genes = [g for g in PAM50_GENES if g not in rna.index]
    if missing_genes:
        raise RuntimeError(f"Missing PAM50 genes in CPTAC RNAseq: {missing_genes}")

    # genes x samples -> samples x genes
    expr = rna.loc[PAM50_GENES].apply(pd.to_numeric, errors="coerce").T
    # Align samples present in both
    common = expr.index.intersection(clin.index)
    log.log(f"Samples in RNAseq={len(expr)} clinical={len(clin)} intersection={len(common)}")
    expr = expr.loc[common]
    y = clin.loc[common, "PAM50"].map(normalize_subtype)

    out = expr.copy()
    out["PAM50"] = y.to_numpy()
    out.index.name = "sample_id"
    log.log(f"Built source table shape={out.shape}")
    return out


def prepare_matrix(df: pd.DataFrame, log: PrepLogger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Same filtering rules as prepare_discovery.prepare_matrix:
      - preserve native sample IDs
      - exact 50-gene order
      - keep LumA/LumB/Her2/Basal only
      - drop rows with any NA among the 50 genes
      - no expression transform / scaling
    """
    exclusions: list[dict] = []
    work = df.copy()

    sample_ids = pd.Series(work.index.astype(str), name="sample_id").reset_index(drop=True)
    X = work[PAM50_GENES].astype(float).reset_index(drop=True)
    y = work["PAM50"].astype(str).reset_index(drop=True)

    before_counts = y.value_counts().to_dict()
    log.log(f"Class distribution before filtering: {before_counts}")

    keep_mask = y.isin(TARGET_LABELS).reset_index(drop=True)
    for i, sid in enumerate(sample_ids):
        if not keep_mask.iloc[i]:
            exclusions.append(
                {
                    "sample_id": str(sid),
                    "reason": f"subtype_not_in_4class:{y.iloc[i]}",
                    "subtype": str(y.iloc[i]),
                }
            )
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    sample_ids = sample_ids.loc[keep_mask].reset_index(drop=True)

    na_rows = X.isna().any(axis=1).reset_index(drop=True)
    for i in np.where(na_rows.to_numpy())[0]:
        exclusions.append(
            {
                "sample_id": str(sample_ids.iloc[i]),
                "reason": "native_missing_expression",
                "subtype": str(y.iloc[i]),
            }
        )
    X = X.loc[~na_rows].reset_index(drop=True)
    y = y.loc[~na_rows].reset_index(drop=True)
    sample_ids = sample_ids.loc[~na_rows].reset_index(drop=True)

    after_counts = y.value_counts().to_dict()
    log.log(f"Class distribution after filtering: {after_counts}")
    log.log(f"Excluded n={len(exclusions)}")

    final = X.copy()
    final.insert(0, "sample_id", sample_ids.astype(str).to_numpy())
    final["PAM50"] = y.to_numpy()

    prep_meta = {
        "id_status": "preserved_from_source_index",
        "biological_ids_available": True,
        "class_counts_before": {str(k): int(v) for k, v in before_counts.items()},
        "class_counts_after": {str(k): int(v) for k, v in after_counts.items()},
        "n_excluded": len(exclusions),
        "gene_order": list(PAM50_GENES),
        "preprocessing_decisions": [
            "LinkedOmics clinical+RNAseq joined on native sample IDs.",
            "Retained exact 50 PAM50 genes in canonical study order (same as discovery).",
            "No expression transformation applied (values copied as-is from LinkedOmics).",
            "No scaling/centering applied at preparation time.",
            "Rows with labels outside LumA/LumB/Her2/Basal excluded.",
            "Rows with any NA among the 50 genes excluded (same rule as discovery).",
            "Native LinkedOmics sample IDs preserved.",
            "Cohort key: cptac_linkedomics (does not replace discovery).",
        ],
    }
    return final, pd.DataFrame(exclusions), prep_meta


def expression_fingerprint(final: pd.DataFrame) -> dict:
    genes = [c for c in final.columns if c not in {"sample_id", "PAM50"}]
    ordered = final.sort_values("sample_id").reset_index(drop=True)
    values_csv = ordered[genes + ["PAM50"]].to_csv(index=False).encode("utf-8")
    full_csv = ordered.to_csv(index=False).encode("utf-8")
    vals = ordered[genes].to_numpy(dtype=float)
    return {
        "n_samples": int(len(ordered)),
        "n_genes": int(len(genes)),
        "genes": genes,
        "class_distribution": {
            str(k): int(v) for k, v in ordered["PAM50"].value_counts().items()
        },
        "values_and_labels_sha256": sha256_bytes(values_csv),
        "full_matrix_sha256": sha256_bytes(full_csv),
        "expression_summary": {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        },
    }


def write_artifacts(
    final: pd.DataFrame,
    excluded: pd.DataFrame,
    prep_meta: dict,
    clin_path: Path,
    rna_path: Path,
    log: PrepLogger,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUT_DIR / "cptac_linkedomics_pam50_4class.csv"
    final_sorted = final.sort_values("sample_id").reset_index(drop=True)
    final_sorted.to_csv(matrix_path, index=False)
    log.log(f"Wrote {matrix_path}")

    # Determinism check
    again = prepare_matrix(
        build_source_table(clin_path, rna_path, PrepLogger()), PrepLogger()
    )[0].sort_values("sample_id").reset_index(drop=True)
    fp1 = expression_fingerprint(final_sorted)
    fp2 = expression_fingerprint(again)
    if fp1["values_and_labels_sha256"] != fp2["values_and_labels_sha256"]:
        raise RuntimeError("Determinism check FAILED for CPTAC preparation.")
    log.log("Determinism check (repeated preparation) PASSED.")

    excluded_path = OUT_DIR / "cptac_linkedomics_excluded_samples.csv"
    if len(excluded):
        excluded.to_csv(excluded_path, index=False)
    else:
        excluded_path.write_text("", encoding="utf-8")
    log.log(f"Wrote {excluded_path} (n={len(excluded)})")

    fp_path = OUT_DIR / "cptac_linkedomics_fingerprint.json"
    fp_path.write_text(json.dumps(fp1, indent=2), encoding="utf-8")

    inventory = {
        "cohort_key": "cptac_linkedomics",
        "scientific_name_policy": (
            "Public LinkedOmics CPTAC-BRCA prospective reference cohort. "
            "Does not replace the discovery cohort key."
        ),
        "provenance": {
            "confidence": "Confirmed (public LinkedOmics download)",
            "portal": "https://www.linkedomics.org/data_download/CPTAC-BRCA/",
            "clinical_file": str(clin_path),
            "rnaseq_file": str(rna_path),
            "clinical_sha256": sha256_file(clin_path),
            "rnaseq_sha256": sha256_file(rna_path),
            "rnaseq_unit": "log2(FPKM), normalized by gene median",
        },
        "sample_ids": {
            "status": prep_meta["id_status"],
            "biological_ids_available": True,
            "scheme": "LinkedOmics native sample IDs (preserved from source index)",
        },
        "final_matrix": {
            "path": str(matrix_path),
            "schema": list(final_sorted.columns),
            "fingerprint": fp1,
        },
        "exclusions": {
            "path": str(excluded_path),
            "n": int(len(excluded)),
            "class_counts_before": prep_meta["class_counts_before"],
            "class_counts_after": prep_meta["class_counts_after"],
        },
        "preprocessing_decisions": prep_meta["preprocessing_decisions"],
    }
    (OUT_DIR / "cptac_linkedomics_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )

    report = [
        "# CPTAC LinkedOmics preparation report",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "- Cohort key: **`cptac_linkedomics`**",
        "- Source: LinkedOmics CPTAC-BRCA prospective RNAseq + clinical",
        "",
        "## Preparation rules (aligned with discovery)",
        *[f"- {d}" for d in prep_meta["preprocessing_decisions"]],
        "",
        "## Results",
        f"- n after filters: **{fp1['n_samples']}**",
        f"- class distribution: `{fp1['class_distribution']}`",
        f"- excluded: **{len(excluded)}**",
        f"- values+labels sha256: `{fp1['values_and_labels_sha256']}`",
        "",
        "## Note vs discovery",
        "- Discovery keeps n=117 with no native missing values.",
        "- CPTAC LinkedOmics RNAseq has sporadic NAs among PAM50 genes; the shared",
        "  rule (drop rows with any NA) therefore yields fewer than 117 samples.",
        "",
    ]
    (OUT_DIR / "cptac_linkedomics_provenance_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    log.save(OUT_DIR / "cptac_linkedomics_preparation_log.txt")
    return inventory


def main() -> int:
    log = PrepLogger()
    log.log(f"ROOT={ROOT}")
    clin_path, rna_path = stage_raw_sources(log)
    source = build_source_table(clin_path, rna_path, log)
    final, excluded, prep_meta = prepare_matrix(source, log)

    # Assertions analogous to discovery
    genes = [c for c in final.columns if c not in {"sample_id", "PAM50"}]
    assert genes == PAM50_GENES
    assert final[genes].isna().sum().sum() == 0
    assert final["sample_id"].is_unique
    assert set(final["PAM50"]) <= set(TARGET_LABELS)
    assert not final.duplicated(subset=genes + ["PAM50"]).any()
    log.log("All preparation assertions PASSED.")

    inv = write_artifacts(final, excluded, prep_meta, clin_path, rna_path, log)
    log.log("=" * 72)
    log.log("CPTAC LINKEDOMICS PREPARATION COMPLETE")
    log.log(
        f"n={inv['final_matrix']['fingerprint']['n_samples']} "
        f"class_distribution={inv['final_matrix']['fingerprint']['class_distribution']}"
    )
    log.log("STOPPING after preparation (no benchmark).")
    # Append final lines to log file already written; rewrite with latest lines
    log.save(OUT_DIR / "cptac_linkedomics_preparation_log.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
