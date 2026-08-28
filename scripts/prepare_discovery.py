#!/usr/bin/env python3
"""
Discovery-cohort preparation layer.

Treats the laboratory matrix used in the original conference experiment as the
discovery cohort until provenance is conclusively established.

Does NOT rename the cohort to CPTAC in scientific outputs.
Does NOT run imputation/classification benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "discovery"
OUT_DIR = ROOT / "data" / "processed" / "discovery"
ARCHIVE_CANDIDATES = [
    ROOT / "archive" / "legacy_root" / "df_MI_no_missing.csv",
    ROOT / "archive" / "database" / "df_MI_no_missing.csv",
    ROOT / "data" / "processed" / "discovery" / "df_MI_no_missing.csv",
]

# Canonical gene order from the original study matrix / PAM50 panel.
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
EXPECTED_CLASS_COUNTS = {"LumA": 57, "Basal": 29, "LumB": 17, "Her2": 14}


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


def locate_or_stage_raw_source(log: PrepLogger) -> Path:
    """
    Ensure an immutable copy exists under data/raw/discovery/.
    Prefer an existing raw copy; otherwise stage from archive candidates.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "df_MI_no_missing.csv"

    if raw_path.exists():
        log.log(f"Using existing immutable raw source: {raw_path}")
        return raw_path

    for cand in ARCHIVE_CANDIDATES:
        if cand.exists():
            shutil.copy2(cand, raw_path)
            log.log(f"Staged raw source from {cand} -> {raw_path}")
            return raw_path

    raise FileNotFoundError(
        "Could not locate df_MI_no_missing.csv in archive or processed paths."
    )


def search_for_biological_ids(log: PrepLogger) -> dict:
    """
    Attempt to recover original biological sample IDs from local project files.
    Returns a structured search report. Does not invent biological IDs.

    Excludes known other-cohort trees (METABRIC/TCGA raw downloads) so their
    sample IDs are not mistaken for discovery-cohort recovery evidence.
    """
    report = {
        "searched_paths": [],
        "biological_ids_found": False,
        "evidence": [],
        "other_cohort_hits_ignored": [],
        "conclusion": "",
    }

    patterns_checked = []
    search_roots = [
        ROOT / "archive",
        ROOT / "notebooks",
        ROOT / "data" / "raw" / "discovery",
        ROOT / "data" / "processed" / "discovery",
    ]
    interesting_names = (
        "df_mi",
        "pam50",
        "sample",
        "clinical",
        "patient",
        "cptac",
        "discovery",
        "brca",
    )
    # Paths that belong to other cohorts / downloads — not discovery ID sources.
    exclude_path_tokens = (
        "metabric",
        "tcga",
        "data\\raw\\metabric",
        "data/raw/metabric",
        "data\\processed\\metabric",
        "data/processed/metabric",
    )
    # Tokens that indicate other-cohort IDs, not discovery recovery.
    other_cohort_id_tokens = ("MB-", "TCGA-")

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            path_s = str(path).replace("/", "\\").lower()
            if any(tok in path_s for tok in exclude_path_tokens):
                continue
            name_l = path.name.lower()
            if not any(k in name_l for k in interesting_names):
                continue
            if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".ipynb", ".json"}:
                continue
            report["searched_paths"].append(str(path))
            if path.stat().st_size > 50_000_000:
                continue
            try:
                if path.suffix.lower() != ".csv":
                    continue
                df = pd.read_csv(path, nrows=5)
                cols = [str(c).lower() for c in df.columns]
                df2 = pd.read_csv(path, index_col=0, nrows=5)
                idx_vals = [str(v) for v in df2.index.tolist()]
                patterns_checked.append(
                    {"path": str(path), "columns": list(df.columns)[:12], "index_sample": idx_vals}
                )
                other_like = any(
                    any(tok in v for tok in other_cohort_id_tokens) for v in idx_vals
                )
                discovery_like = any(
                    any(tok in v for tok in ("01BR", "X01BR", "CPT", "CPTAC"))
                    for v in idx_vals
                )
                has_id_col = any(
                    c in cols for c in ("sample_id", "sample", "patient_id", "case_id")
                )
                # Synthetic discovery_row_* IDs are not biological recovery.
                synthetic_only = all(str(v).startswith("discovery_row_") for v in idx_vals)
                if other_like and not discovery_like:
                    report["other_cohort_hits_ignored"].append(
                        {"path": str(path), "index_sample": idx_vals}
                    )
                    continue
                if synthetic_only:
                    continue
                if discovery_like or (has_id_col and not synthetic_only and discovery_like):
                    report["biological_ids_found"] = True
                    report["evidence"].append(
                        {
                            "path": str(path),
                            "reason": "discovery_like_index_or_id_column",
                            "index_sample": idx_vals,
                            "columns": list(df.columns)[:20],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                patterns_checked.append({"path": str(path), "error": str(exc)})

    report["patterns_checked_sample"] = patterns_checked[:30]
    if report["biological_ids_found"]:
        report["conclusion"] = (
            "Potential discovery-like biological IDs found in ancillary files; "
            "manual review required before adopting them for the discovery matrix."
        )
    else:
        report["conclusion"] = (
            "No biological sample IDs recovered from local notebooks/CSVs/archives "
            "for the discovery matrix. Available source uses a reset integer index "
            "(0..n-1). Other-cohort IDs (e.g. METABRIC MB-*) were ignored."
        )
    log.log(report["conclusion"])
    return report


def read_source_matrix(path: Path, log: PrepLogger) -> pd.DataFrame:
    # Notebook historically used index_col=0 (unnamed integer index).
    df = pd.read_csv(path, index_col=0)
    log.log(f"Loaded source matrix shape={df.shape} from {path}")
    if "PAM50" not in df.columns:
        raise ValueError("Source matrix missing PAM50 column.")
    return df


def prepare_matrix(df: pd.DataFrame, log: PrepLogger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build discovery_pam50_4class.csv schema.
    Returns (final_df, excluded_df, prep_meta).
    """
    exclusions: list[dict] = []
    work = df.copy()

    # Detect whether biological IDs exist
    index_vals = [str(v) for v in work.index.tolist()]
    looks_biological = any(
        any(tok in v for tok in ("MB-", "TCGA-", "01BR", "X01BR", "CPT", "SAMPLE"))
        for v in index_vals
    )
    is_range = list(work.index) == list(range(len(work)))

    if "sample_id" in work.columns:
        sample_ids = work["sample_id"].astype(str)
        work = work.drop(columns=["sample_id"])
        id_status = "native_sample_id_column"
        biological_ids_available = True
    elif looks_biological and not is_range:
        sample_ids = pd.Series(index_vals, name="sample_id")
        id_status = "preserved_from_source_index"
        biological_ids_available = True
    else:
        # Synthetic IDs only — do not imply biological identity.
        sample_ids = pd.Series(
            [f"discovery_row_{i:04d}" for i in range(1, len(work) + 1)],
            name="sample_id",
        )
        id_status = "synthetic_row_ids"
        biological_ids_available = False
        log.log(
            "Original biological sample IDs unavailable; "
            "assigned synthetic IDs discovery_row_0001.."
        )

    # Gene order
    missing_genes = [g for g in PAM50_GENES if g not in work.columns]
    if missing_genes:
        raise RuntimeError(f"Missing PAM50 genes in source: {missing_genes}")
    X = work[PAM50_GENES].astype(float)
    y = work["PAM50"].astype(str)

    # Class filter — already expected to be 4-class; record any exclusions
    before_counts = y.value_counts().to_dict()
    log.log(f"Class distribution before filtering: {before_counts}")

    keep_mask = y.isin(TARGET_LABELS)
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

    # Native missing expression
    na_rows = X.isna().any(axis=1)
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

    final = X.copy()
    final.insert(0, "sample_id", sample_ids.astype(str).to_numpy())
    final["PAM50"] = y.to_numpy()

    prep_meta = {
        "id_status": id_status,
        "biological_ids_available": biological_ids_available,
        "class_counts_before": {str(k): int(v) for k, v in before_counts.items()},
        "class_counts_after": {str(k): int(v) for k, v in after_counts.items()},
        "n_excluded": len(exclusions),
        "gene_order": list(PAM50_GENES),
        "preprocessing_decisions": [
            "Source CSV read with index_col=0 (matches original notebook).",
            "Retained exact 50 PAM50 genes in canonical study order.",
            "No expression transformation applied (values copied as-is).",
            "No scaling/centering applied at preparation time.",
            "Rows with labels outside LumA/LumB/Her2/Basal excluded if present.",
            "Rows with any NA among the 50 genes excluded if present.",
            "Synthetic IDs used only when biological IDs are unavailable.",
            "Cohort key remains 'discovery' (not renamed to CPTAC).",
        ],
    }
    return final, pd.DataFrame(exclusions), prep_meta


def expression_fingerprint(final: pd.DataFrame) -> dict:
    genes = [c for c in final.columns if c not in {"sample_id", "PAM50"}]
    ordered = final.sort_values("sample_id").reset_index(drop=True)
    # Value fingerprint ignores synthetic IDs so ID policy changes don't alter value hash.
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
            "std": float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
        },
    }


def assert_preparation(
    final: pd.DataFrame,
    source_df: pd.DataFrame,
    prep_meta: dict,
    log: PrepLogger,
) -> None:
    genes = PAM50_GENES
    errors: list[str] = []

    if list(final.columns) != ["sample_id", *genes, "PAM50"]:
        errors.append("Unexpected column order/schema in final matrix.")
    if len(genes) != 50 or len(set(genes)) != 50:
        errors.append("PAM50 gene list invalid.")
    if final["sample_id"].duplicated().any():
        errors.append("Duplicated sample_id.")
    if final.duplicated(subset=genes + ["PAM50"]).any():
        # soft note: identical expression+label rows
        log.log("NOTE: found duplicated expression+label rows (kept; not ID duplicates).")
    if set(final["PAM50"].unique()) != set(TARGET_LABELS):
        errors.append(f"Unexpected labels: {sorted(final['PAM50'].unique())}")
    if final[genes].isna().any().any():
        errors.append("Missing expression values remain.")

    # Class counts must match original experiment
    counts = final["PAM50"].value_counts().to_dict()
    for k, v in EXPECTED_CLASS_COUNTS.items():
        if counts.get(k) != v:
            errors.append(f"Class count mismatch for {k}: got {counts.get(k)}, expected {v}")

    if len(final) != 117:
        errors.append(f"Expected 117 samples, got {len(final)}")

    # Expression values must match source after gene/label alignment
    src = source_df.copy()
    src_y = src["PAM50"].astype(str)
    src_X = src[genes].astype(float)
    keep = src_y.isin(TARGET_LABELS)
    src_X = src_X.loc[keep]
    src_y = src_y.loc[keep]
    # Compare in original row order (before synthetic ID sort)
    migrated_X = final[genes].to_numpy(dtype=float)
    source_X = src_X.to_numpy(dtype=float)
    if migrated_X.shape != source_X.shape:
        errors.append(
            f"Shape mismatch vs source after filter: {migrated_X.shape} vs {source_X.shape}"
        )
    else:
        max_abs = float(np.max(np.abs(migrated_X - source_X)))
        if max_abs > 0:
            errors.append(f"Expression values differ from source; max_abs_diff={max_abs}")
        else:
            log.log("Expression values match source exactly (max_abs_diff=0).")
        if not np.array_equal(final["PAM50"].to_numpy(), src_y.to_numpy()):
            errors.append("Label vector order/content differs from filtered source.")

    if errors:
        raise AssertionError("Discovery preparation assertions failed:\n- " + "\n- ".join(errors))
    log.log("All preparation assertions PASSED.")


def write_provenance_md(
    path: Path,
    *,
    inventory: dict,
    id_search: dict,
    fingerprint: dict,
) -> None:
    lines = [
        "# Discovery cohort provenance report",
        "",
        f"- Generated (UTC): `{inventory['generated_at_utc']}`",
        f"- Cohort key: **`discovery`** (not renamed to CPTAC)",
        f"- Provenance confidence: **{inventory['provenance']['confidence']}**",
        "",
        "## Source",
        f"- Raw immutable path: `{inventory['source_files']['raw_matrix']['path']}`",
        f"- SHA256: `{inventory['source_files']['raw_matrix']['sha256']}`",
        f"- Notebook historical path hint: "
        "`/content/drive/MyDrive/Database/brca_metabric/df_MI_no_missing.csv`",
        "",
        "## Sample ID status",
        f"- Status: **{inventory['sample_ids']['status']}**",
        f"- Biological IDs available: **{inventory['sample_ids']['biological_ids_available']}**",
        f"- Scheme: `{inventory['sample_ids']['scheme']}`",
        f"- ID search conclusion: {id_search.get('conclusion')}",
        "",
        "## Scientific naming policy",
        "- Use **discovery cohort** / **laboratory cohort** in paper text until provenance is closed.",
        "- Do **not** label outputs as CPTAC unless sample IDs / source documentation confirm it.",
        "",
        "## Evidence summary",
        "- Matrix has 117 samples and PAM50 4-class counts "
        "(LumA 57, Basal 29, LumB 17, Her2 14), which match CPTAC prospective "
        "proportions after dropping Normal-like — **circumstantial only**.",
        "- Local files retain a reset integer index; no biological IDs recovered.",
        "",
        "## Fingerprint",
        f"- values+labels SHA256: `{fingerprint['values_and_labels_sha256']}`",
        f"- full matrix SHA256: `{fingerprint['full_matrix_sha256']}`",
        "",
        "## Preprocessing decisions",
    ]
    for d in inventory["preprocessing_decisions"]:
        lines.append(f"- {d}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    log = PrepLogger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.log(f"ROOT={ROOT}")

    raw_path = locate_or_stage_raw_source(log)
    raw_sha = sha256_file(raw_path)
    log.log(f"Raw source sha256={raw_sha}")

    id_search = search_for_biological_ids(log)
    source_df = read_source_matrix(raw_path, log)
    final, exclusions, prep_meta = prepare_matrix(source_df, log)

    assert_preparation(final, source_df, prep_meta, log)

    # Deterministic write order by synthetic/native sample_id
    final_ordered = final.sort_values("sample_id").reset_index(drop=True)
    # BUT value parity vs source used unsorted filtered order in assert_preparation.
    # For the saved artifact, keep a stable sort by sample_id; also store row_order map.
    # Re-assert values after aligning by original order index embedded in synthetic IDs.
    out_csv = OUT_DIR / "discovery_pam50_4class.csv"
    final_ordered.to_csv(out_csv, index=False)
    log.log(f"Wrote {out_csv}")

    # Verify round-trip determinism + value integrity vs source using ID order
    reloaded = pd.read_csv(out_csv)
    fp1 = expression_fingerprint(final_ordered)
    fp2 = expression_fingerprint(reloaded)
    if fp1["values_and_labels_sha256"] != fp2["values_and_labels_sha256"]:
        raise AssertionError("Determinism failure: reload fingerprint mismatch.")
    log.log("Determinism check (CSV round-trip) PASSED.")

    # Second in-memory prep for determinism of pipeline
    final_b, _, _ = prepare_matrix(source_df, PrepLogger())
    final_b_ordered = final_b.sort_values("sample_id").reset_index(drop=True)
    fp3 = expression_fingerprint(final_b_ordered)
    if fp3["values_and_labels_sha256"] != fp1["values_and_labels_sha256"]:
        raise AssertionError("Determinism failure: repeated preparation mismatch.")
    log.log("Determinism check (repeated preparation) PASSED.")

    # Value match vs source when both sorted by original row number extracted from synthetic IDs
    # discovery_row_0001 corresponds to source row position 0 after filter (1-based).
    genes = PAM50_GENES
    src_keep = source_df.loc[source_df["PAM50"].isin(TARGET_LABELS), genes].astype(float)
    # Map synthetic IDs back to original positions
    if prep_meta["id_status"] == "synthetic_row_ids":
        # After filter with no exclusions, row_0001.. map to filtered order 0..n-1
        # Sorted artifact order differs; compare via merge on reconstructed order index.
        tmp = final_ordered.copy()
        tmp["_ord"] = tmp["sample_id"].str.replace("discovery_row_", "", regex=False).astype(int) - 1
        tmp = tmp.sort_values("_ord")
        max_abs = float(
            np.max(np.abs(tmp[genes].to_numpy(dtype=float) - src_keep.to_numpy(dtype=float)))
        )
        if max_abs > 0:
            raise AssertionError(f"Post-sort value parity failed: max_abs={max_abs}")
        log.log("Post-sort expression parity vs source PASSED (max_abs_diff=0).")

    excl_path = OUT_DIR / "discovery_excluded_samples.csv"
    exclusions.to_csv(excl_path, index=False)
    log.log(f"Wrote {excl_path} (n={len(exclusions)})")

    fingerprint = fp1
    fp_path = OUT_DIR / "discovery_fingerprint.json"
    fp_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    provenance_confidence = "Low"
    # Moderate if class counts match CPTAC circumstantial evidence AND file lineage clear
    if fingerprint["class_distribution"] == EXPECTED_CLASS_COUNTS:
        provenance_confidence = "Moderate"
    if prep_meta["biological_ids_available"]:
        provenance_confidence = "High"

    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_key": "discovery",
        "scientific_name_policy": (
            "Refer to this cohort as the discovery/laboratory cohort. "
            "Do not label as CPTAC unless provenance is confirmed."
        ),
        "provenance": {
            "confidence": provenance_confidence,
            "hypotheses": [
                {
                    "label": "CPTAC_prospective_minus_normal_like",
                    "support": "circumstantial_class_counts",
                    "status": "unconfirmed",
                },
                {
                    "label": "METABRIC_subset",
                    "support": "historical_drive_folder_name_only",
                    "status": "unlikely_unconfirmed",
                },
            ],
            "id_recovery_search": id_search,
        },
        "source_files": {
            "raw_matrix": {
                "path": str(raw_path.resolve()),
                "sha256": raw_sha,
                "bytes": raw_path.stat().st_size,
                "immutable_policy": "Do not modify files under data/raw/discovery/.",
            }
        },
        "sample_ids": {
            "status": prep_meta["id_status"],
            "biological_ids_available": prep_meta["biological_ids_available"],
            "scheme": (
                "discovery_row_XXXX (1-based synthetic)"
                if prep_meta["id_status"] == "synthetic_row_ids"
                else prep_meta["id_status"]
            ),
        },
        "filtering": {
            "n_source_rows": int(len(source_df)),
            "n_final_samples": int(len(final_ordered)),
            "n_excluded": int(len(exclusions)),
            "class_counts_before": prep_meta["class_counts_before"],
            "class_counts_after": prep_meta["class_counts_after"],
            "expected_class_counts": EXPECTED_CLASS_COUNTS,
        },
        "preprocessing_decisions": prep_meta["preprocessing_decisions"],
        "final_matrix": {
            "path": str(out_csv.resolve()),
            "schema": ["sample_id", *PAM50_GENES, "PAM50"],
            "fingerprint": fingerprint,
        },
        "artifacts": {
            "discovery_inventory.json": str(OUT_DIR / "discovery_inventory.json"),
            "discovery_provenance_report.md": str(OUT_DIR / "discovery_provenance_report.md"),
            "discovery_preparation_log.txt": str(OUT_DIR / "discovery_preparation_log.txt"),
            "discovery_excluded_samples.csv": str(excl_path),
            "discovery_pam50_4class.csv": str(out_csv),
            "discovery_fingerprint.json": str(fp_path),
        },
    }

    inv_path = OUT_DIR / "discovery_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    write_provenance_md(
        OUT_DIR / "discovery_provenance_report.md",
        inventory=inventory,
        id_search=id_search,
        fingerprint=fingerprint,
    )

    log.log("=" * 72)
    log.log("DISCOVERY PREPARATION COMPLETE")
    log.log(f"n={len(final_ordered)} class_distribution={fingerprint['class_distribution']}")
    log.log(f"id_status={prep_meta['id_status']}")
    log.log(f"provenance_confidence={provenance_confidence}")
    log.log(f"values_and_labels_sha256={fingerprint['values_and_labels_sha256']}")
    log.log("STOPPING after preparation (no benchmark).")
    log.save(OUT_DIR / "discovery_preparation_log.txt")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
