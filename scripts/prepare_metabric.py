#!/usr/bin/env python3
"""
METABRIC preparation layer for PAM50 4-class independent benchmarking.

Discovers expression profiles from cBioPortal metadata, maps PAM50 genes via
Entrez ID (aliases as fallback), reports clinical subtype fields without
assuming CLAUDIN_SUBTYPE == pure PAM50, and writes reproducible artifacts.

Does NOT run imputation or classification experiments.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "metabric"
OUT_DIR = ROOT / "data" / "processed" / "metabric"

# Target HUGO symbols matching the discovery cohort schema (modern aliases).
PAM50_TARGET_GENES: list[str] = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]

# Stable Entrez Gene IDs for PAM50 (Parker et al. / NCBI; documented alias updates).
# Source notes: CDCA1->NUF2 (83540), KNTC2->NDC80 (10403), ORC6L->ORC6 (23594).
PAM50_ENTREZ: dict[str, int] = {
    "ACTR3B": 57180, "ANLN": 54443, "BAG1": 573, "BCL2": 596, "BIRC5": 332,
    "BLVRA": 644, "CCNB1": 891, "CCNE1": 898, "CDC20": 991, "CDC6": 990,
    "NUF2": 83540, "CDH3": 1001, "CENPF": 1063, "CEP55": 55165, "CXXC5": 51523,
    "EGFR": 1956, "ERBB2": 2064, "ESR1": 2099, "EXO1": 9156, "FGFR4": 2264,
    "FOXA1": 3169, "FOXC1": 2296, "GPR160": 26996, "GRB7": 2886, "KIF2C": 11004,
    "NDC80": 10403, "KRT14": 3861, "KRT17": 3872, "KRT5": 3852, "MAPT": 4137,
    "MDM2": 4193, "MELK": 9833, "MIA": 8190, "MKI67": 4288, "MLPH": 79083,
    "MMP11": 4320, "MYBL2": 4605, "MYC": 4609, "NAT1": 9, "ORC6": 23594,
    "PGR": 5241, "PHGDH": 26227, "PTTG1": 9232, "RRM2": 6241, "SFRP1": 6422,
    "SLC39A6": 25800, "TMEM45B": 120224, "TYMS": 7298, "UBE2C": 11065,
    "UBE2T": 29089,
}

# Documented symbol aliases (fallback only when Entrez match fails).
PAM50_SYMBOL_ALIASES: dict[str, list[str]] = {
    "NUF2": ["CDCA1"],
    "NDC80": ["KNTC2"],
    "ORC6": ["ORC6L"],
}

# Four malignant subtypes used in the discovery cohort protocol.
TARGET_SUBTYPES = ("LumA", "LumB", "Her2", "Basal")
# Labels present in METABRIC "Pam50 + Claudin-low" field that are NOT in the 4-class set.
EXCLUDED_SUBTYPE_LABELS = ("Normal", "claudin-low", "NC")

# Deterministic probe collapse rule.
PROBE_SELECTION_RULE = (
    "Among rows mapping to the same target gene, select the probe/row with the "
    "highest variance across all expression samples (unsupervised, deterministic; "
    "ties broken by smaller Entrez_Gene_Id then lexicographic Hugo_Symbol)."
)


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


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_meta_file(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def discover_expression_profiles(raw_dir: Path, log: PrepLogger) -> list[dict]:
    """Discover mRNA expression profiles from meta_*.txt files (not fixed filenames)."""
    profiles: list[dict] = []
    for meta_path in sorted(raw_dir.glob("meta_*.txt")):
        if meta_path.name in {"meta_study.txt", "meta_clinical_patient.txt", "meta_clinical_sample.txt"}:
            continue
        meta = parse_meta_file(meta_path)
        gat = meta.get("genetic_alteration_type", "").upper()
        if "MRNA" not in gat and "EXPRESSION" not in gat:
            continue
        data_filename = meta.get("data_filename")
        if not data_filename:
            log.log(f"SKIP meta without data_filename: {meta_path.name}")
            continue
        data_path = raw_dir / data_filename
        profiles.append(
            {
                "meta_file": meta_path.name,
                "data_filename": data_filename,
                "data_path": str(data_path),
                "data_exists": data_path.is_file(),
                "genetic_alteration_type": meta.get("genetic_alteration_type"),
                "datatype": meta.get("datatype"),
                "stable_id": meta.get("stable_id"),
                "profile_name": meta.get("profile_name"),
                "profile_description": meta.get("profile_description"),
                "cancer_study_identifier": meta.get("cancer_study_identifier"),
                "show_profile_in_analysis_tab": meta.get("show_profile_in_analysis_tab"),
            }
        )
        log.log(
            f"Discovered expression profile via {meta_path.name}: "
            f"datatype={meta.get('datatype')!r}, file={data_filename}, exists={data_path.is_file()}"
        )
    return profiles


def select_expression_profile(profiles: list[dict], log: PrepLogger) -> dict:
    """
    Prefer CONTINUOUS Illumina microarray log-intensity over Z-SCORE profiles.
    Never claim 'raw' unless the source metadata says so.
    """
    available = [p for p in profiles if p["data_exists"]]
    if not available:
        raise FileNotFoundError(
            "No expression data files found for discovered mRNA profiles. "
            f"Profiles seen: {[p['data_filename'] for p in profiles]}"
        )

    continuous = [
        p for p in available
        if str(p.get("datatype", "")).upper() == "CONTINUOUS"
    ]
    if continuous:
        # Prefer Illumina HT-12 microarray continuous log intensity if present.
        illumina = [
            p for p in continuous
            if "illumina" in (p.get("profile_name") or "").lower()
            or "illumina" in (p.get("profile_description") or "").lower()
            or "illumina" in p["data_filename"].lower()
        ]
        chosen = illumina[0] if illumina else continuous[0]
    else:
        chosen = available[0]
        log.log(
            "WARNING: no CONTINUOUS datatype profile available; "
            f"falling back to {chosen['data_filename']} "
            f"(datatype={chosen.get('datatype')!r})"
        )

    log.log(
        f"Selected expression profile: {chosen['data_filename']} "
        f"| datatype={chosen.get('datatype')!r} "
        f"| name={chosen.get('profile_name')!r}"
    )
    return chosen


def read_cbioportal_clinical(path: Path) -> pd.DataFrame:
    """Read cBioPortal clinical TSV, skipping # comment header lines."""
    return pd.read_csv(path, sep="\t", comment="#", dtype=str)


def is_subtype_candidate(column: str) -> bool:
    tokens = ("SUBTYPE", "PAM50", "CLAUDIN", "INTCLUST", "MOLECULAR", "THREEGENE", "3-GENE", "GENE_CLASSIFIER")
    up = column.upper()
    return any(t in up for t in tokens)


def build_clinical_field_report(
    patient: pd.DataFrame,
    sample: pd.DataFrame,
    log: PrepLogger,
) -> pd.DataFrame:
    rows: list[dict] = []
    for source, df in (("patient", patient), ("sample", sample)):
        for col in df.columns:
            if not is_subtype_candidate(col):
                continue
            series = df[col]
            vc = series.fillna("__MISSING__").value_counts(dropna=False)
            n_missing = int(series.isna().sum() + (series == "").sum())
            # Treat empty strings as missing for rate.
            nonempty = series.replace("", np.nan)
            n_missing = int(nonempty.isna().sum())
            rows.append(
                {
                    "source_table": source,
                    "column_name": col,
                    "n_rows": int(len(series)),
                    "n_unique_non_missing": int(nonempty.dropna().nunique()),
                    "n_missing": n_missing,
                    "missing_rate": float(n_missing / len(series)) if len(series) else np.nan,
                    "unique_values": " | ".join(sorted(nonempty.dropna().astype(str).unique())),
                    "value_counts_json": json.dumps(
                        {str(k): int(v) for k, v in vc.items()},
                        ensure_ascii=True,
                    ),
                    "notes": (
                        "Field display name in clinical header may be "
                        "'Pam50 + Claudin-low subtype'; do NOT equate to pure PAM50."
                        if col == "CLAUDIN_SUBTYPE"
                        else ""
                    ),
                }
            )
            log.log(
                f"Clinical subtype candidate [{source}.{col}]: "
                f"unique={rows[-1]['n_unique_non_missing']}, missing={n_missing}"
            )
    report = pd.DataFrame(rows)
    if report.empty:
        raise RuntimeError("No clinical subtype-candidate fields found.")
    return report


def choose_subtype_field(clinical_report: pd.DataFrame, log: PrepLogger) -> tuple[str, str]:
    """
    Choose a subtype field for 4-class filtering with explicit documentation.
    Prefer CLAUDIN_SUBTYPE only because it contains Pam50+Claudin-low labels
    commonly used for METABRIC intrinsic-subtype analyses — NOT because it is
    pure PAM50.
    """
    # Prefer patient.CLAUDIN_SUBTYPE if present.
    hit = clinical_report[
        (clinical_report["source_table"] == "patient")
        & (clinical_report["column_name"] == "CLAUDIN_SUBTYPE")
    ]
    if not hit.empty:
        log.log(
            "Selected subtype field: patient.CLAUDIN_SUBTYPE "
            "(cBioPortal label: 'Pam50 + Claudin-low subtype'). "
            "This is NOT automatically treated as pure PAM50; "
            "4-class filter keeps LumA/LumB/Her2/Basal and excludes "
            "claudin-low/Normal/NC."
        )
        return "patient", "CLAUDIN_SUBTYPE"

    # Fallback: any field whose unique values cover the 4 target labels.
    for _, row in clinical_report.iterrows():
        vals = set(str(v) for v in row["unique_values"].split(" | ") if v)
        if set(TARGET_SUBTYPES).issubset(vals):
            log.log(
                f"Selected fallback subtype field: {row['source_table']}.{row['column_name']}"
            )
            return str(row["source_table"]), str(row["column_name"])

    raise RuntimeError(
        "Could not identify a clinical field containing LumA/LumB/Her2/Basal labels."
    )


def load_expression_matrix(path: Path, log: PrepLogger) -> pd.DataFrame:
    log.log(f"Loading expression matrix: {path} ({path.stat().st_size} bytes)")
    expr = pd.read_csv(path, sep="\t", dtype={"Hugo_Symbol": str})
    if "Hugo_Symbol" not in expr.columns:
        raise ValueError(f"Expression file missing Hugo_Symbol: {path}")
    if "Entrez_Gene_Id" not in expr.columns:
        raise ValueError(f"Expression file missing Entrez_Gene_Id: {path}")
    # Coerce Entrez to numeric (may contain blanks).
    expr["Entrez_Gene_Id"] = pd.to_numeric(expr["Entrez_Gene_Id"], errors="coerce")
    sample_cols = [c for c in expr.columns if c not in {"Hugo_Symbol", "Entrez_Gene_Id"}]
    log.log(
        f"Expression shape: {expr.shape[0]} rows x {len(sample_cols)} samples "
        f"(plus Hugo_Symbol, Entrez_Gene_Id)"
    )
    return expr


def map_pam50_genes(expr: pd.DataFrame, log: PrepLogger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map each target PAM50 gene primarily by Entrez ID; fallback to documented aliases.
    Collapse multi-probe genes by max variance (deterministic).
    Returns (selected_rows_wide_ready, gene_mapping_report).
    """
    sample_cols = [c for c in expr.columns if c not in {"Hugo_Symbol", "Entrez_Gene_Id"}]
    # Precompute variance per row across samples (unsupervised).
    values = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    row_var = values.var(axis=1, skipna=True, ddof=1)

    report_rows: list[dict] = []
    selected_vectors: dict[str, pd.Series] = {}

    for gene in PAM50_TARGET_GENES:
        entrez = PAM50_ENTREZ[gene]
        aliases = PAM50_SYMBOL_ALIASES.get(gene, [])

        by_entrez = expr.index[expr["Entrez_Gene_Id"] == entrez].tolist()
        match_method = None
        candidate_idx: list[int] = []

        if by_entrez:
            match_method = "entrez"
            candidate_idx = by_entrez
        else:
            # Fallback: exact Hugo match on target or documented aliases.
            symbols = [gene] + aliases
            by_symbol = expr.index[expr["Hugo_Symbol"].isin(symbols)].tolist()
            if by_symbol:
                match_method = "symbol_alias_fallback"
                candidate_idx = by_symbol
            else:
                match_method = "unmatched"
                candidate_idx = []

        candidate_info = []
        for idx in candidate_idx:
            candidate_info.append(
                {
                    "row_index": int(idx),
                    "Hugo_Symbol": str(expr.at[idx, "Hugo_Symbol"]),
                    "Entrez_Gene_Id": (
                        None
                        if pd.isna(expr.at[idx, "Entrez_Gene_Id"])
                        else int(expr.at[idx, "Entrez_Gene_Id"])
                    ),
                    "variance": float(row_var.loc[idx]) if pd.notna(row_var.loc[idx]) else np.nan,
                }
            )

        selected_probe = None
        selected_variance = np.nan
        selected_entrez = None
        selected_hugo = None
        status = "mapped"

        if not candidate_idx:
            status = "unmatched"
            log.log(f"GENE UNMATCHED: {gene} (Entrez {entrez}, aliases={aliases})")
        else:
            # Deterministic selection: max variance; ties -> smaller Entrez; then Hugo.
            def sort_key(info: dict):
                var = info["variance"]
                var_key = -np.inf if (var is None or (isinstance(var, float) and np.isnan(var))) else -float(var)
                ent = info["Entrez_Gene_Id"]
                ent_key = ent if ent is not None else 10**18
                return (var_key, ent_key, info["Hugo_Symbol"], info["row_index"])

            ordered = sorted(candidate_info, key=sort_key)
            best = ordered[0]
            selected_idx = best["row_index"]
            selected_probe = f"row:{selected_idx}|Hugo:{best['Hugo_Symbol']}|Entrez:{best['Entrez_Gene_Id']}"
            selected_variance = best["variance"]
            selected_entrez = best["Entrez_Gene_Id"]
            selected_hugo = best["Hugo_Symbol"]
            selected_vectors[gene] = values.loc[selected_idx]
            if len(candidate_idx) > 1:
                log.log(
                    f"MULTI-PROBE {gene}: {len(candidate_idx)} candidates; "
                    f"selected {selected_probe} var={selected_variance:.6g}"
                )

        report_rows.append(
            {
                "target_gene": gene,
                "target_entrez_id": entrez,
                "documented_aliases": "|".join(aliases) if aliases else "",
                "match_method": match_method,
                "n_candidate_probes": len(candidate_idx),
                "candidate_probes_json": json.dumps(candidate_info, ensure_ascii=True),
                "selected_probe_id": selected_probe,
                "selected_hugo_symbol": selected_hugo,
                "selected_entrez_id": selected_entrez,
                "selected_variance": selected_variance,
                "probe_selection_rule": PROBE_SELECTION_RULE if candidate_idx else "",
                "status": status,
            }
        )

    mapping_report = pd.DataFrame(report_rows)
    unmatched = mapping_report.loc[mapping_report["status"] == "unmatched", "target_gene"].tolist()
    if unmatched:
        raise RuntimeError(f"Failed to map PAM50 genes: {unmatched}")

    # Build samples x genes matrix with original sample IDs as index.
    mat = pd.DataFrame({g: selected_vectors[g] for g in PAM50_TARGET_GENES})
    mat.index.name = "sample_id"
    log.log(f"Mapped expression matrix: {mat.shape[0]} samples x {mat.shape[1]} genes")
    return mat, mapping_report


def normalize_subtype_label(label: str) -> str | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    s = str(label).strip()
    if s == "" or s.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    # Preserve known labels; map common synonyms carefully.
    mapping = {
        "luma": "LumA",
        "luminal a": "LumA",
        "lumb": "LumB",
        "luminal b": "LumB",
        "her2": "Her2",
        "her2-enriched": "Her2",
        "her2enriched": "Her2",
        "basal": "Basal",
        "basal-like": "Basal",
        "normal": "Normal",
        "normal-like": "Normal",
        "claudin-low": "claudin-low",
        "nc": "NC",
    }
    key = s.lower()
    return mapping.get(key, s)


def attach_subtypes(
    expr_mat: pd.DataFrame,
    patient: pd.DataFrame,
    sample: pd.DataFrame,
    source_table: str,
    subtype_col: str,
    log: PrepLogger,
) -> pd.DataFrame:
    if source_table == "patient":
        clin = patient[["PATIENT_ID", subtype_col]].copy()
        # METABRIC sample IDs often equal patient IDs (MB-####).
        clin = clin.rename(columns={"PATIENT_ID": "sample_id", subtype_col: "subtype_raw"})
    else:
        if "SAMPLE_ID" not in sample.columns:
            raise RuntimeError("sample clinical table missing SAMPLE_ID")
        clin = sample[["SAMPLE_ID", subtype_col]].copy()
        clin = clin.rename(columns={"SAMPLE_ID": "sample_id", subtype_col: "subtype_raw"})

    clin["sample_id"] = clin["sample_id"].astype(str)
    clin = clin.drop_duplicates(subset=["sample_id"], keep="first")

    out = expr_mat.copy()
    out = out.join(clin.set_index("sample_id"), how="left")
    out["subtype_norm"] = out["subtype_raw"].map(normalize_subtype_label)
    n_with = int(out["subtype_norm"].notna().sum())
    log.log(f"Samples with non-missing subtype after join: {n_with}/{len(out)}")
    return out


def missingness_report(
    df: pd.DataFrame,
    gene_cols: list[str],
    log: PrepLogger,
) -> pd.DataFrame:
    """Report native missingness by gene, sample, and subtype BEFORE filtering."""
    rows: list[dict] = []

    # By gene
    for g in gene_cols:
        n_miss = int(df[g].isna().sum())
        rows.append(
            {
                "level": "gene",
                "key": g,
                "n_missing": n_miss,
                "n_total": int(len(df)),
                "missing_rate": float(n_miss / len(df)) if len(df) else np.nan,
                "subtype": "",
            }
        )

    # By sample
    sample_miss = df[gene_cols].isna().sum(axis=1)
    for sid, n_miss in sample_miss.items():
        rows.append(
            {
                "level": "sample",
                "key": str(sid),
                "n_missing": int(n_miss),
                "n_total": int(len(gene_cols)),
                "missing_rate": float(n_miss / len(gene_cols)),
                "subtype": (
                    str(df.at[sid, "subtype_norm"])
                    if "subtype_norm" in df.columns and pd.notna(df.at[sid, "subtype_norm"])
                    else ""
                ),
            }
        )

    # By subtype (among samples with a subtype label)
    if "subtype_norm" in df.columns:
        for subtype, subdf in df.groupby(df["subtype_norm"].fillna("__MISSING__"), dropna=False):
            total_cells = int(subdf[gene_cols].size)
            n_miss = int(subdf[gene_cols].isna().sum().sum())
            rows.append(
                {
                    "level": "subtype",
                    "key": str(subtype),
                    "n_missing": n_miss,
                    "n_total": total_cells,
                    "missing_rate": float(n_miss / total_cells) if total_cells else np.nan,
                    "subtype": str(subtype),
                }
            )

    report = pd.DataFrame(rows)
    gene_any = int((df[gene_cols].isna().sum() > 0).sum())
    sample_any = int((sample_miss > 0).sum())
    log.log(
        f"Native missingness summary: genes_with_any_NA={gene_any}, "
        f"samples_with_any_NA={sample_any}, "
        f"total_NA_cells={int(df[gene_cols].isna().sum().sum())}"
    )
    return report


def filter_to_benchmark_matrix(
    df: pd.DataFrame,
    gene_cols: list[str],
    log: PrepLogger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply filters and return (final_matrix, exclusion_log).
    Class distributions before/after are logged.
    """
    exclusions: list[dict] = []

    def class_counts(frame: pd.DataFrame, col: str = "subtype_norm") -> dict:
        if col not in frame.columns:
            return {}
        return {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).items()}

    before_all = class_counts(df)
    log.log(f"Class distribution BEFORE any filtering: {before_all}")

    # 1) Drop samples without subtype
    mask_no_subtype = df["subtype_norm"].isna()
    for sid in df.index[mask_no_subtype]:
        exclusions.append(
            {
                "sample_id": str(sid),
                "reason": "missing_subtype_label",
                "subtype_raw": str(df.at[sid, "subtype_raw"]) if pd.notna(df.at[sid, "subtype_raw"]) else "",
                "subtype_norm": "",
            }
        )
    step1 = df.loc[~mask_no_subtype].copy()
    log.log(f"After dropping missing subtype: n={len(step1)}; counts={class_counts(step1)}")

    # 2) Keep only 4 target subtypes
    mask_keep = step1["subtype_norm"].isin(TARGET_SUBTYPES)
    for sid in step1.index[~mask_keep]:
        exclusions.append(
            {
                "sample_id": str(sid),
                "reason": f"subtype_not_in_4class:{step1.at[sid, 'subtype_norm']}",
                "subtype_raw": str(step1.at[sid, "subtype_raw"]),
                "subtype_norm": str(step1.at[sid, "subtype_norm"]),
            }
        )
    step2 = step1.loc[mask_keep].copy()
    log.log(
        f"Class distribution AFTER 4-class filter (before NA filter): "
        f"n={len(step2)}; counts={class_counts(step2)}"
    )

    # 3) Drop samples with any native missing among the 50 genes
    mask_na = step2[gene_cols].isna().any(axis=1)
    for sid in step2.index[mask_na]:
        exclusions.append(
            {
                "sample_id": str(sid),
                "reason": "native_missing_expression_in_pam50",
                "subtype_raw": str(step2.at[sid, "subtype_raw"]),
                "subtype_norm": str(step2.at[sid, "subtype_norm"]),
            }
        )
    step3 = step2.loc[~mask_na].copy()
    log.log(
        f"Class distribution AFTER dropping native-missing expression: "
        f"n={len(step3)}; counts={class_counts(step3)}"
    )

    final = step3[list(gene_cols)].copy()
    final["PAM50"] = step3["subtype_norm"].values
    final.index.name = "sample_id"
    final = final.reset_index()

    excl_df = pd.DataFrame(exclusions)
    return final, excl_df


def run_assertions(final: pd.DataFrame, gene_cols: list[str], log: PrepLogger) -> None:
    errors: list[str] = []

    # Exactly 50 unique PAM50 genes
    if len(gene_cols) != 50:
        errors.append(f"Expected 50 gene columns, got {len(gene_cols)}")
    if len(set(gene_cols)) != 50:
        errors.append("Gene columns are not unique")
    if set(gene_cols) != set(PAM50_TARGET_GENES):
        missing = set(PAM50_TARGET_GENES) - set(gene_cols)
        extra = set(gene_cols) - set(PAM50_TARGET_GENES)
        errors.append(f"Gene set mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    # No duplicated sample IDs
    if final["sample_id"].duplicated().any():
        errors.append("Duplicated sample_id in final matrix")

    # No duplicated output gene columns
    if final.columns.duplicated().any():
        errors.append("Duplicated columns in final matrix")

    # Only four intended subtype labels
    labels = set(final["PAM50"].unique())
    if labels != set(TARGET_SUBTYPES):
        errors.append(f"Unexpected PAM50 labels: {sorted(labels)}")

    # No missing expression values
    if final[gene_cols].isna().any().any():
        errors.append("Final matrix still contains missing expression values")

    # Non-empty
    if len(final) == 0:
        errors.append("Final matrix is empty")

    if errors:
        for e in errors:
            log.log(f"ASSERT FAIL: {e}")
        raise AssertionError("Preparation assertions failed:\n- " + "\n- ".join(errors))

    log.log("All automated assertions PASSED.")


def deterministic_fingerprint(final: pd.DataFrame) -> str:
    """Stable content hash for determinism checks."""
    # Sort by sample_id for stability.
    ordered = final.sort_values("sample_id").reset_index(drop=True)
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expression_value_summary(final: pd.DataFrame, gene_cols: list[str]) -> dict:
    vals = final[gene_cols].to_numpy(dtype=float)
    return {
        "n_values": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)),
        "median": float(np.median(vals)),
        "p01": float(np.quantile(vals, 0.01)),
        "p99": float(np.quantile(vals, 0.99)),
    }


def main() -> int:
    log = PrepLogger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.log(f"ROOT={ROOT}")
    log.log(f"RAW_DIR={RAW_DIR}")
    log.log(f"OUT_DIR={OUT_DIR}")

    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Raw METABRIC directory not found: {RAW_DIR}")

    study_meta_path = RAW_DIR / "meta_study.txt"
    study_meta = parse_meta_file(study_meta_path) if study_meta_path.exists() else {}

    # 1) Discover + select expression profile from metadata
    profiles = discover_expression_profiles(RAW_DIR, log)
    if not profiles:
        raise RuntimeError("No mRNA expression profiles discovered from metadata.")
    chosen = select_expression_profile(profiles, log)
    expr_path = Path(chosen["data_path"])

    # 2) Clinical tables + subtype field report
    patient_path = RAW_DIR / "data_clinical_patient.txt"
    sample_path = RAW_DIR / "data_clinical_sample.txt"
    if not patient_path.exists() or not sample_path.exists():
        raise FileNotFoundError("Missing clinical patient/sample files in raw METABRIC dir.")

    patient = read_cbioportal_clinical(patient_path)
    sample = read_cbioportal_clinical(sample_path)
    log.log(f"Loaded clinical patient n={len(patient)}, sample n={len(sample)}")

    clinical_report = build_clinical_field_report(patient, sample, log)
    clinical_report_path = OUT_DIR / "clinical_field_report.csv"
    clinical_report.to_csv(clinical_report_path, index=False)
    log.log(f"Wrote {clinical_report_path}")

    source_table, subtype_col = choose_subtype_field(clinical_report, log)

    # 3) Expression load + gene mapping
    expr = load_expression_matrix(expr_path, log)
    expr_mat, mapping_report = map_pam50_genes(expr, log)
    mapping_path = OUT_DIR / "gene_mapping_report.csv"
    mapping_report.to_csv(mapping_path, index=False)
    log.log(f"Wrote {mapping_path}")

    gene_cols = list(PAM50_TARGET_GENES)

    # 4) Attach subtypes (preserve sample IDs)
    annotated = attach_subtypes(expr_mat, patient, sample, source_table, subtype_col, log)

    # 5) Native missingness BEFORE filtering
    miss_report = missingness_report(annotated, gene_cols, log)
    miss_path = OUT_DIR / "native_missingness_report.csv"
    miss_report.to_csv(miss_path, index=False)
    log.log(f"Wrote {miss_path}")

    # 6) Filter to 4-class complete matrix
    final, exclusions = filter_to_benchmark_matrix(annotated, gene_cols, log)
    excl_path = OUT_DIR / "excluded_samples.csv"
    exclusions.to_csv(excl_path, index=False)
    log.log(f"Wrote {excl_path}")

    # 7) Assertions
    run_assertions(final, gene_cols, log)

    # 8) Determinism fingerprint
    fingerprint = deterministic_fingerprint(final)
    log.log(f"Deterministic content fingerprint (sha256): {fingerprint}")

    # Re-run fingerprint on in-memory copy to confirm stability of serialization path
    fingerprint2 = deterministic_fingerprint(final.copy())
    if fingerprint != fingerprint2:
        raise AssertionError("Determinism check failed: fingerprint mismatch on identical frame.")
    log.log("Determinism check (in-memory rerun) PASSED.")

    # 9) Write final matrix
    final_path = OUT_DIR / "metabric_pam50_4class.csv"
    # Stable column order: sample_id, 50 genes (canonical order), PAM50
    final_ordered = final[["sample_id", *gene_cols, "PAM50"]].sort_values("sample_id")
    final_ordered.to_csv(final_path, index=False)
    log.log(f"Wrote {final_path}")

    # Verify file round-trip determinism
    reloaded = pd.read_csv(final_path)
    fingerprint3 = deterministic_fingerprint(reloaded)
    if fingerprint3 != fingerprint:
        raise AssertionError(
            "Determinism check failed: written CSV fingerprint differs from in-memory."
        )
    log.log("Determinism check (CSV round-trip) PASSED.")

    # 10) Inventory JSON
    value_summary = expression_value_summary(final_ordered, gene_cols)
    class_dist = {str(k): int(v) for k, v in final_ordered["PAM50"].value_counts().items()}
    aliases_used = mapping_report.loc[
        mapping_report["match_method"] == "symbol_alias_fallback",
        ["target_gene", "documented_aliases", "selected_hugo_symbol"],
    ]
    multi_probe = mapping_report.loc[mapping_report["n_candidate_probes"] > 1]

    exclusion_reasons = (
        exclusions["reason"].value_counts().to_dict() if len(exclusions) else {}
    )

    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": study_meta,
        "source_raw_dir": str(RAW_DIR),
        "source_files": {
            "expression": {
                "path": str(expr_path),
                "sha256": file_sha256(expr_path),
                "bytes": expr_path.stat().st_size,
            },
            "clinical_patient": {
                "path": str(patient_path),
                "sha256": file_sha256(patient_path),
                "bytes": patient_path.stat().st_size,
            },
            "clinical_sample": {
                "path": str(sample_path),
                "sha256": file_sha256(sample_path),
                "bytes": sample_path.stat().st_size,
            },
        },
        "discovered_expression_profiles": profiles,
        "selected_expression_profile": {
            k: chosen[k]
            for k in chosen
            if k != "data_path"
        },
        "expression_profile_documentation": {
            "datatype": chosen.get("datatype"),
            "platform_or_profile_name": chosen.get("profile_name"),
            "profile_description": chosen.get("profile_description"),
            "genetic_alteration_type": chosen.get("genetic_alteration_type"),
            "stable_id": chosen.get("stable_id"),
            "normalization_or_transformation_as_declared": chosen.get("profile_description"),
            "note": (
                "Do not describe this matrix as 'raw'. Source metadata declares "
                f"datatype={chosen.get('datatype')!r} and profile_description="
                f"{chosen.get('profile_description')!r}."
            ),
        },
        "subtype_field_selection": {
            "source_table": source_table,
            "column_name": subtype_col,
            "interpretation": (
                "CLAUDIN_SUBTYPE corresponds to cBioPortal display name "
                "'Pam50 + Claudin-low subtype'. It is NOT assumed to be pure PAM50. "
                "Benchmark filtering retains LumA/LumB/Her2/Basal only."
            ),
            "target_subtypes": list(TARGET_SUBTYPES),
            "excluded_labels_policy": list(EXCLUDED_SUBTYPE_LABELS),
        },
        "gene_mapping": {
            "n_target_genes": 50,
            "primary_key": "Entrez_Gene_Id",
            "alias_fallback": PAM50_SYMBOL_ALIASES,
            "probe_selection_rule": PROBE_SELECTION_RULE,
            "n_entrez_matches": int((mapping_report["match_method"] == "entrez").sum()),
            "n_alias_fallbacks": int((mapping_report["match_method"] == "symbol_alias_fallback").sum()),
            "n_multi_probe_genes": int((mapping_report["n_candidate_probes"] > 1).sum()),
            "alias_fallback_rows": aliases_used.to_dict(orient="records"),
            "multi_probe_genes": multi_probe[
                ["target_gene", "n_candidate_probes", "selected_probe_id", "selected_variance"]
            ].to_dict(orient="records"),
        },
        "filtering": {
            "n_expression_samples_initial": int(expr_mat.shape[0]),
            "n_excluded": int(len(exclusions)),
            "exclusion_reason_counts": {str(k): int(v) for k, v in exclusion_reasons.items()},
            "n_final_samples": int(len(final_ordered)),
            "class_distribution_final": class_dist,
        },
        "final_matrix": {
            "path": str(final_path),
            "n_samples": int(len(final_ordered)),
            "n_genes": 50,
            "genes": gene_cols,
            "sample_id_preserved": True,
            "content_sha256": fingerprint,
            "expression_value_summary": value_summary,
        },
        "artifacts": {
            "metabric_inventory.json": str(OUT_DIR / "metabric_inventory.json"),
            "clinical_field_report.csv": str(clinical_report_path),
            "gene_mapping_report.csv": str(mapping_path),
            "native_missingness_report.csv": str(miss_path),
            "metabric_pam50_4class.csv": str(final_path),
            "excluded_samples.csv": str(excl_path),
            "preparation_log.txt": str(OUT_DIR / "preparation_log.txt"),
        },
    }

    inv_path = OUT_DIR / "metabric_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    log.log(f"Wrote {inv_path}")

    # Final human-readable summary block
    log.log("=" * 72)
    log.log("PREPARATION COMPLETE — SUMMARY")
    log.log("=" * 72)
    log.log(f"Final sample count: {len(final_ordered)}")
    log.log(f"Class distribution: {class_dist}")
    log.log(f"Recovered genes (50): {', '.join(gene_cols)}")
    if len(aliases_used):
        log.log(f"Alias/symbol fallbacks used: {aliases_used.to_dict(orient='records')}")
    else:
        log.log("Alias/symbol fallbacks used: none (all genes matched by Entrez ID)")
    if len(multi_probe):
        log.log(
            "Multi-probe genes (selected by max variance): "
            + json.dumps(
                multi_probe[["target_gene", "selected_hugo_symbol", "selected_variance"]]
                .to_dict(orient="records"),
                ensure_ascii=True,
            )
        )
    else:
        log.log("Multi-probe genes: none")
    log.log(f"Excluded samples: {len(exclusions)}; reasons={exclusion_reasons}")
    log.log(
        "Expression-value summary: "
        + json.dumps(value_summary, ensure_ascii=True)
    )
    log.log(
        "Selected expression profile documentation: "
        f"datatype={chosen.get('datatype')!r}; "
        f"profile_name={chosen.get('profile_name')!r}; "
        f"profile_description={chosen.get('profile_description')!r}"
    )
    log.log("STOPPING after preparation (no imputation/classification run).")

    log_path = OUT_DIR / "preparation_log.txt"
    log.save(log_path)
    print(f"\nLog saved to: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Always attempt to persist a partial log if logger exists in locals.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
