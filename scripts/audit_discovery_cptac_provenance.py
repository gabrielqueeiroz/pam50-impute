#!/usr/bin/env python3
"""Download LinkedOmics CPTAC-BRCA reference data and audit discovery provenance."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "external" / "cptac_brca_linkedomics"
OUT_DIR = ROOT / "artifacts" / "discovery_cptac_provenance_audit"
DISCOVERY_CSV = ROOT / "data" / "processed" / "discovery" / "discovery_pam50_4class.csv"
RAW_DISCOVERY = ROOT / "data" / "raw" / "discovery" / "df_MI_no_missing.csv"

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]

# Documented LinkedOmics CPTAC prospective BRCA files (122 samples).
CANDIDATE_URLS = {
    "clinical": [
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_CLI.tsi",
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/CPTAC_BRCA_2018_CLI.tsi",
    ],
    "rnaseq": [
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_RNA_GENE.cct",
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_RNA_Gene.cct",
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_RNAseq_gene.cct",
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/HS_CPTAC_BRCA_2018_RNASeq.cct",
        "https://www.linkedomics.org/data_download/CPTAC-BRCA/CPTAC_BRCA_2018_RNA_Gene.cct",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def try_download(urls: list[str], dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    if dest.exists() and dest.stat().st_size > 1000:
        return {
            "ok": True,
            "path": str(dest),
            "sha256": sha256_file(dest),
            "bytes": dest.stat().st_size,
            "source": "cached",
            "attempts": [],
        }
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bioinfo-provenance-audit/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 500 or b"<html" in data[:200].lower():
                attempts.append({"url": url, "ok": False, "reason": "html_or_too_small", "bytes": len(data)})
                continue
            dest.write_bytes(data)
            return {
                "ok": True,
                "path": str(dest),
                "sha256": sha256_file(dest),
                "bytes": len(data),
                "source": url,
                "attempts": attempts,
            }
        except Exception as exc:  # noqa: BLE001
            attempts.append({"url": url, "ok": False, "reason": str(exc)})
    return {"ok": False, "path": str(dest), "attempts": attempts}


def discover_urls_from_index() -> dict:
    """Parse LinkedOmics index HTML for direct file links."""
    url = "https://www.linkedomics.org/data_download/CPTAC-BRCA/"
    req = urllib.request.Request(url, headers={"User-Agent": "bioinfo-provenance-audit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    abs_links = []
    for h in hrefs:
        if h.startswith("http"):
            abs_links.append(h)
        elif h.startswith("/"):
            abs_links.append("https://www.linkedomics.org" + h)
        else:
            abs_links.append("https://www.linkedomics.org/data_download/CPTAC-BRCA/" + h.lstrip("./"))
    return {
        "index_url": url,
        "all_hrefs": abs_links,
        "clinical_like": [u for u in abs_links if re.search(r"cli|clinical|\.tsi", u, re.I)],
        "rna_like": [u for u in abs_links if re.search(r"rna|\.cct", u, re.I)],
    }


def read_linkedomics_matrix(path: Path) -> pd.DataFrame:
    """LinkedOmics matrices are usually tab-separated with an index column."""
    return pd.read_csv(path, sep="\t", index_col=0)


def load_cptac_clinical(path: Path) -> tuple[pd.Series, list[str], str]:
    """
    Return (subtypes Series indexed by sample_id, sample_ids, subtype_field).
    LinkedOmics CLI.tsi for CPTAC-BRCA is samples x attributes (with an IDX type row).
    """
    clin = read_linkedomics_matrix(path)
    # Drop type-annotation row if present
    drop_idx = [i for i in clin.index if str(i).upper() in {"IDX", "ATTRIBUTE_TYPE", "TYPE"}]
    if drop_idx:
        clin = clin.drop(index=drop_idx)

    # Orientation A: samples x attributes (PAM50 as column)
    for cand in ["PAM50", "PAM50.mRNA", "PAM50_subtype", "Subtype", "subtype", "PAM50 Call"]:
        if cand in clin.columns:
            subtypes = clin[cand].map(normalize_subtype)
            return subtypes, list(clin.index.astype(str)), cand

    # Orientation B: attributes x samples (PAM50 as row)
    for cand in ["PAM50", "PAM50.mRNA", "PAM50_subtype", "Subtype", "subtype", "PAM50 Call"]:
        if cand in clin.index:
            subtypes = clin.loc[cand].map(normalize_subtype)
            return subtypes, list(clin.columns.astype(str)), cand

    # Heuristic over columns then rows
    for col in clin.columns:
        vals = clin[col].astype(str)
        hits = sum(
            1 for v in vals if normalize_subtype(v) in {"LumA", "LumB", "Her2", "Basal", "Normal-like"}
        )
        if hits >= 50:
            subtypes = clin[col].map(normalize_subtype)
            return subtypes, list(clin.index.astype(str)), str(col)
    for idx in clin.index:
        vals = clin.loc[idx].astype(str)
        hits = sum(
            1 for v in vals if normalize_subtype(v) in {"LumA", "LumB", "Her2", "Basal", "Normal-like"}
        )
        if hits >= 50:
            subtypes = clin.loc[idx].map(normalize_subtype)
            return subtypes, list(clin.columns.astype(str)), str(idx)

    raise ValueError("Could not locate PAM50 subtype field in clinical matrix.")


def normalize_subtype(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    mapping = {
        "Luminal A": "LumA",
        "LumA": "LumA",
        "lumA": "LumA",
        "Luminal B": "LumB",
        "LumB": "LumB",
        "lumB": "LumB",
        "Her2": "Her2",
        "HER2": "Her2",
        "Her2-enriched": "Her2",
        "Basal": "Basal",
        "basal-like": "Basal",
        "Basal-like": "Basal",
        "Normal-like": "Normal-like",
        "Normal": "Normal-like",
        "Normal like": "Normal-like",
    }
    return mapping.get(s, s)


def row_fingerprint(v: np.ndarray, decimals: int = 6) -> str:
    arr = np.asarray(v, dtype=float)
    rounded = np.round(arr, decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def search_local_project() -> dict:
    """Search notebooks/scripts/archives for CPTAC linkage strings."""
    roots = [ROOT / "archive", ROOT / "notebooks", ROOT / "scripts", ROOT / "data", ROOT / "README.md"]
    patterns = [
        r"CPTAC",
        r"cptac",
        r"01BR",
        r"11BR",
        r"linkedomics",
        r"LinkedOmics",
        r"Normal-like",
        r"df_MI_no_missing",
        r"brca_metabric",
    ]
    hits: list[dict] = []
    searched = 0
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".ipynb", ".md", ".txt", ".csv", ".json", ".tsv", ".r"}:
                continue
            if path.stat().st_size > 20_000_000:
                continue
            searched += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pat in patterns:
                if re.search(pat, text):
                    # Capture a short context snippet
                    m = re.search(pat, text)
                    start = max(0, m.start() - 60)
                    end = min(len(text), m.end() + 80)
                    snippet = re.sub(r"\s+", " ", text[start:end])[:180]
                    hits.append({"path": str(path.relative_to(ROOT)), "pattern": pat, "snippet": snippet})
                    break
    # Deduplicate by path+pattern
    uniq = []
    seen = set()
    for h in hits:
        key = (h["path"], h["pattern"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return {"files_searched": searched, "hits": uniq[:200], "n_hits": len(uniq)}


def main() -> int:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()

    local_search = search_local_project()
    index_info = discover_urls_from_index()

    # Extend candidates from index
    clin_urls = list(dict.fromkeys(CANDIDATE_URLS["clinical"] + index_info.get("clinical_like", [])))
    rna_urls = list(dict.fromkeys(CANDIDATE_URLS["rnaseq"] + index_info.get("rna_like", [])))

    clin_dl = try_download(clin_urls, REF_DIR / "HS_CPTAC_BRCA_2018_CLI.tsi")
    rna_dl = try_download(rna_urls, REF_DIR / "HS_CPTAC_BRCA_2018_RNA_Gene.cct")

    report: dict = {
        "generated_at_utc": stamp,
        "cohort_key_policy": "discovery (do not rename to CPTAC unless Confirmed)",
        "reference_source": {
            "name": "LinkedOmics CPTAC-BRCA prospective cohort",
            "portal": "https://www.linkedomics.org/data_download/CPTAC-BRCA/",
            "documented_n_samples": 122,
            "rnaseq_unit": "log2(FPKM), normalized by gene median",
            "clinical_download": clin_dl,
            "rnaseq_download": rna_dl,
            "index_parse": {
                "n_hrefs": len(index_info.get("all_hrefs", [])),
                "clinical_like": index_info.get("clinical_like", [])[:20],
                "rna_like": index_info.get("rna_like", [])[:20],
            },
        },
        "local_project_search": local_search,
    }

    if not clin_dl["ok"]:
        report["status"] = "FAIL_DOWNLOAD_CLINICAL"
        (OUT_DIR / "cptac_provenance_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)[:4000])
        return 1

    clin_path = Path(clin_dl["path"])
    try:
        subtypes, sample_ids, subtype_col = load_cptac_clinical(clin_path)
    except Exception as exc:  # noqa: BLE001
        clin = read_linkedomics_matrix(clin_path)
        report["status"] = "FAIL_NO_SUBTYPE_FIELD"
        report["clinical_load_error"] = repr(exc)
        report["clinical_index_sample"] = [str(i) for i in list(clin.index)[:40]]
        report["clinical_columns_sample"] = [str(c) for c in list(clin.columns)[:10]]
        (OUT_DIR / "cptac_provenance_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No subtype field found:", repr(exc))
        return 1

    subtype_counts_all = subtypes.value_counts(dropna=False).to_dict()
    four_class_mask = subtypes.isin(["LumA", "LumB", "Her2", "Basal"])
    subtype_counts_4 = subtypes[four_class_mask].value_counts().to_dict()
    n_normal = int((subtypes == "Normal-like").sum())

    discovery = pd.read_csv(DISCOVERY_CSV)
    disc_X = discovery[PAM50_GENES].to_numpy(dtype=float)
    disc_y = discovery["PAM50"].astype(str).to_list()
    disc_counts = pd.Series(disc_y).value_counts().to_dict()

    report["class_count_comparison"] = {
        "discovery": disc_counts,
        "cptac_all": {str(k): int(v) for k, v in subtype_counts_all.items()},
        "cptac_4class_excluding_normal_like": {str(k): int(v) for k, v in subtype_counts_4.items()},
        "cptac_n_total": int(len(subtypes)),
        "cptac_n_normal_like": n_normal,
        "cptac_n_4class": int(four_class_mask.sum()),
        "discovery_n": int(len(discovery)),
        "exact_4class_count_match": (
            subtype_counts_4.get("LumA") == disc_counts.get("LumA")
            and subtype_counts_4.get("Basal") == disc_counts.get("Basal")
            and subtype_counts_4.get("LumB") == disc_counts.get("LumB")
            and subtype_counts_4.get("Her2") == disc_counts.get("Her2")
            and int(four_class_mask.sum()) == int(len(discovery))
        ),
        "subtype_field_used": subtype_col,
        "sample_id_examples": sample_ids[:8],
    }

    expression_block: dict = {"available": False}
    match_block: dict = {"available": False}

    if rna_dl["ok"]:
        rna = read_linkedomics_matrix(Path(rna_dl["path"]))
        # genes x samples expected
        if set(PAM50_GENES).issubset(set(map(str, rna.index))):
            rna_pam = rna.loc[PAM50_GENES]
            gene_axis = "index"
        elif set(PAM50_GENES).issubset(set(map(str, rna.columns))):
            rna_pam = rna[PAM50_GENES].T
            gene_axis = "columns"
        else:
            # gene symbols may have suffixes
            idx_map = {str(i).split("|")[0].split(".")[0]: i for i in rna.index}
            present = [g for g in PAM50_GENES if g in idx_map]
            if len(present) >= 45:
                rna_pam = rna.loc[[idx_map[g] for g in present]]
                rna_pam.index = present
                gene_axis = "index_mapped"
            else:
                rna_pam = None
                gene_axis = "missing"
                expression_block = {
                    "available": False,
                    "reason": "PAM50 genes not found in RNAseq matrix",
                    "n_index": int(rna.shape[0]),
                    "n_cols": int(rna.shape[1]),
                    "index_sample": [str(i) for i in list(rna.index)[:20]],
                }

        if rna_pam is not None:
            # Align samples to clinical order where possible
            common_samples = [s for s in sample_ids if s in rna_pam.columns.astype(str)]
            if not common_samples:
                # maybe columns already match
                common_samples = list(rna_pam.columns.astype(str))
            rna_pam = rna_pam[common_samples].apply(pd.to_numeric, errors="coerce")

            genes_present = [g for g in PAM50_GENES if g in rna_pam.index]
            rna_mat = rna_pam.loc[genes_present].T  # samples x genes
            # Restrict to 4-class if subtypes available
            sub_map = dict(zip(sample_ids, subtypes.tolist()))
            rna_mat["PAM50"] = [sub_map.get(s) for s in rna_mat.index.astype(str)]
            rna4 = rna_mat[rna_mat["PAM50"].isin(["LumA", "LumB", "Her2", "Basal"])].copy()
            Xc = rna4[genes_present].to_numpy(dtype=float)
            yc = rna4["PAM50"].astype(str).to_numpy()

            # Discovery uses full 50 genes; if CPTAC missing some, compare on intersection
            disc_use = discovery[genes_present].to_numpy(dtype=float)
            disc_lab = discovery["PAM50"].astype(str).to_numpy()

            expression_block = {
                "available": True,
                "gene_axis": gene_axis,
                "n_pam50_genes_in_cptac": len(genes_present),
                "missing_genes": [g for g in PAM50_GENES if g not in genes_present],
                "cptac_4class_n": int(len(rna4)),
                "discovery_expression_summary": {
                    "min": float(np.nanmin(disc_use)),
                    "max": float(np.nanmax(disc_use)),
                    "mean": float(np.nanmean(disc_use)),
                    "std": float(np.nanstd(disc_use)),
                    "median": float(np.nanmedian(disc_use)),
                },
                "cptac_expression_summary": {
                    "min": float(np.nanmin(Xc)),
                    "max": float(np.nanmax(Xc)),
                    "mean": float(np.nanmean(Xc)),
                    "std": float(np.nanstd(Xc)),
                    "median": float(np.nanmedian(Xc)),
                },
            }

            # Exact fingerprints at several roundings / transforms
            def build_fps(X: np.ndarray, decimals: int) -> dict[str, list[int]]:
                out: dict[str, list[int]] = {}
                for i in range(X.shape[0]):
                    fp = row_fingerprint(X[i], decimals=decimals)
                    out.setdefault(fp, []).append(i)
                return out

            transforms = {
                "raw": lambda X: X,
                "zscore_per_sample": lambda X: (X - X.mean(axis=1, keepdims=True))
                / np.clip(X.std(axis=1, keepdims=True), 1e-12, None),
                "center_per_gene": lambda X: X - X.mean(axis=0, keepdims=True),
                "rank_per_sample": lambda X: np.apply_along_axis(
                    lambda r: pd.Series(r).rank(method="average").to_numpy(), 1, X
                ),
            }

            exact_results = {}
            for tname, tfun in transforms.items():
                dX = tfun(disc_use)
                cX = tfun(Xc)
                for dec in (6, 4, 3, 2):
                    dfps = build_fps(dX, dec)
                    cfps = build_fps(cX, dec)
                    shared = set(dfps) & set(cfps)
                    n_disc_matched = sum(len(dfps[fp]) for fp in shared)
                    exact_results[f"{tname}_round{dec}"] = {
                        "n_shared_fingerprints": len(shared),
                        "n_discovery_rows_matched": n_disc_matched,
                        "n_discovery": int(dX.shape[0]),
                        "n_cptac": int(cX.shape[0]),
                    }

            # Nearest-profile matching (cosine + pearson) on z-scored rows
            dZ = transforms["zscore_per_sample"](disc_use)
            cZ = transforms["zscore_per_sample"](Xc)
            best = []
            for i in range(dZ.shape[0]):
                sims = [cosine_sim(dZ[i], cZ[j]) for j in range(cZ.shape[0])]
                j = int(np.nanargmax(sims))
                best.append(
                    {
                        "discovery_row": int(i),
                        "discovery_label": str(disc_lab[i]),
                        "best_cptac_sample": str(rna4.index[j]),
                        "best_cptac_label": str(yc[j]),
                        "cosine": float(sims[j]),
                        "pearson": pearson(dZ[i], cZ[j]),
                        "label_match": bool(str(disc_lab[i]) == str(yc[j])),
                    }
                )
            best_df = pd.DataFrame(best)
            best_df.to_csv(OUT_DIR / "nearest_profile_matches.csv", index=False)

            # Subset relationship: is every discovery row within eps of some CPTAC row?
            max_abs = []
            for i in range(disc_use.shape[0]):
                diffs = np.max(np.abs(Xc - disc_use[i]), axis=1)
                max_abs.append(float(np.min(diffs)))
            max_abs = np.asarray(max_abs)

            match_block = {
                "available": True,
                "exact_fingerprint_results": exact_results,
                "nearest_profile_summary": {
                    "median_best_cosine": float(best_df["cosine"].median()),
                    "mean_best_cosine": float(best_df["cosine"].mean()),
                    "min_best_cosine": float(best_df["cosine"].min()),
                    "frac_cosine_ge_0.99": float((best_df["cosine"] >= 0.99).mean()),
                    "frac_cosine_ge_0.95": float((best_df["cosine"] >= 0.95).mean()),
                    "frac_label_match_among_nearest": float(best_df["label_match"].mean()),
                    "top5_best": best_df.nlargest(5, "cosine").to_dict(orient="records"),
                    "bottom5_best": best_df.nsmallest(5, "cosine").to_dict(orient="records"),
                },
                "subset_maxabs_summary": {
                    "median_min_maxabs": float(np.median(max_abs)),
                    "mean_min_maxabs": float(np.mean(max_abs)),
                    "frac_exact_zero": float((max_abs == 0).mean()),
                    "frac_maxabs_le_1e-6": float((max_abs <= 1e-6).mean()),
                    "frac_maxabs_le_1e-3": float((max_abs <= 1e-3).mean()),
                    "frac_maxabs_le_0.1": float((max_abs <= 0.1).mean()),
                    "frac_maxabs_le_1.0": float((max_abs <= 1.0).mean()),
                },
            }

            # Also try matching after per-gene median centering of discovery to mimic LinkedOmics
            gene_med = np.nanmedian(Xc, axis=0, keepdims=True)
            d_med = disc_use - np.nanmedian(disc_use, axis=0, keepdims=True)
            c_med = Xc - gene_med
            dZ2 = transforms["zscore_per_sample"](d_med)
            cZ2 = transforms["zscore_per_sample"](c_med)
            cos2 = []
            for i in range(dZ2.shape[0]):
                sims = [cosine_sim(dZ2[i], cZ2[j]) for j in range(cZ2.shape[0])]
                cos2.append(float(np.nanmax(sims)))
            match_block["median_centered_nearest_cosine"] = {
                "median": float(np.median(cos2)),
                "mean": float(np.mean(cos2)),
                "frac_ge_0.99": float((np.asarray(cos2) >= 0.99).mean()),
                "frac_ge_0.95": float((np.asarray(cos2) >= 0.95).mean()),
            }

    report["expression_comparison"] = expression_block
    report["row_matching"] = match_block

    # Conclusion classification
    evidence_for: list[str] = []
    evidence_against: list[str] = []

    # Local search evidence
    cptac_local = [h for h in local_search["hits"] if h["pattern"].lower() in {"cptac", "01br", "11br", "linkedomics"}]
    metabric_path_hits = [h for h in local_search["hits"] if h["pattern"] == "brca_metabric"]
    if not cptac_local:
        evidence_against.append(
            "No local notebooks/scripts/exports contain CPTAC sample IDs (01BR/11BR) or LinkedOmics download provenance for the discovery matrix."
        )
    else:
        evidence_for.append(f"Local files mention CPTAC-related tokens ({len(cptac_local)} hits).")
    if metabric_path_hits:
        evidence_against.append(
            "Historical Colab path references brca_metabric/ for df_MI_no_missing.csv (folder name suggests METABRIC, though this is weak evidence)."
        )

    cc = report["class_count_comparison"]
    if cc["exact_4class_count_match"] and cc["cptac_n_4class"] == cc["discovery_n"]:
        evidence_for.append(
            f"Exact PAM50 4-class counts match CPTAC prospective after excluding Normal-like "
            f"(n={cc['discovery_n']}; {cc['discovery']})."
        )
    else:
        evidence_against.append(
            f"Class counts do not exactly match CPTAC 4-class distribution "
            f"(discovery={cc['discovery']}, cptac_4={cc['cptac_4class_excluding_normal_like']})."
        )

    if expression_block.get("available"):
        ds = expression_block["discovery_expression_summary"]
        cs = expression_block["cptac_expression_summary"]
        # Range comparison
        if abs(ds["min"] - cs["min"]) < 0.5 and abs(ds["max"] - cs["max"]) < 0.5:
            evidence_for.append(
                f"Expression ranges are similar (discovery [{ds['min']:.3f},{ds['max']:.3f}] vs "
                f"CPTAC [{cs['min']:.3f},{cs['max']:.3f}])."
            )
        else:
            evidence_against.append(
                f"Expression ranges differ (discovery [{ds['min']:.3f},{ds['max']:.3f}] vs "
                f"CPTAC [{cs['min']:.3f},{cs['max']:.3f}]), suggesting different transform/platform or scaling."
            )
        if expression_block.get("n_pam50_genes_in_cptac") == 50:
            evidence_for.append("All 50 PAM50 genes are present in the LinkedOmics CPTAC RNAseq matrix.")
        else:
            evidence_against.append(
                f"Only {expression_block.get('n_pam50_genes_in_cptac')} / 50 PAM50 genes found in CPTAC RNAseq."
            )

    if match_block.get("available"):
        exact0 = match_block["exact_fingerprint_results"].get("raw_round6", {})
        if exact0.get("n_discovery_rows_matched", 0) == 117:
            evidence_for.append("Exact row fingerprints match all 117 discovery rows to CPTAC (Confirmed-level).")
        elif exact0.get("n_discovery_rows_matched", 0) > 0:
            evidence_for.append(
                f"Exact raw fingerprints match {exact0['n_discovery_rows_matched']}/117 discovery rows."
            )
        else:
            evidence_against.append("No exact raw expression row fingerprints match between discovery and CPTAC RNAseq.")

        ns = match_block["nearest_profile_summary"]
        if ns["frac_cosine_ge_0.99"] >= 0.9:
            evidence_for.append(
                f"Nearest-profile cosine >=0.99 for {ns['frac_cosine_ge_0.99']*100:.1f}% of discovery rows."
            )
        elif ns["frac_cosine_ge_0.95"] >= 0.9:
            evidence_for.append(
                f"Nearest-profile cosine >=0.95 for {ns['frac_cosine_ge_0.95']*100:.1f}% of discovery rows "
                f"(median={ns['median_best_cosine']:.4f})."
            )
        else:
            evidence_against.append(
                f"Nearest-profile similarity is modest (median cosine={ns['median_best_cosine']:.4f}; "
                f"frac>=0.95={ns['frac_cosine_ge_0.95']:.3f})."
            )

        ss = match_block["subset_maxabs_summary"]
        if ss["frac_exact_zero"] > 0.5:
            evidence_for.append("Majority of discovery rows are exact subsets of CPTAC expression rows.")
        elif ss["frac_maxabs_le_1e-3"] > 0.5:
            evidence_for.append("Majority of discovery rows nearly exact (max-abs <= 1e-3) vs some CPTAC row.")
        else:
            evidence_against.append(
                f"Discovery rows are not near-exact CPTAC subsets (frac max-abs<=0.1 = {ss['frac_maxabs_le_0.1']:.3f})."
            )

    # Classify
    exact_all = (
        match_block.get("available")
        and match_block["exact_fingerprint_results"].get("raw_round6", {}).get("n_discovery_rows_matched", 0) == 117
    )
    high_sim = (
        match_block.get("available")
        and match_block["nearest_profile_summary"]["frac_cosine_ge_0.99"] >= 0.9
        and cc["exact_4class_count_match"]
    )
    plausible = cc.get("exact_4class_count_match", False) and expression_block.get("available", False)
    not_cptac = (
        match_block.get("available")
        and match_block["nearest_profile_summary"]["median_best_cosine"] < 0.7
        and not cc.get("exact_4class_count_match", False)
    )

    if exact_all:
        conclusion = "Confirmed CPTAC"
    elif high_sim:
        conclusion = "Highly likely CPTAC"
    elif plausible and match_block.get("available") and match_block["nearest_profile_summary"]["median_best_cosine"] >= 0.9:
        conclusion = "Highly likely CPTAC"
    elif plausible and match_block.get("available") and match_block["nearest_profile_summary"]["median_best_cosine"] >= 0.8:
        conclusion = "Plausibly CPTAC"
    elif plausible:
        conclusion = "Plausibly CPTAC"
    elif not_cptac:
        conclusion = "Not CPTAC"
    else:
        conclusion = "Inconclusive"

    report["conclusion"] = conclusion
    report["evidence_for"] = evidence_for
    report["evidence_against"] = evidence_against
    report["naming_policy"] = (
        "Keep cohort key and scientific label as 'discovery'. Do not rename to CPTAC "
        "unless conclusion is Confirmed CPTAC or explicit source-generation evidence exists."
    )
    report["status"] = "COMPLETE"

    # Markdown report
    md = [
        "# Discovery ↔ CPTAC provenance audit",
        "",
        f"- Generated (UTC): `{stamp}`",
        f"- **Conclusion: {conclusion}**",
        f"- Cohort key remains: **`discovery`** (not renamed)",
        "",
        "## Reference dataset",
        "- Source: LinkedOmics CPTAC-BRCA prospective cohort",
        "- Portal: https://www.linkedomics.org/data_download/CPTAC-BRCA/",
        f"- Clinical download ok: `{clin_dl['ok']}` (`{clin_dl.get('source')}`)",
        f"- RNAseq download ok: `{rna_dl['ok']}` (`{rna_dl.get('source')}`)",
        f"- Documented n=122; RNAseq unit: log2(FPKM), gene-median normalized",
        "",
        "## Class counts",
        f"- Discovery: `{cc['discovery']}` (n={cc['discovery_n']})",
        f"- CPTAC all: `{cc['cptac_all']}` (n={cc['cptac_n_total']})",
        f"- CPTAC 4-class (excl. Normal-like): `{cc['cptac_4class_excluding_normal_like']}` (n={cc['cptac_n_4class']})",
        f"- Exact 4-class count match: **{cc['exact_4class_count_match']}**",
        f"- Subtype field: `{cc['subtype_field_used']}`",
        f"- CPTAC sample ID examples: `{cc['sample_id_examples']}`",
        "",
        "## Local project search",
        f"- Files searched: {local_search['files_searched']}",
        f"- Hits: {local_search['n_hits']}",
    ]
    for h in local_search["hits"][:25]:
        md.append(f"- `{h['path']}` · pattern `{h['pattern']}` · …{h['snippet']}…")
    md += ["", "## Expression"]
    if expression_block.get("available"):
        md += [
            f"- PAM50 genes in CPTAC RNAseq: {expression_block['n_pam50_genes_in_cptac']}/50",
            f"- Discovery range: {expression_block['discovery_expression_summary']}",
            f"- CPTAC range: {expression_block['cptac_expression_summary']}",
        ]
    else:
        md.append(f"- Expression comparison unavailable: {expression_block}")
    md += ["", "## Row matching"]
    if match_block.get("available"):
        md.append("### Exact fingerprints")
        for k, v in match_block["exact_fingerprint_results"].items():
            md.append(f"- `{k}`: {v}")
        md.append("### Nearest profiles (z-scored cosine)")
        ns = match_block["nearest_profile_summary"]
        md += [
            f"- median best cosine: {ns['median_best_cosine']:.4f}",
            f"- frac >= 0.99: {ns['frac_cosine_ge_0.99']:.3f}",
            f"- frac >= 0.95: {ns['frac_cosine_ge_0.95']:.3f}",
            f"- label match among nearest: {ns['frac_label_match_among_nearest']:.3f}",
        ]
        md.append("### Subset max-abs")
        md.append(str(match_block["subset_maxabs_summary"]))
    else:
        md.append("Row matching unavailable (RNAseq not downloaded or genes missing).")
    md += [
        "",
        "## Supporting evidence",
    ]
    md += [f"- {e}" for e in evidence_for] if evidence_for else ["- (none)"]
    md += [
        "",
        "## Contradictory evidence",
    ]
    md += [f"- {e}" for e in evidence_against] if evidence_against else ["- (none)"]
    md += [
        "",
        "## Policy",
        report["naming_policy"],
        "",
    ]
    (OUT_DIR / "cptac_provenance_audit.md").write_text("\n".join(md), encoding="utf-8")
    (OUT_DIR / "cptac_provenance_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Lightweight inventory sidecar under processed discovery (does not rename cohort)
    side = {
        "audit_artifact_dir": str(OUT_DIR),
        "conclusion": conclusion,
        "cohort_key_unchanged": "discovery",
        "generated_at_utc": stamp,
    }
    (ROOT / "data" / "processed" / "discovery" / "discovery_cptac_provenance_audit_pointer.json").write_text(
        json.dumps(side, indent=2), encoding="utf-8"
    )

    print(f"CONCLUSION={conclusion}")
    print(f"Wrote {OUT_DIR / 'cptac_provenance_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
