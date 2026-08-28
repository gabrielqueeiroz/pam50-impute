#!/usr/bin/env python3
"""
Finalize CPTAC-derived laboratory discovery cohort.

- Recover biological Patient_IDs
- Build originally_observed mask from pre-imputation cptac.csv
- Rewrite discovery_pam50_4class.csv with recovered IDs
- Write metadata / fingerprint / reports

Does NOT run benchmarks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]
TARGET = ("LumA", "LumB", "Her2", "Basal")
EXPECTED = {"LumA": 57, "Basal": 29, "LumB": 17, "Her2": 14}

DISC_IN = ROOT / "data" / "processed" / "discovery" / "discovery_pam50_4class.csv"
CPTAC_CSV = ROOT / "legacy" / "cptac.csv"
SUBTYPE_CSV = ROOT / "legacy" / "cptac_subtype.csv"
PUTATIVE = (
    ROOT
    / "artifacts"
    / "discovery_cptac_provenance_audit"
    / "putative_discovery_to_cptac_sample_map.csv"
)
OUT_PROC = ROOT / "data" / "processed" / "discovery"
OUT_ART = ROOT / "artifacts" / "discovery_preparation"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_legacy_expr() -> pd.DataFrame:
    """Load laboratory CPTAC expression with notebook missingness convention."""
    df = pd.read_csv(CPTAC_CSV)
    df = df.set_index("Patient_ID")
    df.index = df.index.astype(str)
    # Notebook: df_metabric = df_metabric.replace(0, np.nan)
    df = df.replace(0, np.nan)
    missing = [g for g in PAM50 if g not in df.columns]
    if missing:
        raise RuntimeError(f"PAM50 genes missing from cptac.csv: {missing}")
    return df[PAM50].apply(pd.to_numeric, errors="coerce")


def recover_ids(disc: pd.DataFrame, legacy: pd.DataFrame, subtypes: pd.Series) -> pd.DataFrame:
    """
    Map each discovery row to a unique CPTAC Patient_ID by maximizing exact
    agreement on cells that are finite in legacy (originally observed).
    Cross-check putative map and subtype labels.
    """
    putative = {}
    if PUTATIVE.exists():
        pm = pd.read_csv(PUTATIVE)
        putative = {
            str(r["discovery_row_id"]): str(r["matched_cptac_id"])
            for _, r in pm.iterrows()
        }

    Xd = disc[PAM50].to_numpy(float)
    old_ids = disc["sample_id"].astype(str).tolist()
    yd = disc["PAM50"].astype(str).tolist()

    leg_ids = list(legacy.index.astype(str))
    Xl = legacy.loc[leg_ids, PAM50].to_numpy(float)
    # Precompute finite masks
    fin_l = np.isfinite(Xl)

    rows = []
    used = set()
    for i, old in enumerate(old_ids):
        best_j = None
        best_score = (-1, -1.0)  # (n_exact, pearson)
        best_stats = {}
        for j, cid in enumerate(leg_ids):
            if cid in used:
                continue
            mask = fin_l[j]
            n_obs = int(mask.sum())
            if n_obs < 10:
                continue
            xd = Xd[i, mask]
            xl = Xl[j, mask]
            n_exact = int(np.sum(np.isclose(xd, xl, rtol=0, atol=0)))
            # Prefer exact matches; tie-break with pearson
            if np.std(xd) > 0 and np.std(xl) > 0:
                pear = float(np.corrcoef(xd, xl)[0, 1])
            else:
                pear = float("nan")
            score = (n_exact, pear if np.isfinite(pear) else -1.0)
            if score > best_score:
                best_score = score
                best_j = j
                best_stats = {
                    "n_observed_compared": n_obs,
                    "n_exact": n_exact,
                    "pearson_observed": pear,
                    "frac_exact": n_exact / n_obs if n_obs else 0.0,
                }
        if best_j is None:
            raise RuntimeError(f"No CPTAC match for discovery row {old}")
        cid = leg_ids[best_j]
        used.add(cid)
        put = putative.get(old)
        disc_lab = yd[i]
        leg_lab = str(subtypes.loc[cid]) if cid in subtypes.index else None
        discrepancies = []
        if put and put != cid:
            discrepancies.append(f"putative_map_disagrees:{put}")
        if leg_lab and disc_lab != leg_lab:
            discrepancies.append(f"label_mismatch_discovery={disc_lab}_subtype={leg_lab}")
        conf = "high" if best_stats["frac_exact"] >= 0.95 else (
            "moderate" if best_stats["frac_exact"] >= 0.80 else "low"
        )
        rows.append(
            {
                "previous_row_id": old,
                "discovery_row_index": i,
                "recovered_Patient_ID": cid,
                "discovery_PAM50": disc_lab,
                "subtype_from_cptac_subtype": leg_lab,
                "mapping_source": "exact_agreement_on_legacy_observed_cells",
                "mapping_confidence": conf,
                "putative_map_id": put,
                "n_observed_compared": best_stats["n_observed_compared"],
                "n_exact": best_stats["n_exact"],
                "frac_exact": best_stats["frac_exact"],
                "pearson_observed": best_stats["pearson_observed"],
                "discrepancy": ";".join(discrepancies) if discrepancies else "",
            }
        )

    map_df = pd.DataFrame(rows)
    if len(map_df) != 117:
        raise AssertionError(f"Expected 117 mappings, got {len(map_df)}")
    if map_df["recovered_Patient_ID"].duplicated().any():
        raise AssertionError("Duplicated recovered Patient_IDs")
    if set(map_df["recovered_Patient_ID"]) != set(legacy.index.astype(str)) & set(
        map_df["recovered_Patient_ID"]
    ):
        # all recovered must be in legacy
        missing = set(map_df["recovered_Patient_ID"]) - set(legacy.index.astype(str))
        if missing:
            raise AssertionError(f"Recovered IDs not in legacy: {missing}")
    # Must cover all discovery rows
    if set(map_df["previous_row_id"]) != set(old_ids):
        raise AssertionError("Unmatched discovery rows")
    return map_df


def main() -> int:
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    OUT_ART.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()

    disc = pd.read_csv(DISC_IN)
    # If already recovered (Patient_ID style), keep expression order; rebuild from current
    if not disc["sample_id"].astype(str).str.startswith("discovery_row_").all():
        # May already be finalized; still rebuild mask/metadata from legacy
        print("Note: sample_id column does not look synthetic; remapping from expression.")

    subtype = pd.read_csv(SUBTYPE_CSV)
    subtype["Patient_ID"] = subtype["Patient_ID"].astype(str)
    subtypes = subtype.set_index("Patient_ID")["PAM50"].astype(str)
    # Restrict to 4-class
    subtypes = subtypes[subtypes.isin(TARGET)]

    legacy = load_legacy_expr()
    # Align legacy to subtype 117
    common = legacy.index.intersection(subtypes.index)
    if len(common) != 117:
        raise RuntimeError(f"Expected 117 overlapping IDs, got {len(common)}")
    legacy = legacy.loc[common]
    subtypes = subtypes.loc[common]

    # If discovery already has CPTAC IDs, synthesize previous_row_id from order
    if disc["sample_id"].astype(str).str.startswith("discovery_row_").all():
        map_df = recover_ids(disc, legacy, subtypes)
        # Reorder discovery expression to mapping order (preserve correspondence)
        disc_by_old = disc.set_index("sample_id")
        ordered_old = map_df["previous_row_id"].tolist()
        expr = disc_by_old.loc[ordered_old, PAM50 + ["PAM50"]].reset_index(drop=True)
        new_ids = map_df["recovered_Patient_ID"].tolist()
    else:
        # Already biological IDs — verify against legacy and build identity map
        ids = disc["sample_id"].astype(str).tolist()
        if len(set(ids)) != 117 or set(ids) != set(common.astype(str)):
            # Fall back: remap by expression
            disc = disc.copy()
            disc["sample_id"] = [f"discovery_row_{i:04d}" for i in range(1, len(disc) + 1)]
            map_df = recover_ids(disc, legacy, subtypes)
            disc_by_old = disc.set_index("sample_id")
            ordered_old = map_df["previous_row_id"].tolist()
            expr = disc_by_old.loc[ordered_old, PAM50 + ["PAM50"]].reset_index(drop=True)
            new_ids = map_df["recovered_Patient_ID"].tolist()
        else:
            map_df = pd.DataFrame(
                {
                    "previous_row_id": ids,
                    "discovery_row_index": range(len(ids)),
                    "recovered_Patient_ID": ids,
                    "discovery_PAM50": disc["PAM50"].astype(str),
                    "subtype_from_cptac_subtype": [str(subtypes.loc[i]) for i in ids],
                    "mapping_source": "already_biological_ids",
                    "mapping_confidence": "high",
                    "putative_map_id": ids,
                    "n_observed_compared": np.nan,
                    "n_exact": np.nan,
                    "frac_exact": np.nan,
                    "pearson_observed": np.nan,
                    "discrepancy": [
                        ""
                        if str(disc.loc[i, "PAM50"]) == str(subtypes.loc[sid])
                        else f"label_mismatch_discovery={disc.loc[i,'PAM50']}_subtype={subtypes.loc[sid]}"
                        for i, sid in enumerate(ids)
                    ],
                }
            )
            expr = disc[PAM50 + ["PAM50"]].copy()
            new_ids = ids

    # Prefer laboratory subtype labels for consistency with CPTAC clinical
    final_labels = [str(subtypes.loc[sid]) for sid in new_ids]
    # Keep expression from discovery (post-imputation laboratory matrix)
    final = expr[PAM50].copy()
    final.insert(0, "sample_id", new_ids)
    final["PAM50"] = final_labels

    # Assertions
    assert len(final) == 117
    assert list(final.columns[1:-1]) == PAM50
    assert final["sample_id"].is_unique
    assert set(final["PAM50"]) <= set(TARGET)
    counts = final["PAM50"].value_counts().to_dict()
    assert counts == EXPECTED, counts
    assert final[PAM50].isna().sum().sum() == 0

    # Observation mask from legacy (pre-imputation), aligned to final IDs/genes
    # True = originally observed (finite after replace(0,nan))
    obs = legacy.loc[new_ids, PAM50].notna()
    obs.insert(0, "sample_id", new_ids)
    # Also store as boolean CSV (True/False)
    mask_path = OUT_PROC / "discovery_originally_observed_mask.csv"
    obs.to_csv(mask_path, index=False)

    obs_bool = obs[PAM50].to_numpy(dtype=bool)
    n_total = 117 * 50
    n_obs = int(obs_bool.sum())
    n_imp = int((~obs_bool).sum())
    ambiguous: list[dict] = []
    # Ambiguous: none under this convention; document zeros treated as missing
    mask_report = {
        "generated_at_utc": stamp,
        "convention": [
            "Loaded legacy/cptac.csv",
            "Applied replace(0, NaN) as in 1Preprocess_analyzes_CPTAC_Genes.ipynb",
            "originally_observed = finite after that transform",
            "False cells are legacy-imputed (or zero-as-missing) in the discovery matrix",
        ],
        "total_cells": n_total,
        "n_originally_observed": n_obs,
        "pct_originally_observed": n_obs / n_total * 100,
        "n_legacy_imputed": n_imp,
        "pct_legacy_imputed": n_imp / n_total * 100,
        "by_gene": {
            g: {
                "n_observed": int(obs[g].sum()),
                "n_imputed": int((~obs[g]).sum()),
            }
            for g in PAM50
        },
        "by_subtype": {},
        "by_sample": {
            sid: {
                "n_observed": int(obs_bool[i].sum()),
                "n_imputed": int((~obs_bool[i]).sum()),
                "PAM50": final_labels[i],
            }
            for i, sid in enumerate(new_ids)
        },
        "ambiguous_cells": ambiguous,
        "n_ambiguous": 0,
    }
    for lab in TARGET:
        idx = [i for i, L in enumerate(final_labels) if L == lab]
        subm = obs_bool[idx]
        mask_report["by_subtype"][lab] = {
            "n_samples": len(idx),
            "n_observed": int(subm.sum()),
            "n_imputed": int((~subm).sum()),
            "pct_imputed": float((~subm).sum() / subm.size * 100) if subm.size else 0.0,
        }

    (OUT_ART / "discovery_observation_mask_report.json").write_text(
        json.dumps(mask_report, indent=2), encoding="utf-8"
    )
    md_mask = [
        "# Discovery observation mask report",
        "",
        f"- Generated (UTC): `{stamp}`",
        f"- Total cells: **{n_total}** (117 × 50)",
        f"- Originally observed: **{n_obs}** ({n_obs/n_total*100:.2f}%)",
        f"- Legacy-imputed / zero-as-missing: **{n_imp}** ({n_imp/n_total*100:.2f}%)",
        f"- Ambiguous cells: **0**",
        "",
        "## Convention",
        "- `replace(0, NaN)` then finite ⇒ observed",
        "",
        "## By subtype",
    ]
    for lab, d in mask_report["by_subtype"].items():
        md_mask.append(
            f"- {lab}: observed={d['n_observed']}, imputed={d['n_imputed']} "
            f"({d['pct_imputed']:.2f}% imputed)"
        )
    md_mask += ["", "## Genes with any imputed cells"]
    for g, d in mask_report["by_gene"].items():
        if d["n_imputed"] > 0:
            md_mask.append(f"- {g}: imputed={d['n_imputed']} / 117")
    (OUT_ART / "discovery_observation_mask_report.md").write_text(
        "\n".join(md_mask) + "\n", encoding="utf-8"
    )

    # Write mapping audit
    map_path = OUT_ART / "discovery_sample_id_mapping.csv"
    map_df.to_csv(map_path, index=False)

    # Deterministic order: sort by sample_id for saved matrix (expression follows)
    final_sorted = final.sort_values("sample_id").reset_index(drop=True)
    # Reorder mask to match
    obs_sorted = obs.set_index("sample_id").loc[final_sorted["sample_id"], PAM50].reset_index()
    obs_sorted.to_csv(mask_path, index=False)
    final_path = OUT_PROC / "discovery_pam50_4class.csv"
    final_sorted.to_csv(final_path, index=False)

    # Fingerprint
    values_csv = final_sorted[PAM50 + ["PAM50"]].to_csv(index=False).encode("utf-8")
    full_csv = final_sorted.to_csv(index=False).encode("utf-8")
    fp = {
        "n_samples": 117,
        "n_genes": 50,
        "genes": PAM50,
        "class_distribution": {
            str(k): int(v) for k, v in final_sorted["PAM50"].value_counts().items()
        },
        "values_and_labels_sha256": sha256_bytes(values_csv),
        "full_matrix_sha256": sha256_bytes(full_csv),
        "observation_mask_sha256": sha256_bytes(
            obs_sorted.to_csv(index=False).encode("utf-8")
        ),
        "n_originally_observed_cells": n_obs,
        "n_legacy_imputed_cells": n_imp,
    }
    # Round-trip determinism
    again = pd.read_csv(final_path).sort_values("sample_id").reset_index(drop=True)
    assert sha256_bytes(again[PAM50 + ["PAM50"]].to_csv(index=False).encode()) == fp[
        "values_and_labels_sha256"
    ]

    (OUT_PROC / "fingerprint.json").write_text(json.dumps(fp, indent=2), encoding="utf-8")
    # Keep legacy name too
    (OUT_PROC / "discovery_fingerprint.json").write_text(
        json.dumps(fp, indent=2), encoding="utf-8"
    )

    metadata = {
        "cohort_key": "discovery",
        "scientific_identity": "CPTAC-derived laboratory discovery cohort",
        "not_described_as": [
            "current LinkedOmics export",
            "byte-identical official CPTAC release",
        ],
        "source_files": {
            "laboratory_expression": str(CPTAC_CSV),
            "laboratory_subtype": str(SUBTYPE_CSV),
            "prior_processed_matrix": str(DISC_IN),
            "putative_map": str(PUTATIVE) if PUTATIVE.exists() else None,
        },
        "n_samples": 117,
        "n_genes": 50,
        "class_distribution": fp["class_distribution"],
        "exclusions": {
            "normal_like_excluded": 5,
            "note": "cptac_subtype.csv already contains the 117 four-class samples",
        },
        "legacy_completion": {
            "description": (
                "Missing PAM50 values in the laboratory CPTAC input were completed "
                "by subtype-specific imputation workflows (legacy notebooks 3–6)."
            ),
            "n_legacy_imputed_cells": n_imp,
            "pct_legacy_imputed": n_imp / n_total * 100,
            "observation_mask": str(mask_path),
        },
        "linkedomics_note": (
            "Cannot reproduce the final matrix byte-for-byte from the public "
            "LinkedOmics HS_CPTAC_BRCA_2018_RNA_GENE.cct export (affine scale "
            "difference); laboratory cptac.csv is the authoritative pre-imputation input."
        ),
        "sample_ids": {
            "status": "recovered_Patient_ID",
            "biological_ids_available": True,
            "mapping_audit": str(map_path),
        },
        "fingerprint": fp,
        "generated_at_utc": stamp,
    }
    (OUT_PROC / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Update inventory pointer fields lightly
    inv_path = OUT_PROC / "discovery_inventory.json"
    if inv_path.exists():
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
    else:
        inv = {}
    inv.update(
        {
            "cohort_key": "discovery",
            "scientific_name_policy": metadata["scientific_identity"],
            "sample_ids": metadata["sample_ids"],
            "final_matrix": {"path": str(final_path), "fingerprint": fp},
            "observation_mask": str(mask_path),
            "provenance": {
                "confidence": "High — CPTAC-derived laboratory cohort",
                "scientific_identity": metadata["scientific_identity"],
                "linkedomics_note": metadata["linkedomics_note"],
            },
        }
    )
    inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")

    prep_report = {
        "generated_at_utc": stamp,
        "status": "COMPLETE",
        "n": 117,
        "class_distribution": fp["class_distribution"],
        "sample_id_recovery": {
            "n_recovered": int(len(map_df)),
            "n_unique": int(map_df["recovered_Patient_ID"].nunique()),
            "n_with_discrepancy": int((map_df["discrepancy"] != "").sum()),
            "mapping_file": str(map_path),
        },
        "observation_mask": {
            "n_observed": n_obs,
            "n_imputed": n_imp,
            "pct_imputed": n_imp / n_total * 100,
            "path": str(mask_path),
        },
        "fingerprint": fp["values_and_labels_sha256"],
        "artifacts": {
            "matrix": str(final_path),
            "mask": str(mask_path),
            "metadata": str(OUT_PROC / "metadata.json"),
            "fingerprint": str(OUT_PROC / "fingerprint.json"),
            "id_mapping": str(map_path),
        },
    }
    (OUT_ART / "discovery_preparation_report.json").write_text(
        json.dumps(prep_report, indent=2), encoding="utf-8"
    )
    md = [
        "# Discovery preparation report (CPTAC-derived laboratory cohort)",
        "",
        f"- Generated (UTC): `{stamp}`",
        "- Cohort key: **`discovery`**",
        "- Scientific identity: **CPTAC-derived laboratory discovery cohort**",
        "",
        "## Dimensions",
        f"- n=**117**, genes=**50**",
        f"- Classes: `{fp['class_distribution']}`",
        "",
        "## Sample IDs",
        f"- Recovered **{len(map_df)}/117** Patient_IDs (1:1)",
        f"- Discrepancy rows: **{int((map_df['discrepancy'] != '').sum())}**",
        f"- Audit: `{map_path}`",
        "",
        "## Observation mask",
        f"- Observed: **{n_obs}** ({n_obs/n_total*100:.2f}%)",
        f"- Legacy-imputed: **{n_imp}** ({n_imp/n_total*100:.2f}%)",
        f"- Mask: `{mask_path}`",
        "",
        "## Fingerprint",
        f"- `{fp['values_and_labels_sha256']}`",
        "",
        "## Notes",
        "- Not a byte-identical LinkedOmics export.",
        "- Primary benchmark metrics must target originally observed cells only.",
        "",
    ]
    (OUT_ART / "discovery_preparation_report.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    print(json.dumps(prep_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
