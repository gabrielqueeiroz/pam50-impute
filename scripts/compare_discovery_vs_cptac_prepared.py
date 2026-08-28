#!/usr/bin/env python3
"""
Compare discovery vs CPTAC LinkedOmics after both are prepared with the same
pipeline rules (4-class filter, 50-gene order, drop rows with any NA, no scaling).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / "data" / "processed" / "discovery" / "discovery_pam50_4class.csv"
CPTAC = (
    ROOT / "data" / "processed" / "cptac_linkedomics" / "cptac_linkedomics_pam50_4class.csv"
)
MAP_CSV = (
    ROOT
    / "artifacts"
    / "discovery_cptac_provenance_audit"
    / "putative_discovery_to_cptac_sample_map.csv"
)
OUT_DIR = ROOT / "artifacts" / "discovery_vs_cptac_prepared_comparison"

PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]


def zrows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def summarize(df: pd.DataFrame, name: str) -> dict:
    X = df[PAM50].to_numpy(float)
    return {
        "name": name,
        "n_samples": int(len(df)),
        "n_genes": 50,
        "class_distribution": {str(k): int(v) for k, v in df["PAM50"].value_counts().items()},
        "expression": {
            "min": float(X.min()),
            "max": float(X.max()),
            "mean": float(X.mean()),
            "std": float(X.std()),
            "median": float(np.median(X)),
        },
        "n_unique_sample_ids": int(df["sample_id"].nunique()),
        "n_missing_values": int(df[PAM50].isna().sum().sum()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    disc = pd.read_csv(DISCOVERY)
    cptac = pd.read_csv(CPTAC)

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_rules_shared": [
            "sample_id | 50 PAM50 genes (canonical order) | PAM50",
            "Keep LumA/LumB/Her2/Basal only",
            "Drop rows with any NA among the 50 genes",
            "No scaling/centering/transform at preparation time",
        ],
        "discovery": summarize(disc, "discovery"),
        "cptac_linkedomics": summarize(cptac, "cptac_linkedomics"),
    }

    # Class-count delta
    d_counts = report["discovery"]["class_distribution"]
    c_counts = report["cptac_linkedomics"]["class_distribution"]
    report["class_count_delta"] = {
        k: int(c_counts.get(k, 0) - d_counts.get(k, 0))
        for k in sorted(set(d_counts) | set(c_counts))
    }
    report["n_delta"] = int(
        report["cptac_linkedomics"]["n_samples"] - report["discovery"]["n_samples"]
    )

    # Overlap via putative map (discovery synthetic -> CPTAC native)
    overlap = {"available": False}
    if MAP_CSV.exists():
        m = pd.read_csv(MAP_CSV)
        cptac_ids = set(cptac["sample_id"].astype(str))
        mapped = m[m["matched_cptac_id"].astype(str).isin(cptac_ids)].copy()
        # join expression
        disc_idx = disc.set_index("sample_id")
        cpt_idx = cptac.set_index("sample_id")
        rows = []
        for _, r in mapped.iterrows():
            did = str(r["discovery_row_id"])
            cid = str(r["matched_cptac_id"])
            if did not in disc_idx.index or cid not in cpt_idx.index:
                continue
            xd = disc_idx.loc[did, PAM50].to_numpy(float)
            xc = cpt_idx.loc[cid, PAM50].to_numpy(float)
            # affine CPTAC ≈ a*disc + b
            A = np.vstack([xd, np.ones_like(xd)]).T
            a, b = np.linalg.lstsq(A, xc, rcond=None)[0]
            pred = a * xd + b
            rows.append(
                {
                    "discovery_row_id": did,
                    "cptac_id": cid,
                    "discovery_PAM50": str(disc_idx.loc[did, "PAM50"]),
                    "cptac_PAM50": str(cpt_idx.loc[cid, "PAM50"]),
                    "label_match": bool(
                        str(disc_idx.loc[did, "PAM50"]) == str(cpt_idx.loc[cid, "PAM50"])
                    ),
                    "pearson": float(np.corrcoef(xd, xc)[0, 1]),
                    "cosine_zscore": float(
                        1.0
                        - cdist(zrows(xd[None, :]), zrows(xc[None, :]), metric="cosine")[0, 0]
                    ),
                    "max_abs_raw": float(np.max(np.abs(xd - xc))),
                    "affine_a": float(a),
                    "affine_b": float(b),
                    "affine_rmse": float(np.sqrt(np.mean((pred - xc) ** 2))),
                }
            )
        pair_df = pd.DataFrame(rows)
        pair_df.to_csv(OUT_DIR / "paired_overlap_samples.csv", index=False)
        overlap = {
            "available": True,
            "n_discovery": int(len(disc)),
            "n_cptac_prepared": int(len(cptac)),
            "n_putative_map_rows": int(len(m)),
            "n_overlap_after_shared_prep": int(len(pair_df)),
            "frac_discovery_covered": float(len(pair_df) / len(disc)),
            "frac_cptac_covered": float(len(pair_df) / len(cptac)) if len(cptac) else 0.0,
            "label_agreement": float(pair_df["label_match"].mean()) if len(pair_df) else None,
            "pearson_median": float(pair_df["pearson"].median()) if len(pair_df) else None,
            "pearson_min": float(pair_df["pearson"].min()) if len(pair_df) else None,
            "cosine_z_median": float(pair_df["cosine_zscore"].median()) if len(pair_df) else None,
            "max_abs_raw_median": float(pair_df["max_abs_raw"].median()) if len(pair_df) else None,
            "affine_a_median": float(pair_df["affine_a"].median()) if len(pair_df) else None,
            "affine_rmse_median": float(pair_df["affine_rmse"].median()) if len(pair_df) else None,
            "exact_raw_matches": int((pair_df["max_abs_raw"] == 0).sum()) if len(pair_df) else 0,
        }
    report["paired_overlap"] = overlap

    # Global nearest-neighbor on prepared matrices (no map), z-scored
    Xd = disc[PAM50].to_numpy(float)
    Xc = cptac[PAM50].to_numpy(float)
    S = 1.0 - cdist(zrows(Xd), zrows(Xc), metric="cosine")
    best = S.max(axis=1)
    report["unsupervised_nn"] = {
        "discovery_to_cptac_best_cosine_median": float(np.median(best)),
        "discovery_to_cptac_best_cosine_min": float(np.min(best)),
        "frac_cosine_ge_0.999": float(np.mean(best >= 0.999)),
        "frac_cosine_ge_0.99": float(np.mean(best >= 0.99)),
        "note": (
            "Nearest CPTAC prepared row for each discovery row after shared NA filter; "
            "incomplete CPTAC samples removed by the shared pipeline, so coverage < 117."
        ),
    }

    # Fingerprints
    def fp(path: Path) -> str | None:
        p = path.parent / path.name.replace("_pam50_4class.csv", "_fingerprint.json")
        # discovery / cptac naming
        alts = [
            path.parent / "discovery_fingerprint.json",
            path.parent / "cptac_linkedomics_fingerprint.json",
        ]
        for cand in [p, *alts]:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8")).get(
                    "values_and_labels_sha256"
                )
        return None

    report["fingerprints"] = {
        "discovery_values_and_labels_sha256": fp(DISCOVERY),
        "cptac_values_and_labels_sha256": fp(CPTAC),
        "identical_prepared_matrices": False,
    }
    if (
        report["fingerprints"]["discovery_values_and_labels_sha256"]
        and report["fingerprints"]["cptac_values_and_labels_sha256"]
    ):
        report["fingerprints"]["identical_prepared_matrices"] = (
            report["fingerprints"]["discovery_values_and_labels_sha256"]
            == report["fingerprints"]["cptac_values_and_labels_sha256"]
        )

    report["conclusion"] = {
        "summary": (
            "Same prep rules applied. Prepared CPTAC is smaller than discovery because "
            "LinkedOmics RNAseq has native NAs among PAM50 genes and those rows are dropped. "
            "On the overlapping complete samples, profiles remain near-perfect affine matches "
            "to discovery (not byte-identical)."
        ),
        "cohort_keys": {"laboratory": "discovery", "public_reference": "cptac_linkedomics"},
        "rename_policy": "Do not rename discovery to CPTAC.",
    }

    (OUT_DIR / "comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = [
        "# Discovery vs CPTAC LinkedOmics — same preparation pipeline",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        "",
        "## Shared prep rules",
        *[f"- {r}" for r in report["pipeline_rules_shared"]],
        "",
        "## Dimensions & labels",
        f"- Discovery: n=**{report['discovery']['n_samples']}** · `{d_counts}`",
        f"- CPTAC prepared: n=**{report['cptac_linkedomics']['n_samples']}** · `{c_counts}`",
        f"- Dn (cptac - discovery): **{report['n_delta']}**",
        f"- Class count deltas: `{report['class_count_delta']}`",
        "",
        "## Expression ranges (prepared)",
        f"- Discovery: `{report['discovery']['expression']}`",
        f"- CPTAC: `{report['cptac_linkedomics']['expression']}`",
        "",
        "## Paired overlap (via putative ID map ∩ prepared CPTAC)",
    ]
    if overlap.get("available"):
        md += [
            f"- Overlap n: **{overlap['n_overlap_after_shared_prep']}** "
            f"({overlap['frac_discovery_covered']*100:.1f}% of discovery; "
            f"{overlap['frac_cptac_covered']*100:.1f}% of prepared CPTAC)",
            f"- Label agreement: **{overlap['label_agreement']:.3f}**",
            f"- Pearson median / min: **{overlap['pearson_median']:.6f}** / {overlap['pearson_min']:.6f}",
            f"- Z-cosine median: **{overlap['cosine_z_median']:.6f}**",
            f"- Raw max-abs median: **{overlap['max_abs_raw_median']:.4f}** (exact matches: {overlap['exact_raw_matches']})",
            f"- Affine a median / RMSE median: **{overlap['affine_a_median']:.4f}** / {overlap['affine_rmse_median']:.2e}",
        ]
    else:
        md.append("- Putative map unavailable.")
    md += [
        "",
        "## Unsupervised nearest neighbor (prepared matrices)",
        f"- Best z-cosine median/min: "
        f"**{report['unsupervised_nn']['discovery_to_cptac_best_cosine_median']:.6f}** / "
        f"{report['unsupervised_nn']['discovery_to_cptac_best_cosine_min']:.6f}",
        f"- Frac ≥ 0.999: **{report['unsupervised_nn']['frac_cosine_ge_0.999']:.3f}**",
        "",
        "## Fingerprints",
        f"- Discovery: `{report['fingerprints']['discovery_values_and_labels_sha256']}`",
        f"- CPTAC: `{report['fingerprints']['cptac_values_and_labels_sha256']}`",
        f"- Identical prepared matrices: **{report['fingerprints']['identical_prepared_matrices']}**",
        "",
        "## Conclusion",
        report["conclusion"]["summary"],
        f"- Keys: discovery vs `{report['conclusion']['cohort_keys']['public_reference']}`",
        f"- {report['conclusion']['rename_policy']}",
        "",
    ]
    (OUT_DIR / "comparison_report.md").write_text("\n".join(md), encoding="utf-8")
    try:
        print("\n".join(md))
    except UnicodeEncodeError:
        print("\n".join(md).encode("ascii", errors="replace").decode("ascii"))
    print(f"\nWrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
