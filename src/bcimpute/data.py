"""Generic cohort loading for discovery and METABRIC matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import COHORT_PATHS, PAM50_GENES, ROOT, TARGET_LABELS


@dataclass
class CohortData:
    """In-memory cohort with preserved IDs, gene order, labels, and metadata."""

    name: str
    sample_ids: pd.Index
    X: pd.DataFrame
    y: pd.Series
    gene_names: list[str]
    label_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # True = cell was originally observed (not completed by a prior lab imputation).
    originally_observed_mask: pd.DataFrame | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def class_distribution(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.y.value_counts().items()}

    def X_values_copy(self) -> pd.DataFrame:
        """Defensive copy of the complete expression matrix."""
        return self.X.copy()


def _load_inventory(path: Path, cohort_key: str) -> dict[str, Any]:
    candidates = [
        path.parent / f"{cohort_key}_inventory.json",
        path.parent / "metabric_inventory.json",
        path.parent / "discovery_inventory.json",
    ]
    for cand in candidates:
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}


def _normalize_processed_matrix(
    df: pd.DataFrame,
    path: Path,
    cohort_key: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Index, dict]:
    """
    Shared schema for analysis-ready matrices:
      sample_id | <50 PAM50 genes> | PAM50
    """
    required = {"sample_id", "PAM50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cohort_key} matrix missing columns {missing}: {path}")

    sample_ids = df["sample_id"].astype(str)
    y = df["PAM50"].astype(str)
    X = df.drop(columns=["sample_id", "PAM50"])

    inv = _load_inventory(path, cohort_key)
    meta: dict[str, Any] = {
        "source_path": str(path.resolve()),
        "has_native_sample_id_column": True,
        "cohort_key": cohort_key,
        "matrix_schema": "sample_id + PAM50_genes + PAM50",
    }
    if inv:
        # Keep a compact, parallel metadata block for both cohorts.
        meta["inventory"] = {
            "cohort_key": inv.get("cohort_key", cohort_key),
            "provenance": inv.get("provenance") or inv.get("study"),
            "expression_profile_documentation": inv.get(
                "expression_profile_documentation"
            )
            or inv.get("source_files"),
            "sample_ids": inv.get("sample_ids"),
            "final_matrix": inv.get("final_matrix"),
            "scientific_name_policy": inv.get("scientific_name_policy"),
        }
    return X, y, pd.Index(sample_ids, name="sample_id"), meta


def load_cohort(name: str, path: Path | None = None) -> CohortData:
    """
    Generic loader for discovery and METABRIC cohorts.

    Both cohorts return the same CohortData schema/metadata fields.
    """
    key = name.lower().strip()
    aliases = {
        "discovery_cohort": "discovery",
        "lab": "discovery",
        "laboratory": "discovery",
    }
    key = aliases.get(key, key)

    if path is None:
        if key not in COHORT_PATHS:
            raise KeyError(f"Unknown cohort {name!r}. Known: {sorted(COHORT_PATHS)}")
        path = COHORT_PATHS[key]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if key in {"discovery", "metabric"}:
        df = pd.read_csv(path)
        X, y, sample_ids, meta = _normalize_processed_matrix(df, path, key)
        cohort_name = key
    else:
        df_try = pd.read_csv(path)
        if "sample_id" in df_try.columns:
            X, y, sample_ids, meta = _normalize_processed_matrix(df_try, path, key)
            cohort_name = key
        else:
            raise ValueError(
                f"Unsupported cohort file schema for {path}. "
                "Expected columns: sample_id, PAM50 genes, PAM50."
            )

    missing_genes = [g for g in PAM50_GENES if g not in X.columns]
    if missing_genes:
        raise ValueError(f"Cohort {cohort_name} missing PAM50 genes: {missing_genes}")

    X = X[PAM50_GENES].astype(float)
    X.index = sample_ids
    y = pd.Series(y.to_numpy(), index=sample_ids, name="PAM50")

    labels = set(y.unique())
    if not labels.issubset(set(TARGET_LABELS)):
        raise ValueError(f"Unexpected labels in {cohort_name}: {sorted(labels)}")

    if X.isna().any().any():
        raise ValueError(f"Complete matrix for {cohort_name} contains missing values.")

    if sample_ids.duplicated().any():
        raise ValueError(f"Duplicated sample IDs in {cohort_name}.")

    # Parallel metadata fields for every cohort
    meta.update(
        {
            "cohort_name": cohort_name,
            "n_samples": int(len(X)),
            "n_genes": int(X.shape[1]),
            "gene_order": list(PAM50_GENES),
            "label_name": "PAM50",
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "project_root": str(ROOT),
            "sample_id_status": (meta.get("inventory") or {}).get("sample_ids"),
        }
    )

    # Optional cell-provenance mask (aligned to X). Default: all observed.
    obs_mask = _load_observation_mask(path, cohort_name, X)
    meta["n_originally_observed_cells"] = int(obs_mask.to_numpy().sum())
    meta["n_legacy_imputed_cells"] = int((~obs_mask.to_numpy()).sum())
    meta["observation_mask_policy"] = (
        "file" if (path.parent / f"{cohort_name}_originally_observed_mask.csv").exists()
        else "all_observed_default"
    )

    # Prefer processed metadata.json when present (discovery scientific identity).
    meta_json = path.parent / "metadata.json"
    if meta_json.exists() and cohort_name == "discovery":
        try:
            meta["scientific_identity"] = json.loads(
                meta_json.read_text(encoding="utf-8")
            ).get("scientific_identity")
        except Exception:  # noqa: BLE001
            pass
    if "scientific_identity" not in meta:
        if cohort_name == "discovery":
            meta["scientific_identity"] = "CPTAC-derived laboratory discovery cohort"
        elif cohort_name == "metabric":
            meta["scientific_identity"] = "METABRIC PAM50 4-class cohort"

    return CohortData(
        name=cohort_name,
        sample_ids=sample_ids,
        X=X,
        y=y,
        gene_names=list(PAM50_GENES),
        label_name="PAM50",
        metadata=meta,
        originally_observed_mask=obs_mask,
    )


def _load_observation_mask(
    matrix_path: Path, cohort_key: str, X: pd.DataFrame
) -> pd.DataFrame:
    """
    Load originally_observed_mask if present; otherwise all-True (fully observed).
    """
    candidates = [
        matrix_path.parent / f"{cohort_key}_originally_observed_mask.csv",
    ]
    if cohort_key == "discovery":
        candidates.append(matrix_path.parent / "discovery_originally_observed_mask.csv")
    for cand in candidates:
        if not cand.exists():
            continue
        raw = pd.read_csv(cand)
        if "sample_id" not in raw.columns:
            raise ValueError(f"Observation mask missing sample_id: {cand}")
        raw["sample_id"] = raw["sample_id"].astype(str)
        raw = raw.set_index("sample_id")
        # Coerce to bool (CSV may store True/False or 1/0)
        for c in PAM50_GENES:
            if c not in raw.columns:
                raise ValueError(f"Observation mask missing gene {c}: {cand}")
        mask = raw[PAM50_GENES].astype(bool)
        mask = mask.reindex(index=X.index.astype(str))
        if mask.isna().any().any():
            raise ValueError(f"Observation mask not aligned to X for {cohort_key}")
        if list(mask.columns) != list(PAM50_GENES):
            mask = mask[PAM50_GENES]
        return mask

    # Default: all cells treated as originally observed (e.g. METABRIC).
    return pd.DataFrame(True, index=X.index, columns=list(PAM50_GENES))
