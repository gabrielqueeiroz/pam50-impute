"""Experimental protocol configuration (shared across cohorts)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

PAM50_GENES: list[str] = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]

TARGET_LABELS = ("LumA", "LumB", "Her2", "Basal")

# Full-protocol RFECA variants (use with --only-rfeca re-runs).
RFECA_ONLY_IMPUTERS: list[str] = [
    "RFECA_SVR(k=5)",
    "RFECA_SVR(k=10)",
    "RFECA_SVR(k=20)",
]

# Faithful Original RFECA/RFACA (parallel module; distinct from legacy RFECA_SVR).
ORIGINAL_RFECA_RFACA_IMPUTERS: list[str] = [
    "OriginalRFECA",
    "OriginalRFACA",
]

# Principal dissertation method (RFACA kept for smoke/audit only).
ORIGINAL_RFECA_ONLY_IMPUTERS: list[str] = [
    "OriginalRFECA",
]


@dataclass
class ExperimentConfig:
    """Single source of truth for the experimental protocol."""

    cohort_name: str = "metabric"
    missing_rates: list[float] = field(default_factory=lambda: [0.10])
    n_repetitions: int = 2
    n_splits: int = 3
    random_state: int = 42
    # Primary classifier used in the study tables / paper focus.
    primary_classifier: str = "EnsembleSoft"
    imputers: list[str] = field(
        default_factory=lambda: ["SimpleMean", "KNN(k=5,dist)", "RFECA_SVR(k=5)"]
    )
    rfeca_top_k: int = 5
    rfeca_use_abs_corr: bool = True
    rfeca_kernel: str = "linear"
    rfeca_C: float = 1.0
    rfeca_epsilon: float = 0.1
    rfeca_min_train_samples: int = 10
    # Faithful Original RFECA/RFACA (src/bcimpute/imputation_original/).
    # Defaults unused unless those names appear in ``imputers``.
    original_rfeca_validation: str = "kfold"  # loocv | kfold | stratified_kfold
    original_rfeca_n_splits: int = 5
    original_rfeca_kernel: str = "linear"
    original_rfeca_C: float = 1.0
    original_rfeca_epsilon: float = 0.1
    # Notebook SVR has no StandardScaler; principal experiment uses False.
    original_rfeca_use_scaler: bool = False
    original_rfeca_min_train_samples: int = 10
    # Cap on ranked candidate pool (PAM50 → up to 49 other genes).
    original_rfeca_max_candidates: int | None = 49
    original_rfeca_selection_protocol: str = "leakage_safe"
    # MissForest-like (IterativeImputer + ExtraTreesRegressor)
    missforest_n_estimators: int = 20
    missforest_max_iter: int = 5
    # Threads inside ExtraTrees only (outer CV stays sequential). Cap for laptop safety.
    missforest_n_jobs: int = 4
    # Forbid any precomputed full-cohort correlation file.
    allow_precomputed_correlation: bool = False
    n_jobs: int = 1
    tag: str = "smoke"
    # Primary: only originally observed cells may be artificial-missing targets.
    # Sensitivity (non-default): "all_complete_cells"
    target_cell_policy: str = "originally_observed_only"
    # Missingness mechanism for artificial masks (MCAR now; MAR next).
    missingness_mechanism: str = "mcar"
    # Seed scheme for artificial masks: "v2" (collision-free) or "legacy".
    missingness_seed_scheme: str = "v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.to_dict())


def smoke_metabric_config() -> ExperimentConfig:
    return ExperimentConfig(
        cohort_name="metabric",
        missing_rates=[0.10],
        n_repetitions=2,
        n_splits=3,
        random_state=42,
        primary_classifier="EnsembleSoft",
        imputers=["SimpleMean", "KNN(k=5,dist)", "RFECA_SVR(k=5)", "MissForest"],
        rfeca_top_k=5,
        missforest_n_estimators=10,
        missforest_max_iter=3,
        missforest_n_jobs=4,
        tag="smoke",
        n_jobs=1,
        missingness_mechanism="mcar",
    )


def full_benchmark_config(
    cohort_name: str,
    *,
    missingness_mechanism: str = "mcar",
) -> ExperimentConfig:
    """Full protocol (MissForest included). Set missingness_mechanism to 'mar' for MAR."""
    mechanism = missingness_mechanism.lower().strip()
    if mechanism not in {"mcar", "mar"}:
        raise ValueError(f"Unknown missingness_mechanism: {mechanism!r}")
    tag = "full" if mechanism == "mcar" else f"full_{mechanism}"
    return ExperimentConfig(
        cohort_name=cohort_name,
        missing_rates=[0.0, 0.05, 0.10, 0.20, 0.30],
        n_repetitions=10,
        n_splits=5,
        random_state=42,
        primary_classifier="EnsembleSoft",
        imputers=[
            "SimpleMean",
            "KNN(k=5,dist)",
            "RFECA_SVR(k=5)",
            "RFECA_SVR(k=10)",
            "RFECA_SVR(k=20)",
            "MissForest",
        ],
        missforest_n_estimators=20,
        missforest_max_iter=5,
        missforest_n_jobs=4,
        tag=tag,
        n_jobs=1,
        missingness_mechanism=mechanism,
        missingness_seed_scheme="v2",
    )


def apply_rfeca_only(cfg: ExperimentConfig) -> ExperimentConfig:
    """Restrict imputers to the three RFECA(k) variants; retag for artifacts."""
    cfg.imputers = list(RFECA_ONLY_IMPUTERS)
    if "rfeca" not in cfg.tag:
        cfg.tag = f"{cfg.tag}_rfeca"
    return cfg


def apply_original_rfeca_rfaca_only(cfg: ExperimentConfig) -> ExperimentConfig:
    """Restrict imputers to faithful OriginalRFECA + OriginalRFACA only."""
    cfg.imputers = list(ORIGINAL_RFECA_RFACA_IMPUTERS)
    if "original_rfeca_rfaca" not in cfg.tag:
        cfg.tag = f"{cfg.tag}_original_rfeca_rfaca"
    return cfg


def apply_original_rfeca_only(cfg: ExperimentConfig) -> ExperimentConfig:
    """Principal experiment: OriginalRFECA only (no RFACA, no legacy)."""
    cfg.imputers = list(ORIGINAL_RFECA_ONLY_IMPUTERS)
    cfg.original_rfeca_use_scaler = False
    cfg.original_rfeca_max_candidates = 49
    cfg.original_rfeca_selection_protocol = "leakage_safe"
    cfg.original_rfeca_kernel = "linear"
    if cfg.cohort_name == "discovery":
        cfg.original_rfeca_validation = "loocv"
    else:
        cfg.original_rfeca_validation = "kfold"
        cfg.original_rfeca_n_splits = 5
    marker = "original_rfeca_leakage_safe_no_scaler_maxcand49"
    if marker not in cfg.tag:
        cfg.tag = f"{cfg.tag}_{marker}"
    return cfg


def with_original_rfeca_rfaca(cfg: ExperimentConfig) -> ExperimentConfig:
    """Append OriginalRFECA/OriginalRFACA without removing legacy imputers."""
    for name in ORIGINAL_RFECA_RFACA_IMPUTERS:
        if name not in cfg.imputers:
            cfg.imputers.append(name)
    if "with_original" not in cfg.tag:
        cfg.tag = f"{cfg.tag}_with_original"
    return cfg


COHORT_PATHS: dict[str, Path] = {
    "discovery": (
        ROOT / "data" / "processed" / "discovery" / "discovery_pam50_4class.csv"
    ),
    "metabric": (
        ROOT / "data" / "processed" / "metabric" / "metabric_pam50_4class.csv"
    ),
}


def smoke_discovery_config(
    *,
    missingness_mechanism: str = "mcar",
) -> ExperimentConfig:
    """Same smoke protocol as METABRIC, pointed at the discovery cohort."""
    mechanism = missingness_mechanism.lower().strip()
    if mechanism not in {"mcar", "mar"}:
        raise ValueError(f"Unknown missingness_mechanism: {mechanism!r}")
    tag = "smoke" if mechanism == "mcar" else f"smoke_{mechanism}"
    return ExperimentConfig(
        cohort_name="discovery",
        missing_rates=[0.10],
        n_repetitions=2,
        n_splits=3,
        random_state=42,
        primary_classifier="EnsembleSoft",
        imputers=["SimpleMean", "KNN(k=5,dist)", "RFECA_SVR(k=5)", "MissForest"],
        rfeca_top_k=5,
        missforest_n_estimators=10,
        missforest_max_iter=3,
        missforest_n_jobs=4,
        tag=tag,
        n_jobs=1,
        target_cell_policy="originally_observed_only",
        missingness_mechanism=mechanism,
        missingness_seed_scheme="v2",
    )


DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACT_ROOT = ROOT / "artifacts"
ARCHIVE_ROOT = ROOT / "archive"
NOTEBOOKS_ROOT = ROOT / "notebooks"
SCRIPTS_ROOT = ROOT / "scripts"
EXPERIMENTS_ROOT = ROOT / "experiments"
