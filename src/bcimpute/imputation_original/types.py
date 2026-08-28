"""Shared types for faithful Original RFECA / RFACA imputers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ValidationStrategy = Literal["loocv", "kfold", "stratified_kfold"]


@dataclass
class FoldSelectionDetail:
    """Per-fold selection trace (for leakage / variability audits)."""

    fold: int
    n_train: int
    n_val: int
    correlation_order: list[str]
    prefix_genes: list[str]
    selected_features: list[str]


@dataclass
class SubsetEvaluation:
    """OOF evaluation of one correlation-prefix length (after RFE/RFA)."""

    prefix_len: int
    prefix_genes_final: list[str]
    selected_features_final: list[str]
    n_predictors: int
    rmse: float
    n_oof_predictions: int
    fold_details: list[FoldSelectionDetail] = field(default_factory=list)

    @property
    def predictors(self) -> list[str]:
        return list(self.selected_features_final)


@dataclass
class FallbackEvent:
    """Recorded whenever column-mean fallback is used instead of SVR."""

    gene: str
    reason: str
    n_observed: int
    n_candidates: int
    prefix_len: int | None
    n_rows_affected: int
    dataset: str = ""
    mechanism: str = ""
    missing_rate: float | None = None
    replicate: int | None = None
    exception_type: str = ""
    exception_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneAuditRecord:
    gene: str
    n_observed: int
    n_missing: int
    candidates: list[str]
    correlation_abs_order: list[str]
    correlation_values: dict[str, float]
    subsets_evaluated: list[SubsetEvaluation]
    winning_prefix_len: int
    winning_prefix_genes: list[str]
    winning_predictors: list[str]
    n_predictors_selected: int
    validation_strategy: str
    validation_n_splits: int | None
    svr_params: dict[str, Any]
    selection_seconds: float
    final_fit_seconds: float
    selector_kind: str = "RFE"
    n_prefixes_evaluated: int = 0
    n_rfe_fits: int = 0
    n_svr_fits: int = 0
    impute_seconds: float = 0.0
    status: str = "ok"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["subsets_evaluated"] = []
        for s in self.subsets_evaluated:
            sd = asdict(s)
            sd["predictors"] = list(s.selected_features_final)
            d["subsets_evaluated"].append(sd)
        return d


@dataclass
class FitAudit:
    """Collected during fit(); optional persist via write helpers."""

    method: str
    validation_strategy: str
    random_state: int
    selection_protocol: str = "leakage_safe"
    use_scaler: bool = False
    svr_kernel: str = "linear"
    max_candidates: int | None = 49
    genes: list[GeneAuditRecord] = field(default_factory=list)
    fallback_events: list[FallbackEvent] = field(default_factory=list)
    n_rfe_fits_total: int = 0
    n_svr_fits_total: int = 0
    exceptions: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "validation_strategy": self.validation_strategy,
            "random_state": self.random_state,
            "selection_protocol": self.selection_protocol,
            "use_scaler": self.use_scaler,
            "svr_kernel": self.svr_kernel,
            "max_candidates": self.max_candidates,
            "n_rfe_fits_total": self.n_rfe_fits_total,
            "n_svr_fits_total": self.n_svr_fits_total,
            "fallback_events": [e.to_dict() for e in self.fallback_events],
            "exceptions": list(self.exceptions),
            "genes": [g.to_dict() for g in self.genes],
        }
