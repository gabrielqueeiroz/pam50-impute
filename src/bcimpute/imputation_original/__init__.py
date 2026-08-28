"""Faithful Original RFECA / RFACA imputers (parallel to legacy RFECAImputerSVR)."""

from .rfaca import OriginalRFACAImputer
from .rfeca import OriginalRFECAImputer
from .types import FallbackEvent, FitAudit, GeneAuditRecord, SubsetEvaluation
from .utils import (
    correlation_prefixes,
    rfaca_subsets,
    rfeca_subsets,
    safe_mkdir,
    write_json,
)

# Registry names used in ExperimentConfig.imputers / CSV "imputer" column.
# Distinct from legacy "RFECA_SVR(k=*)" identifiers.
ORIGINAL_RFECA_NAME = "OriginalRFECA"
ORIGINAL_RFACA_NAME = "OriginalRFACA"
ORIGINAL_IMPUTER_NAMES = (ORIGINAL_RFECA_NAME, ORIGINAL_RFACA_NAME)

__all__ = [
    "OriginalRFECAImputer",
    "OriginalRFACAImputer",
    "ORIGINAL_RFECA_NAME",
    "ORIGINAL_RFACA_NAME",
    "ORIGINAL_IMPUTER_NAMES",
    "FitAudit",
    "GeneAuditRecord",
    "SubsetEvaluation",
    "FallbackEvent",
    "rfeca_subsets",
    "rfaca_subsets",
    "correlation_prefixes",
    "safe_mkdir",
    "write_json",
]
