"""Faithful Original RFECA (correlation prefixes + sklearn RFE), TARGET-WISE."""

from __future__ import annotations

from .base import BaseOriginalCorrelationImputer
from .selection import SelectorKind
from .target_wise import INPUT_PROTOCOL


class OriginalRFECAImputer(BaseOriginalCorrelationImputer):
    """
    Original-style RFECA with TARGET-WISE complete predictors:

      Predictors always come from the original complete matrix.
      Artificial missingness is applied only to the target gene column
      (positions from the persisted multivariate mask).
    """

    method_name = "OriginalRFECA"
    selector_kind = "RFE"

    def __init__(self, **kwargs):
        kwargs.setdefault("input_protocol", INPUT_PROTOCOL)
        kwargs.setdefault("use_scaler", False)
        kwargs.setdefault("max_candidates", 49)
        kwargs.setdefault("selection_protocol", "leakage_safe")
        kwargs.setdefault("candidate_rule", "full_matrix")
        super().__init__(**kwargs)

    def _selector_kind(self) -> SelectorKind:
        return "RFE"
