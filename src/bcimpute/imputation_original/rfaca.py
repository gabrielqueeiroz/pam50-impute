"""Faithful Original RFACA (correlation prefixes + RecursiveFeatureAddition)."""

from __future__ import annotations

from .base import BaseOriginalCorrelationImputer
from .selection import SelectorKind


class OriginalRFACAImputer(BaseOriginalCorrelationImputer):
    """
    Original-style RFACA (smoke/audit only).

    Defaults to multivariate_masked input for historical smoke tests; the
    principal dissertation method is OriginalRFECA TARGET-WISE.
    """

    method_name = "OriginalRFACA"
    selector_kind = "RFA"

    def __init__(self, **kwargs):
        kwargs.setdefault("input_protocol", "multivariate_masked")
        kwargs.setdefault("use_scaler", False)
        super().__init__(**kwargs)

    def _selector_kind(self) -> SelectorKind:
        return "RFA"
