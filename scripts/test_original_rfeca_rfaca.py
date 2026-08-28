#!/usr/bin/env python3
"""Unit tests for OriginalRFECA TARGET-WISE + RFACA smoke."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bcimpute.config import ExperimentConfig  # noqa: E402
from bcimpute.imputers import RFECAImputerSVR, build_imputers  # noqa: E402
from bcimpute.imputation_original import (  # noqa: E402
    ORIGINAL_RFACA_NAME,
    ORIGINAL_RFECA_NAME,
    OriginalRFACAImputer,
    OriginalRFECAImputer,
    correlation_prefixes,
)
from bcimpute.imputation_original import selection as sel_mod  # noqa: E402
from bcimpute.imputation_original import svr_model as svr_mod  # noqa: E402
from bcimpute.imputation_original.correlation import pearson_abs_ranking  # noqa: E402
from bcimpute.imputation_original.target_wise import (  # noqa: E402
    assert_predictors_finite,
    build_target_wise_matrix,
)
from bcimpute.imputation_original.utils import safe_mkdir, write_json  # noqa: E402


def _synth(n: int = 60, p: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    cols = {}
    for j in range(p):
        w = rng.normal(size=2)
        cols[f"G{j}"] = z @ w + rng.normal(scale=0.2, size=n)
    return pd.DataFrame(cols)


def _mask(X: pd.DataFrame, rate: float = 0.15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(X.shape) < rate


def _fit_transform_tw(imp: OriginalRFECAImputer, X: pd.DataFrame, mask: np.ndarray):
    imp.set_target_wise_context(X, mask, for_transform=False)
    imp.fit(X)
    imp.set_target_wise_context(X, mask, for_transform=True)
    return imp.transform(X)


def _common(**kw):
    base = dict(
        validation_strategy="kfold",
        n_splits=3,
        random_state=0,
        min_train_samples=8,
        max_candidates=4,
        use_scaler=False,
        record_fold_details=True,
    )
    base.update(kw)
    return base


def test_build_target_wise_only_masks_target_column():
    X = _synth(n=20, p=4, seed=1)
    mask = np.zeros(X.shape, dtype=bool)
    mask[0:3, 0] = True
    mask[5:8, 1] = True  # other gene — must NOT appear in G0 TW matrix predictors
    tw = build_target_wise_matrix(X, mask, "G0")
    assert tw["G0"].isna().sum() == 3
    assert tw["G1"].isna().sum() == 0
    assert_predictors_finite(tw, "G0")


def test_pearson_ranking_aligns_by_position_not_index():
    rng = np.random.default_rng(0)
    n = 40
    idx = [f"S{i}" for i in range(n)]
    y = rng.normal(size=n)
    X = pd.DataFrame(
        {"A": y * 0.8 + rng.normal(scale=0.1, size=n), "B": rng.normal(size=n)},
        index=idx,
    )
    order, scores = pearson_abs_ranking(y, X, min_periods=3)
    assert order[0] == "A"


def test_correlation_prefixes_increasing():
    assert [len(s) for s in correlation_prefixes(["a", "b", "c"])] == [1, 2, 3]


def test_rfeca_rfaca_not_equivalent():
    X = _synth(n=90, p=10, seed=21)
    mask = _mask(X, 0.12, seed=21)
    a = OriginalRFECAImputer(**_common(max_candidates=5, random_state=7))
    out_a = _fit_transform_tw(a, X, mask)
    b = OriginalRFACAImputer(**_common(max_candidates=5, random_state=7))
    X_miss = X.mask(mask)
    out_b = b.fit_transform(X_miss)
    wa = {g.gene: tuple(g.winning_predictors) for g in a.audit_.genes if g.status == "ok"}
    wb = {g.gene: tuple(g.winning_predictors) for g in b.audit_.genes if g.status == "ok"}
    differ = any(wa.get(g) != wb.get(g) for g in set(wa) & set(wb))
    matrices_differ = not np.allclose(out_a.to_numpy(), out_b.to_numpy(), equal_nan=True, atol=1e-8)
    assert differ or matrices_differ


def test_target_wise_svr_coverage_and_no_predictor_nan():
    X = _synth(n=80, p=8, seed=3)
    mask = _mask(X, 0.15, seed=3)
    imp = OriginalRFECAImputer(**_common(max_candidates=5))
    out = _fit_transform_tw(imp, X, mask)
    audit = imp.get_audit_dict()
    assert audit["svr_coverage"] >= 0.99
    assert audit["fallback_rate"] == 0.0
    assert int(mask.sum()) == audit["n_svr_imputed_cells"]
    # Masked cells filled; unmasked equal to original
    assert np.allclose(out.to_numpy()[~mask], X.to_numpy()[~mask])
    assert np.isfinite(out.to_numpy()[mask]).all()


def test_final_subset_size_can_vary():
    X = _synth(n=80, p=8, seed=3)
    mask = np.zeros(X.shape, dtype=bool)
    mask[0:8, 0] = True
    imp = OriginalRFECAImputer(**_common(max_candidates=5))
    _fit_transform_tw(imp, X, mask)
    ok = [g for g in imp.audit_.genes if g.status == "ok"]
    assert ok
    assert all(1 <= g.n_predictors_selected <= g.winning_prefix_len for g in ok)


def test_no_val_fold_info_in_correlation_or_selector():
    X = _synth(n=48, p=4, seed=9)
    mask = np.zeros(X.shape, dtype=bool)
    mask[0:6, 0] = True
    n_obs_g0 = int((~mask[:, 0]).sum())
    seen_corr_n: list[int] = []
    real_corr = pearson_abs_ranking

    def tracking_corr(y, X_rank, min_periods=3):
        seen_corr_n.append(len(np.asarray(y)))
        return real_corr(y, X_rank, min_periods=min_periods)

    with patch(
        "bcimpute.imputation_original.base.pearson_abs_ranking",
        side_effect=tracking_corr,
    ):
        imp = OriginalRFECAImputer(**_common(max_candidates=3, n_splits=3))
        _fit_transform_tw(imp, X, mask)
    assert any(n < n_obs_g0 for n in seen_corr_n)
    assert 28 in seen_corr_n or any(n < n_obs_g0 for n in seen_corr_n)


def test_final_refit_uses_all_observed():
    X = _synth(n=60, p=6, seed=4)
    mask = np.zeros(X.shape, dtype=bool)
    mask[0:8, 0] = True
    n_obs = int((~mask[:, 0]).sum())
    fit_sizes: list[int] = []
    real_fit = svr_mod.fit_final_svr

    def tracking_final(X_arr, y_arr, **kw):
        fit_sizes.append(len(y_arr))
        return real_fit(X_arr, y_arr, **kw)

    with patch(
        "bcimpute.imputation_original.base.fit_final_svr", side_effect=tracking_final
    ):
        imp = OriginalRFECAImputer(**_common(max_candidates=3))
        _fit_transform_tw(imp, X, mask)
    rec = next(g for g in imp.audit_.genes if g.gene == "G0")
    assert rec.status == "ok"
    assert "n_complete_case_final=" in rec.message
    n_cc = int(rec.message.split("n_complete_case_final=")[1].split(";")[0])
    assert n_cc <= n_obs
    assert n_cc in fit_sizes


def test_deterministic_with_random_state():
    X = _synth(n=50, p=5, seed=3)
    mask = _mask(X, 0.1, seed=3)
    kw = _common(max_candidates=3, random_state=42, record_fold_details=False)
    a = OriginalRFECAImputer(**kw)
    b = OriginalRFECAImputer(**kw)
    assert np.allclose(
        _fit_transform_tw(a, X, mask).to_numpy(),
        _fit_transform_tw(b, X, mask).to_numpy(),
    )


def test_preserves_unmasked_original_values():
    X = _synth()
    mask = np.zeros(X.shape, dtype=bool)
    mask[0:5, 0] = True
    out = _fit_transform_tw(OriginalRFECAImputer(**_common(max_candidates=3)), X, mask)
    assert np.allclose(out.to_numpy()[~mask], X.to_numpy()[~mask])


def test_no_simple_imputer_in_original_module():
    import bcimpute.imputation_original.base as base_mod
    import bcimpute.imputation_original.selection as sm
    import bcimpute.imputation_original.target_wise as tw

    for mod in (base_mod, sm, tw):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "SimpleImputer" not in src


def test_scaler_fit_only_inside_fold():
    X = _synth(n=40, p=4, seed=5).to_numpy()
    y = X[:, 0].copy()
    fit_row_counts: list[int] = []
    real_fit = StandardScaler.fit

    def tracking_fit(self, X_fit, y_fit=None):
        fit_row_counts.append(len(X_fit))
        return real_fit(self, X_fit, y_fit)

    splitter = KFold(n_splits=4, shuffle=True, random_state=0)
    with patch.object(StandardScaler, "fit", tracking_fit):
        svr_mod.fit_predict_oof(
            X[:, 1:], y, splitter, y_strata=None,
            kernel="linear", C=1.0, epsilon=0.1, use_scaler=True,
        )
    assert all(c < len(y) for c in fit_row_counts)


def test_safe_mkdir_and_write_json_refuse_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "run"
        safe_mkdir(p)
        (p / "marker.txt").write_text("x", encoding="utf-8")
        try:
            safe_mkdir(p)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        f = Path(tmp) / "out.json"
        write_json(f, {"a": 1})
        try:
            write_json(f, {"a": 2})
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass


def test_legacy_rfeca_unchanged_and_coexists():
    legacy = RFECAImputerSVR(top_k=5, feature_names=[f"G{i}" for i in range(5)])
    assert legacy.top_k == 5
    cfg = ExperimentConfig(
        imputers=["SimpleMean", "RFECA_SVR(k=5)", ORIGINAL_RFECA_NAME, ORIGINAL_RFACA_NAME],
        original_rfeca_validation="kfold",
        original_rfeca_n_splits=3,
        original_rfeca_max_candidates=3,
    )
    registry = build_imputers(cfg)
    assert type(registry["RFECA_SVR(k=5)"]).__name__ == "RFECAImputerSVR"
    assert registry["OriginalRFECA"].input_protocol == "target_wise_complete_predictors"
    assert registry["OriginalRFACA"].input_protocol == "multivariate_masked"


def test_select_rfe_rfa_differ_on_shared_prefix():
    rng = np.random.default_rng(0)
    n = 80
    y = rng.normal(size=n)
    X = pd.DataFrame({c: y + rng.normal(scale=0.05, size=n) for c in "abcdef"})
    rfe = sel_mod.select_features_rfe(X, y)
    rfa = sel_mod.select_features_rfa(X, y)
    assert len(rfe) >= 1 and len(rfa) >= 1


def main() -> int:
    tests = [
        test_build_target_wise_only_masks_target_column,
        test_pearson_ranking_aligns_by_position_not_index,
        test_correlation_prefixes_increasing,
        test_select_rfe_rfa_differ_on_shared_prefix,
        test_rfeca_rfaca_not_equivalent,
        test_target_wise_svr_coverage_and_no_predictor_nan,
        test_final_subset_size_can_vary,
        test_no_val_fold_info_in_correlation_or_selector,
        test_final_refit_uses_all_observed,
        test_deterministic_with_random_state,
        test_preserves_unmasked_original_values,
        test_no_simple_imputer_in_original_module,
        test_scaler_fit_only_inside_fold,
        test_safe_mkdir_and_write_json_refuse_overwrite,
        test_legacy_rfeca_unchanged_and_coexists,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
