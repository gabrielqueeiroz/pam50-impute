"""Quick checks: collision-free seeds + inductive RFECA."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcimpute.imputers import RFECAImputerSVR
from bcimpute.missingness import missingness_seed


def test_seed_v2_no_collision():
    rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    reps = range(10)
    seen = {}
    for mech in ("mcar", "mar"):
        for rate in rates:
            for rep in reps:
                s = missingness_seed(42, rate, rep, mechanism=mech, scheme="v2")
                key = (mech, s)
                assert key not in seen, f"collision {seen[key]} vs {(mech, rate, rep)} seed={s}"
                seen[key] = (mech, rate, rep)
    # legacy still collides as documented
    a = missingness_seed(42, 0.05, 5, mechanism="mcar", scheme="legacy")
    b = missingness_seed(42, 0.10, 0, mechanism="mcar", scheme="legacy")
    assert a == b == 1052
    print("PASS seeds")


def test_rfeca_inductive_no_refit_on_transform():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(8)]
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=genes)
    # punch holes in a held-out block
    X_train = X.iloc[:30].copy()
    X_test = X.iloc[30:].copy()
    X_test.iloc[0, 0] = np.nan
    X_test.iloc[1, 2] = np.nan

    imp = RFECAImputerSVR(top_k=3, min_train_samples=5, feature_names=genes)
    imp.fit(X_train)
    assert len(imp._gene_models_) > 0

    # Monkeypatch: any .fit on stored SVRs during transform must fail the test
    fits_during_transform = []
    for gene, (inner, scaler, model, preds) in imp._gene_models_.items():
        orig = model.fit

        def _guard(*a, _orig=orig, _g=gene, **k):
            fits_during_transform.append(_g)
            return _orig(*a, **k)

        model.fit = _guard  # type: ignore[method-assign]

    out = imp.transform(X_test)
    assert not fits_during_transform, f"SVR.fit called in transform for {fits_during_transform}"
    assert out.isna().sum().sum() == 0
    print("PASS rfeca inductive")


if __name__ == "__main__":
    test_seed_v2_no_collision()
    test_rfeca_inductive_no_refit_on_transform()
    print("ALL OK")
