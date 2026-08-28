"""Classifier pipelines matching the study protocol."""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_pipelines_with_imputer(imputer, random_state: int = 42) -> dict:
    """
    Build classification pipelines.

    StandardScaler is a pipeline step fitted inside each CV fold on training data only.
    """
    pipeline_svc = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)),
        ]
    )

    pipeline_lr = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )

    pipeline_rf = Pipeline(
        [
            ("imputer", imputer),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )

    pipeline_gb = Pipeline(
        [
            ("imputer", imputer),
            ("clf", GradientBoostingClassifier(random_state=random_state)),
        ]
    )

    svc_soft = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),
        ]
    )
    rf_pipe = RandomForestClassifier(n_estimators=200, random_state=random_state)
    gb_pipe = GradientBoostingClassifier(random_state=random_state)

    ensemble_soft = VotingClassifier(
        estimators=[
            ("svc", svc_soft),
            ("lr", lr_pipe),
            ("rf", rf_pipe),
            ("gb", gb_pipe),
        ],
        voting="soft",
    )

    pipeline_ensemble_soft = Pipeline(
        [
            ("imputer", imputer),
            ("clf", ensemble_soft),
        ]
    )

    return {
        "SVC": pipeline_svc,
        "LogReg": pipeline_lr,
        "RF": pipeline_rf,
        "GB": pipeline_gb,
        "EnsembleSoft": pipeline_ensemble_soft,
    }
