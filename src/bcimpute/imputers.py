"""Imputation estimators. RFECA computes correlations and SVR models only from the training fold."""



from __future__ import annotations



from typing import Any



import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR





def make_missforest(

    *,

    n_estimators: int = 20,

    max_iter: int = 5,

    random_state: int = 42,

    n_jobs: int = 1,

) -> IterativeImputer:

    """

    MissForest-like imputer (sklearn IterativeImputer + ExtraTreesRegressor).



    Not identical to the original R missForest package; documented as such in reports.

    ExtraTrees is used for speed vs RandomForest while preserving the RF-imputation idea.

    """

    return IterativeImputer(

        estimator=ExtraTreesRegressor(

            n_estimators=n_estimators,

            random_state=random_state,

            n_jobs=n_jobs,

        ),

        max_iter=max_iter,

        random_state=random_state,

        sample_posterior=False,

        skip_complete=True,

    )





class RFECAImputerSVR(BaseEstimator, TransformerMixin):

    """

    Correlation-guided SVR imputation (inductive).



    Leakage-safe design:

      - Gene-gene correlations are computed in fit() from the training matrix only.

      - Per-gene SimpleImputer / StandardScaler / SVR are fitted in fit() on training

        rows where the target gene is observed.

      - transform() only applies those fitted objects (no SVR.fit / scaler.fit).

      - No precomputed full-cohort correlation CSV is used or accepted.

    """



    def __init__(

        self,

        top_k: int = 5,

        use_abs_corr: bool = True,

        kernel: str = "linear",

        C: float = 1.0,

        epsilon: float = 0.1,

        min_train_samples: int = 10,

        fallback_strategy: str = "mean",

        feature_names: list[str] | None = None,

        # Explicitly rejected for leakage prevention.

        corr_csv_path: str | None = None,

        allow_precomputed_correlation: bool = False,

    ):

        self.top_k = top_k

        self.use_abs_corr = use_abs_corr

        self.kernel = kernel

        self.C = C

        self.epsilon = epsilon

        self.min_train_samples = min_train_samples

        self.fallback_strategy = fallback_strategy

        self.feature_names = feature_names

        self.corr_csv_path = corr_csv_path

        self.allow_precomputed_correlation = allow_precomputed_correlation



    def _to_dataframe(self, X) -> pd.DataFrame:

        if isinstance(X, pd.DataFrame):

            X_df = X.copy()

            X_df.columns = X_df.columns.astype(str)

            return X_df

        if self.feature_names is not None:

            return pd.DataFrame(X, columns=[str(c) for c in self.feature_names]).copy()

        raise ValueError(

            "X was provided as ndarray without feature_names. "

            "Pass a DataFrame or set feature_names."

        )



    def fit(self, X, y=None):

        if self.corr_csv_path is not None and not self.allow_precomputed_correlation:

            raise ValueError(

                "RFECAImputerSVR refuses precomputed correlation matrices "

                "(corr_csv_path is set while allow_precomputed_correlation=False). "

                "Correlations must be computed from the training fold inside fit()."

            )



        X_df = self._to_dataframe(X)

        self.feature_names_in_ = list(X_df.columns)



        # --- Fold-local correlation (training data only) ---

        corr = X_df.corr(method="pearson", min_periods=max(3, self.min_train_samples // 2))

        corr.index = corr.index.astype(str)

        corr.columns = corr.columns.astype(str)

        self.correlation_matrix_ = corr.copy()

        self._correlation_source_ = "training_fold_X.corr()"

        self._fit_object_id_ = id(self)



        if self.fallback_strategy == "zero":

            self._fallback_values_ = pd.Series(0.0, index=self.feature_names_in_)

        else:

            self._fallback_values_ = X_df.mean(axis=0)



        self._top_features_: dict[str, list[str]] = {}

        for gene in self.feature_names_in_:

            if gene not in corr.index:

                self._top_features_[gene] = []

                continue

            s = corr.loc[gene].drop(labels=[gene], errors="ignore").dropna()

            if self.use_abs_corr:

                s = s.reindex(s.abs().sort_values(ascending=False).index)

            else:

                s = s.sort_values(ascending=False)

            self._top_features_[gene] = list(s.index[: self.top_k])



        # --- Per-gene inductive SVR (training rows only) ---

        self._gene_models_: dict[str, tuple[Any, Any, Any, list[str]]] = {}

        fit_warnings: list[str] = []

        for gene, predictors in self._top_features_.items():

            if not predictors:

                continue

            obs = X_df[gene].notna()

            if int(obs.sum()) < self.min_train_samples:

                continue

            valid_predictors = [p for p in predictors if p in X_df.columns]

            if not valid_predictors:

                continue



            X_obs = X_df.loc[obs, valid_predictors].to_numpy(dtype=float)

            y_obs = X_df.loc[obs, gene].to_numpy(dtype=float)



            inner_imp = SimpleImputer(strategy="mean")

            scaler = StandardScaler()

            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

            try:

                X_obs_i = inner_imp.fit_transform(X_obs)

                X_obs_s = scaler.fit_transform(X_obs_i)

                model.fit(X_obs_s, y_obs)

                self._gene_models_[gene] = (inner_imp, scaler, model, valid_predictors)

            except Exception as exc:  # noqa: BLE001 - fallback is intentional

                fit_warnings.append(f"svr_fit_fail:{gene}:{type(exc).__name__}")



        self._last_fit_warnings_ = fit_warnings

        return self



    def transform(self, X):

        if not hasattr(self, "_top_features_") or not hasattr(self, "_gene_models_"):

            raise RuntimeError("RFECAImputerSVR must be fit before transform.")



        X_df = self._to_dataframe(X)

        X_df = X_df.reindex(columns=self.feature_names_in_)



        warnings_local: list[str] = []



        for gene in self.feature_names_in_:

            miss = X_df[gene].isna()

            if int(miss.sum()) == 0:

                continue



            packed = self._gene_models_.get(gene)

            if packed is None:

                X_df[gene] = self._fallback_fill(X_df[gene], gene)

                continue



            inner_imp, scaler, model, valid_predictors = packed

            X_pred = X_df.loc[miss, valid_predictors].to_numpy(dtype=float)

            try:

                X_pred = inner_imp.transform(X_pred)

                X_pred = scaler.transform(X_pred)

                y_hat = model.predict(X_pred)

                if not np.isfinite(y_hat).all():

                    warnings_local.append(f"non_finite_prediction:{gene}")

                    X_df[gene] = self._fallback_fill(X_df[gene], gene)

                else:

                    X_df.loc[miss, gene] = y_hat

            except Exception as exc:  # noqa: BLE001 - fallback is intentional

                warnings_local.append(f"svr_predict_fail:{gene}:{type(exc).__name__}")

                X_df[gene] = self._fallback_fill(X_df[gene], gene)



        self._last_transform_warnings_ = warnings_local

        return X_df



    def _fallback_fill(self, s: pd.Series, gene: str) -> pd.Series:

        fill_value = float(self._fallback_values_.get(gene, 0.0))

        return s.fillna(fill_value)





def build_imputers(config) -> dict[str, Any]:

    """Build the imputer dictionary from ExperimentConfig."""

    from .config import PAM50_GENES



    feature_names = list(PAM50_GENES)



    registry: dict[str, Any] = {}

    for name in config.imputers:

        if name == "SimpleMean":

            registry[name] = SimpleImputer(strategy="mean")

        elif name == "KNN(k=5,dist)":

            registry[name] = KNNImputer(n_neighbors=5, weights="distance")

        elif name == "MissForest":

            registry[name] = make_missforest(

                n_estimators=int(getattr(config, "missforest_n_estimators", 20)),

                max_iter=int(getattr(config, "missforest_max_iter", 5)),

                random_state=int(getattr(config, "random_state", 42)),

                n_jobs=int(getattr(config, "missforest_n_jobs", 4)),

            )

        elif name.startswith("RFECA_SVR"):

            top_k = config.rfeca_top_k

            if "k=" in name:

                top_k = int(name.split("k=")[1].rstrip(")"))

            registry[name] = RFECAImputerSVR(

                top_k=top_k,

                use_abs_corr=config.rfeca_use_abs_corr,

                kernel=config.rfeca_kernel,

                C=config.rfeca_C,

                epsilon=config.rfeca_epsilon,

                min_train_samples=config.rfeca_min_train_samples,

                feature_names=feature_names,

                corr_csv_path=None,

                allow_precomputed_correlation=config.allow_precomputed_correlation,

            )

        elif name in {"OriginalRFECA", "OriginalRFACA"}:
            # Faithful Original RFECA/RFACA — parallel module; legacy RFECA unchanged.
            from .imputation_original import (
                OriginalRFACAImputer,
                OriginalRFECAImputer,
            )

            common = dict(
                validation_strategy=getattr(
                    config, "original_rfeca_validation", "kfold"
                ),
                n_splits=int(getattr(config, "original_rfeca_n_splits", 5)),
                random_state=int(getattr(config, "random_state", 42)),
                kernel=str(
                    getattr(config, "original_rfeca_kernel", config.rfeca_kernel)
                ),
                C=float(getattr(config, "original_rfeca_C", config.rfeca_C)),
                epsilon=float(
                    getattr(config, "original_rfeca_epsilon", config.rfeca_epsilon)
                ),
                use_scaler=bool(getattr(config, "original_rfeca_use_scaler", False)),
                min_train_samples=int(
                    getattr(
                        config,
                        "original_rfeca_min_train_samples",
                        config.rfeca_min_train_samples,
                    )
                ),
                max_candidates=getattr(config, "original_rfeca_max_candidates", 49),
                feature_names=feature_names,
                input_protocol=(
                    "target_wise_complete_predictors"
                    if name == "OriginalRFECA"
                    else "multivariate_masked"
                ),
            )
            if name == "OriginalRFECA":
                registry[name] = OriginalRFECAImputer(**common)
            else:
                registry[name] = OriginalRFACAImputer(**common)

        else:

            raise ValueError(f"Unknown imputer: {name}")

    return registry





def clone_imputer(imputer):

    return clone(imputer)


