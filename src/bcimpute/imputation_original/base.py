"""Base class for notebook-faithful Original RFECA / RFACA (leakage-safe)."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .correlation import complete_predictor_mask, pearson_abs_ranking
from .selection import SelectorKind, select_features
from .svr_model import fit_final_svr, make_svr_pipeline
from .types import (
    FallbackEvent,
    FitAudit,
    FoldSelectionDetail,
    GeneAuditRecord,
    SubsetEvaluation,
    ValidationStrategy,
)
from .utils import column_means, to_dataframe
from .validation import make_splitter, rmse
from .target_wise import (
    INPUT_PROTOCOL,
    MASK_SOURCE,
    PREDICTOR_VALUES,
    assert_predictors_finite,
    build_target_wise_matrix,
)


class BaseOriginalCorrelationImputer(BaseEstimator, TransformerMixin, ABC):
    """
    Notebook procedure with nested, leakage-safe validation.

    selection_protocol = leakage_safe (official):
      Per fold: Pearson → prefix → RFE/RFA → SVR on train only; OOF on val.
      After best prefix length: Pearson + RFE/RFA + SVR on all observed.
    """

    method_name: str = "OriginalBase"
    selector_kind: SelectorKind = "RFE"

    def __init__(
        self,
        *,
        validation_strategy: ValidationStrategy = "kfold",
        n_splits: int = 5,
        random_state: int = 42,
        kernel: str = "linear",
        C: float = 1.0,
        epsilon: float = 0.1,
        use_scaler: bool = False,
        min_train_samples: int = 10,
        min_periods: int = 3,
        max_candidates: int | None = 49,
        feature_names: list[str] | None = None,
        strata: np.ndarray | pd.Series | None = None,
        require_complete_on_full_matrix: bool = False,
        candidate_rule: str = "complete_case_per_subset",
        rfa_scoring: str = "r2",
        rfa_threshold: float = 0.001,
        record_fold_details: bool = False,
        selection_protocol: str = "leakage_safe",
        checkpoint_dir: str | Path | None = None,
        run_context: dict[str, Any] | None = None,
        # target_wise_complete_predictors | multivariate_masked (legacy smoke only)
        input_protocol: str = "target_wise_complete_predictors",
        # If set, only these genes are fit/imputed as targets (predictors still use all columns).
        target_genes: list[str] | None = None,
    ):
        self.validation_strategy = validation_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.use_scaler = use_scaler
        self.min_train_samples = min_train_samples
        self.min_periods = min_periods
        self.max_candidates = max_candidates
        self.feature_names = feature_names
        self.strata = strata
        self.require_complete_on_full_matrix = require_complete_on_full_matrix
        self.candidate_rule = (
            "full_matrix"
            if require_complete_on_full_matrix
            else str(candidate_rule)
        )
        self.rfa_scoring = rfa_scoring
        self.rfa_threshold = rfa_threshold
        self.record_fold_details = record_fold_details
        self.selection_protocol = selection_protocol
        self.checkpoint_dir = checkpoint_dir
        self.run_context = run_context
        self.input_protocol = str(input_protocol)
        self.target_genes = target_genes
    @abstractmethod
    def _selector_kind(self) -> SelectorKind:
        raise NotImplementedError

    def set_run_context(self, **kwargs: Any) -> None:
        """Attach dataset/mechanism/rate/rep metadata for fallback logs."""
        ctx = dict(self.run_context or {})
        ctx.update(kwargs)
        self.run_context = ctx

    def set_target_wise_context(
        self,
        X_original: pd.DataFrame | np.ndarray,
        mask: np.ndarray,
        *,
        for_transform: bool = False,
    ) -> None:
        """
        Bind original complete matrix + persisted multivariate mask.

        ``mask`` marks artificial missingness; only the target-gene column is
        applied when building each gene's working matrix.
        """
        X_df = to_dataframe(X_original, self.feature_names)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != X_df.shape:
            raise ValueError(
                f"mask shape {mask_arr.shape} != X_original shape {X_df.shape}"
            )
        if not np.isfinite(X_df.to_numpy(dtype=float)).all():
            raise ValueError(
                "TARGET-WISE requires a fully finite original matrix (predictors)."
            )
        if for_transform:
            self._tw_X_transform_ = X_df
            self._tw_mask_transform_ = mask_arr
        else:
            self._tw_X_original_ = X_df
            self._tw_mask_ = mask_arr

    def _ctx(self) -> dict[str, Any]:
        return dict(self.run_context or {})

    def _is_target_wise(self) -> bool:
        return str(self.input_protocol) == INPUT_PROTOCOL

    def _checkpoint_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return Path(self.checkpoint_dir)

    def _load_gene_checkpoint(self, gene: str) -> GeneAuditRecord | None:
        root = self._checkpoint_path()
        if root is None:
            return None
        path = root / "genes" / f"{gene}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Minimal restore: status + winners; model restored separately if present.
        rec = GeneAuditRecord(
            gene=gene,
            n_observed=int(payload.get("n_observed", 0)),
            n_missing=int(payload.get("n_missing", 0)),
            candidates=list(payload.get("candidates", [])),
            correlation_abs_order=list(payload.get("correlation_abs_order", [])),
            correlation_values=dict(payload.get("correlation_values", {})),
            subsets_evaluated=[],
            winning_prefix_len=int(payload.get("winning_prefix_len", 0)),
            winning_prefix_genes=list(payload.get("winning_prefix_genes", [])),
            winning_predictors=list(payload.get("winning_predictors", [])),
            n_predictors_selected=int(payload.get("n_predictors_selected", 0)),
            validation_strategy=str(payload.get("validation_strategy", "")),
            validation_n_splits=payload.get("validation_n_splits"),
            svr_params=dict(payload.get("svr_params", {})),
            selection_seconds=float(payload.get("selection_seconds", 0.0)),
            final_fit_seconds=float(payload.get("final_fit_seconds", 0.0)),
            selector_kind=str(payload.get("selector_kind", "RFE")),
            n_prefixes_evaluated=int(payload.get("n_prefixes_evaluated", 0)),
            n_rfe_fits=int(payload.get("n_rfe_fits", 0)),
            n_svr_fits=int(payload.get("n_svr_fits", 0)),
            status=str(payload.get("status", "ok")),
            message=str(payload.get("message", "restored_from_checkpoint")),
        )
        # Restore subset RMSE summary if present
        for s in payload.get("subsets_evaluated", []):
            rec.subsets_evaluated.append(
                SubsetEvaluation(
                    prefix_len=int(s["prefix_len"]),
                    prefix_genes_final=list(s.get("prefix_genes_final", [])),
                    selected_features_final=list(s.get("selected_features_final", [])),
                    n_predictors=int(s.get("n_predictors", 0)),
                    rmse=float(s.get("rmse", float("nan"))),
                    n_oof_predictions=int(s.get("n_oof_predictions", 0)),
                )
            )
        return rec

    def _save_gene_checkpoint(self, record: GeneAuditRecord) -> None:
        root = self._checkpoint_path()
        if root is None:
            return
        gene_dir = root / "genes"
        gene_dir.mkdir(parents=True, exist_ok=True)
        path = gene_dir / f"{record.gene}.json"
        if path.exists():
            return  # never overwrite completed gene artifact
        path.write_text(
            json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        # Append progress line
        prog = root / "progress.jsonl"
        with prog.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "gene": record.gene,
                        "status": record.status,
                        "selection_seconds": record.selection_seconds,
                        "n_predictors_selected": record.n_predictors_selected,
                        "winning_prefix_len": record.winning_prefix_len,
                    }
                )
                + "\n"
            )

    def fit(self, X, y=None):
        if self._is_target_wise():
            if not hasattr(self, "_tw_X_original_") or not hasattr(self, "_tw_mask_"):
                raise RuntimeError(
                    "TARGET-WISE fit requires set_target_wise_context(X_original, mask) "
                    "before fit()."
                )
            X_df = self._tw_X_original_
            # Force complete-predictor candidacy (notebook definition A).
            self.candidate_rule = "full_matrix"
        else:
            X_df = to_dataframe(X, self.feature_names)

        self.feature_names_in_ = [str(c) for c in X_df.columns]
        self._fallback_values_ = column_means(X_df)
        self._gene_models_: dict[str, tuple[Any, list[str]]] = {}
        self._correlation_source_ = "training_fold_pairwise_pearson"
        self._n_rfe_fits_ = 0
        self._n_svr_fits_ = 0
        self._n_svr_imputed_cells_ = 0
        self._n_target_masked_cells_ = 0
        self.audit_ = FitAudit(
            method=self.method_name,
            validation_strategy=str(self.validation_strategy),
            random_state=int(self.random_state),
            selection_protocol=str(self.selection_protocol),
            use_scaler=bool(self.use_scaler),
            svr_kernel=str(self.kernel),
            max_candidates=self.max_candidates,
        )

        strata_all = self._resolve_strata(X_df, y)
        ckpt = self._checkpoint_path()
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            models_path = ckpt / "gene_models.joblib"
            if models_path.exists():
                import joblib

                self._gene_models_ = joblib.load(models_path)

        for gene in sorted(self.feature_names_in_):
            if self.target_genes is not None and gene not in set(self.target_genes):
                continue
            restored = self._load_gene_checkpoint(gene)
            if restored is not None and gene in self._gene_models_:
                self.audit_.genes.append(restored)
                self.audit_.n_rfe_fits_total += restored.n_rfe_fits
                self.audit_.n_svr_fits_total += restored.n_svr_fits
                continue
            if restored is not None and restored.status.startswith("skipped"):
                self.audit_.genes.append(restored)
                continue

            if self._is_target_wise():
                X_gene = build_target_wise_matrix(X_df, self._tw_mask_, gene)
                assert_predictors_finite(X_gene, gene)
                self._n_target_masked_cells_ += int(X_gene[gene].isna().sum())
            else:
                X_gene = X_df

            rfe_before = self._n_rfe_fits_
            svr_before = self._n_svr_fits_
            record = self._fit_one_gene(X_gene, gene, strata_all)
            if self._is_target_wise():
                record.message = (
                    f"{record.message}; input_protocol={INPUT_PROTOCOL}; "
                    f"mask_source={MASK_SOURCE}; predictor_values={PREDICTOR_VALUES}"
                )
            record.n_rfe_fits = self._n_rfe_fits_ - rfe_before
            record.n_svr_fits = self._n_svr_fits_ - svr_before
            self.audit_.genes.append(record)
            self.audit_.n_rfe_fits_total += record.n_rfe_fits
            self.audit_.n_svr_fits_total += record.n_svr_fits
            self._save_gene_checkpoint(record)
            if ckpt is not None and self._gene_models_:
                import joblib

                joblib.dump(self._gene_models_, ckpt / "gene_models.joblib")

        return self

    def _resolve_strata(self, X_df: pd.DataFrame, y) -> np.ndarray | None:
        if self.validation_strategy != "stratified_kfold":
            return None
        if self.strata is not None:
            s = pd.Series(self.strata)
            if len(s) != len(X_df):
                raise ValueError("strata length must match X rows.")
            return s.to_numpy()
        if y is not None:
            s = pd.Series(y)
            if len(s) != len(X_df):
                raise ValueError("y length must match X rows for stratified_kfold.")
            return s.astype(str).to_numpy()
        raise ValueError(
            "stratified_kfold requires strata=... on the estimator or y in fit()."
        )

    def _pool_candidates(
        self,
        X_df: pd.DataFrame,
        gene: str,
        row_mask: np.ndarray,
    ) -> list[str]:
        rule = str(self.candidate_rule)
        other = [c for c in self.feature_names_in_ if c != gene]
        y_full = X_df[gene]

        if rule == "full_matrix":
            complete_cols = complete_predictor_mask(X_df[other])
            return [c for c in other if bool(complete_cols.get(c, False))]

        if rule == "target_observed_rows":
            X_sub = X_df.loc[row_mask]
            complete_cols = complete_predictor_mask(X_sub[other])
            return [c for c in other if bool(complete_cols.get(c, False))]

        if rule == "complete_case_per_subset":
            y_rows = y_full.loc[row_mask]
            out: list[str] = []
            for c in other:
                both = y_rows.notna() & X_df.loc[row_mask, c].notna()
                if int(both.sum()) >= max(int(self.min_periods), 2):
                    out.append(c)
            return out

        raise ValueError(f"Unknown candidate_rule={rule!r}")

    def _rank_on_rows(
        self,
        X_df: pd.DataFrame,
        gene: str,
        row_mask: np.ndarray,
        candidates: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        if not candidates:
            return [], {}
        y = X_df.loc[row_mask, gene].to_numpy(dtype=float)
        X_rank = X_df.loc[row_mask, candidates]
        order, scores = pearson_abs_ranking(
            y, X_rank, min_periods=int(self.min_periods)
        )
        if self.max_candidates is not None:
            order = order[: int(self.max_candidates)]
        return order, {k: scores[k] for k in order if k in scores}

    def _select_on_train(
        self, X_prefix: pd.DataFrame, y_train: np.ndarray
    ) -> list[str]:
        self._n_rfe_fits_ += 1
        return select_features(
            X_prefix,
            y_train,
            self._selector_kind(),
            rfa_scoring=str(self.rfa_scoring),
            rfa_threshold=float(self.rfa_threshold),
        )

    def _record_exception(self, where: str, exc: BaseException) -> None:
        self.audit_.exceptions.append(
            {
                "where": where,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def _oof_for_prefix_len(
        self,
        X_df: pd.DataFrame,
        gene: str,
        obs_mask: np.ndarray,
        strata_all: np.ndarray | None,
        prefix_len: int,
    ) -> tuple[float, int, list[FoldSelectionDetail], float]:
        idx_obs = np.flatnonzero(obs_mask)
        n_obs = int(idx_obs.size)
        if n_obs < int(self.min_train_samples):
            return float("nan"), 0, [], float("nan")

        y_obs = X_df[gene].to_numpy(dtype=float)[idx_obs]
        strata_obs = strata_all[idx_obs] if strata_all is not None else None

        n_splits_eff = int(self.n_splits)
        if self.validation_strategy in {"kfold", "stratified_kfold"}:
            n_splits_eff = min(n_splits_eff, n_obs)
            if n_splits_eff < 2:
                return float("nan"), 0, [], float("nan")

        splitter = make_splitter(
            self.validation_strategy,  # type: ignore[arg-type]
            n_splits=n_splits_eff,
            random_state=int(self.random_state),
            y_for_strata=strata_obs,
        )

        oof_pred = np.full(n_obs, np.nan, dtype=float)
        fold_details: list[FoldSelectionDetail] = []
        n_selected_folds: list[int] = []
        split_iter = (
            splitter.split(np.zeros(n_obs), strata_obs)
            if strata_obs is not None
            else splitter.split(np.zeros(n_obs))
        )

        for fold_i, (tr_local, va_local) in enumerate(split_iter):
            tr_global = idx_obs[tr_local]
            va_global = idx_obs[va_local]
            tr_mask = np.zeros(len(X_df), dtype=bool)
            tr_mask[tr_global] = True

            cand = self._pool_candidates(X_df, gene, tr_mask)
            order, _scores = self._rank_on_rows(X_df, gene, tr_mask, cand)
            if len(order) < prefix_len:
                continue
            prefix = list(order[:prefix_len])

            X_tr = X_df.iloc[tr_global][prefix]
            y_tr = y_obs[tr_local]
            tr_ok = X_tr.notna().all(axis=1).to_numpy()
            if int(tr_ok.sum()) < int(self.min_train_samples):
                continue
            X_tr_cc = X_tr.loc[tr_ok]
            y_tr_cc = y_tr[tr_ok]

            try:
                selected = self._select_on_train(X_tr_cc, y_tr_cc)
            except Exception as exc:  # noqa: BLE001
                self._record_exception(f"{gene}|fold={fold_i}|prefix={prefix_len}|RFE", exc)
                continue
            if not selected:
                continue

            X_va = X_df.iloc[va_global][selected]
            va_ok = X_va.notna().all(axis=1).to_numpy()
            if int(va_ok.sum()) == 0:
                continue

            pipe = make_svr_pipeline(
                kernel=self.kernel,
                C=float(self.C),
                epsilon=float(self.epsilon),
                use_scaler=bool(self.use_scaler),
            )
            try:
                pipe.fit(
                    X_tr_cc[selected].to_numpy(dtype=float),
                    y_tr_cc,
                )
                self._n_svr_fits_ += 1
                pred = pipe.predict(X_va.loc[va_ok].to_numpy(dtype=float))
            except Exception as exc:  # noqa: BLE001
                self._record_exception(f"{gene}|fold={fold_i}|prefix={prefix_len}|SVR", exc)
                continue
            oof_pred[va_local[va_ok]] = np.asarray(pred, dtype=float)
            n_selected_folds.append(len(selected))

            if self.record_fold_details:
                fold_details.append(
                    FoldSelectionDetail(
                        fold=int(fold_i),
                        n_train=int(tr_ok.sum()),
                        n_val=int(va_ok.sum()),
                        correlation_order=list(order),
                        prefix_genes=list(prefix),
                        selected_features=list(selected),
                    )
                )

        good = np.isfinite(oof_pred)
        n_pred = int(good.sum())
        mean_n_sel = (
            float(np.mean(n_selected_folds)) if n_selected_folds else float("nan")
        )
        if n_pred < max(2, int(self.min_train_samples) // 2):
            return float("nan"), n_pred, fold_details, mean_n_sel
        score = rmse(y_obs[good], oof_pred[good])
        return score, n_pred, fold_details, mean_n_sel

    def _fit_one_gene(
        self,
        X_df: pd.DataFrame,
        gene: str,
        strata_all: np.ndarray | None,
    ) -> GeneAuditRecord:
        obs_mask = X_df[gene].notna().to_numpy()
        miss_count = int((~obs_mask).sum())
        n_obs = int(obs_mask.sum())
        kind = self._selector_kind()
        svr_params = {
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "use_scaler": self.use_scaler,
            "selector_kind": kind,
            "selection_protocol": self.selection_protocol,
            "rfa_scoring": self.rfa_scoring,
            "rfa_threshold": self.rfa_threshold,
        }
        base_rec = GeneAuditRecord(
            gene=gene,
            n_observed=n_obs,
            n_missing=miss_count,
            candidates=[],
            correlation_abs_order=[],
            correlation_values={},
            subsets_evaluated=[],
            winning_prefix_len=0,
            winning_prefix_genes=[],
            winning_predictors=[],
            n_predictors_selected=0,
            validation_strategy=str(self.validation_strategy),
            validation_n_splits=(
                None
                if self.validation_strategy == "loocv"
                else int(self.n_splits)
            ),
            svr_params=svr_params,
            selection_seconds=0.0,
            final_fit_seconds=0.0,
            selector_kind=kind,
        )

        if n_obs < int(self.min_train_samples):
            base_rec.status = "skipped_few_observed"
            base_rec.message = (
                f"n_observed={n_obs} < min_train_samples={self.min_train_samples}"
            )
            return base_rec

        rule = str(self.candidate_rule)
        try:
            candidates = self._pool_candidates(X_df, gene, obs_mask)
        except ValueError as exc:
            base_rec.status = "skipped_bad_candidate_rule"
            base_rec.message = str(exc)
            self._record_exception(f"{gene}|candidates", exc)
            return base_rec

        if not candidates:
            base_rec.status = "skipped_no_complete_candidates"
            base_rec.message = f"No candidates under rule={rule}."
            return base_rec

        order_full, scores_full = self._rank_on_rows(
            X_df, gene, obs_mask, candidates
        )
        base_rec.candidates = list(candidates)
        base_rec.correlation_abs_order = list(order_full)
        base_rec.correlation_values = dict(scores_full)
        base_rec.message = f"candidate_rule={rule}"

        if not order_full:
            base_rec.status = "skipped_no_valid_correlation"
            base_rec.message = "All pairwise correlations were NaN."
            return base_rec

        max_k = len(order_full)
        evaluations: list[SubsetEvaluation] = []
        t_sel0 = time.perf_counter()

        best_key: tuple | None = None
        best_prefix_len: int | None = None

        for prefix_len in range(1, max_k + 1):
            score, n_oof, fold_details, mean_n_sel = self._oof_for_prefix_len(
                X_df, gene, obs_mask, strata_all, prefix_len
            )
            prefix_final = list(order_full[:prefix_len])
            n_pred_proxy = (
                int(round(mean_n_sel)) if np.isfinite(mean_n_sel) else prefix_len
            )
            evaluations.append(
                SubsetEvaluation(
                    prefix_len=prefix_len,
                    prefix_genes_final=prefix_final,
                    selected_features_final=[],
                    n_predictors=n_pred_proxy,
                    rmse=score,
                    n_oof_predictions=n_oof,
                    fold_details=fold_details if self.record_fold_details else [],
                )
            )
            if not np.isfinite(score):
                continue
            # Deterministic tie-break:
            # 1) lower OOF RMSE
            # 2) fewer predictors after RFE (OOF mean n_selected)
            # 3) smaller prefix length
            # 4) lexicographic prefix gene list
            key = (
                float(score),
                float(mean_n_sel) if np.isfinite(mean_n_sel) else float(prefix_len),
                int(prefix_len),
                tuple(prefix_final),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_prefix_len = prefix_len

        selection_only = time.perf_counter() - t_sel0
        base_rec.subsets_evaluated = evaluations
        base_rec.n_prefixes_evaluated = len(evaluations)

        if best_prefix_len is None:
            base_rec.status = "skipped_no_winning_subset"
            base_rec.message = "No prefix produced a finite OOF RMSE."
            base_rec.selection_seconds = selection_only
            return base_rec

        prefix_win = list(order_full[:best_prefix_len])
        cc_final = obs_mask.copy()
        for p in prefix_win:
            cc_final = cc_final & X_df[p].notna().to_numpy()
        n_cc_final = int(cc_final.sum())
        if n_cc_final < int(self.min_train_samples):
            base_rec.status = "skipped_final_fit_fail"
            base_rec.message = (
                f"Insufficient complete-case rows for final refit ({n_cc_final})."
            )
            base_rec.selection_seconds = selection_only
            return base_rec

        t_fit0 = time.perf_counter()
        try:
            selected_win = self._select_on_train(
                X_df.loc[cc_final, prefix_win],
                X_df.loc[cc_final, gene].to_numpy(dtype=float),
            )
            if not selected_win:
                raise RuntimeError("RFE/RFA returned empty feature set.")
            pipe = fit_final_svr(
                X_df.loc[cc_final, selected_win].to_numpy(dtype=float),
                X_df.loc[cc_final, gene].to_numpy(dtype=float),
                kernel=self.kernel,
                C=float(self.C),
                epsilon=float(self.epsilon),
                use_scaler=bool(self.use_scaler),
            )
            self._n_svr_fits_ += 1
        except Exception as exc:  # noqa: BLE001
            self._record_exception(f"{gene}|final_refit", exc)
            base_rec.status = "skipped_final_fit_fail"
            base_rec.message = f"{type(exc).__name__}: {exc}"
            base_rec.selection_seconds = selection_only
            return base_rec

        final_fit_s = time.perf_counter() - t_fit0
        self._gene_models_[gene] = (pipe, list(selected_win))
        base_rec.winning_prefix_len = int(best_prefix_len)
        base_rec.winning_prefix_genes = list(prefix_win)
        base_rec.winning_predictors = list(selected_win)
        base_rec.n_predictors_selected = len(selected_win)
        base_rec.selection_seconds = selection_only
        base_rec.final_fit_seconds = final_fit_s
        for ev in evaluations:
            if ev.prefix_len == best_prefix_len:
                ev.selected_features_final = list(selected_win)
                ev.n_predictors = len(selected_win)
                break
        base_rec.status = "ok"
        base_rec.message = (
            f"candidate_rule={rule}; selector={kind}; "
            f"prefix_len={best_prefix_len}; n_complete_case_final={n_cc_final}; "
            f"n_selected={len(selected_win)}; "
            f"selection_protocol={self.selection_protocol}; "
            f"use_scaler={bool(self.use_scaler)}"
        )
        return base_rec

    def _append_fallback(
        self,
        *,
        gene: str,
        reason: str,
        n_observed: int,
        n_candidates: int,
        prefix_len: int | None,
        n_rows_affected: int,
        exception_type: str = "",
        exception_message: str = "",
    ) -> None:
        ctx = self._ctx()
        event = FallbackEvent(
            gene=gene,
            reason=reason,
            n_observed=n_observed,
            n_candidates=n_candidates,
            prefix_len=prefix_len,
            n_rows_affected=n_rows_affected,
            dataset=str(ctx.get("dataset", "")),
            mechanism=str(ctx.get("mechanism", "")),
            missing_rate=ctx.get("missing_rate"),
            replicate=ctx.get("replicate"),
            exception_type=exception_type,
            exception_message=exception_message,
        )
        if hasattr(self, "audit_"):
            self.audit_.fallback_events.append(event)

    def transform(self, X):
        if not hasattr(self, "_gene_models_"):
            raise RuntimeError(f"{self.__class__.__name__} must be fit before transform.")

        if self._is_target_wise():
            return self._transform_target_wise(X)
        return self._transform_multivariate(X)

    def _transform_target_wise(self, X):
        """Impute only target-mask cells; predictors always from original matrix."""
        if not hasattr(self, "_tw_X_transform_") or not hasattr(
            self, "_tw_mask_transform_"
        ):
            raise RuntimeError(
                "TARGET-WISE transform requires "
                "set_target_wise_context(..., for_transform=True) before transform()."
            )
        X_orig = self._tw_X_transform_.reindex(columns=self.feature_names_in_)
        mask = self._tw_mask_transform_
        if mask.shape != X_orig.shape:
            raise ValueError("transform mask shape mismatch vs X_original.")
        if not np.isfinite(X_orig.to_numpy(dtype=float)).all():
            raise AssertionError(
                "TARGET-WISE transform: original matrix has non-finite predictors."
            )

        X_out = X_orig.copy()
        t0 = time.perf_counter()
        audit_by_gene = {
            g.gene: g for g in (self.audit_.genes if hasattr(self, "audit_") else [])
        }
        n_svr_cells = 0
        n_mask_cells = 0

        for j, gene in enumerate(self.feature_names_in_):
            if self.target_genes is not None and gene not in set(self.target_genes):
                continue
            miss = mask[:, j]
            if not np.any(miss):
                continue
            n_mask_cells += int(miss.sum())
            idx = np.flatnonzero(miss)
            packed = self._gene_models_.get(gene)
            rec = audit_by_gene.get(gene)
            n_obs = int(rec.n_observed) if rec else 0
            n_cand = len(rec.candidates) if rec else 0
            prefix_len = int(rec.winning_prefix_len) if rec else None

            if packed is None:
                fill = float(self._fallback_values_.get(gene, 0.0))
                X_out.iloc[idx, j] = fill
                self._append_fallback(
                    gene=gene,
                    reason=(
                        rec.status
                        if rec is not None and str(rec.status).startswith("skipped")
                        else "no_fitted_svr_model"
                    ),
                    n_observed=n_obs,
                    n_candidates=n_cand,
                    prefix_len=prefix_len,
                    n_rows_affected=int(idx.size),
                    exception_message=(rec.message if rec else ""),
                )
                continue

            pipe, predictors = packed
            pred_block = X_orig.iloc[idx][predictors]
            if pred_block.isna().any().any() or not np.isfinite(
                pred_block.to_numpy(dtype=float)
            ).all():
                raise AssertionError(
                    f"TARGET-WISE: NaN/Inf in predictors at impute time for {gene}. "
                    "Fallback by missing predictors is forbidden."
                )
            y_hat = pipe.predict(pred_block.to_numpy(dtype=float))
            X_out.iloc[idx, j] = y_hat
            n_svr_cells += int(idx.size)

        # No residual NaNs expected on artificially masked target cells.
        tgt_mask = np.zeros_like(mask, dtype=bool)
        for j, gene in enumerate(self.feature_names_in_):
            if self.target_genes is not None and gene not in set(self.target_genes):
                continue
            tgt_mask[:, j] = mask[:, j]
        if np.any(tgt_mask) and not np.isfinite(X_out.to_numpy(dtype=float)[tgt_mask]).all():
            raise AssertionError("TARGET-WISE left non-finite values on masked cells.")

        self._n_svr_imputed_cells_ = n_svr_cells
        self._n_target_masked_cells_transform_ = n_mask_cells
        impute_s = time.perf_counter() - t0
        if hasattr(self, "audit_") and self.audit_.genes:
            self.audit_.genes[-1].impute_seconds = impute_s
        self._last_transform_seconds_ = impute_s
        # X is unused for predictors; accept for API compatibility.
        _ = X
        return X_out

    def _transform_multivariate(self, X):
        """Legacy multivariate path (RFACA smoke only; not used for principal RFECA)."""
        X_df = to_dataframe(X, self.feature_names)
        X_df = X_df.reindex(columns=self.feature_names_in_)
        X_pred_source = X_df.copy()
        X_out = X_df.copy()
        predictions: dict[str, pd.Series] = {}
        t0 = time.perf_counter()
        audit_by_gene = {
            g.gene: g for g in (self.audit_.genes if hasattr(self, "audit_") else [])
        }

        for gene in sorted(self.feature_names_in_):
            miss = X_pred_source[gene].isna()
            if int(miss.sum()) == 0:
                continue
            packed = self._gene_models_.get(gene)
            rec = audit_by_gene.get(gene)
            n_obs = int(rec.n_observed) if rec else int(X_pred_source[gene].notna().sum())
            n_cand = len(rec.candidates) if rec else 0
            prefix_len = int(rec.winning_prefix_len) if rec else None

            if packed is None:
                fill = float(self._fallback_values_.get(gene, 0.0))
                n_aff = int(miss.sum())
                reason = (
                    rec.status if rec is not None and rec.status.startswith("skipped")
                    else "no_fitted_svr_model"
                )
                self._append_fallback(
                    gene=gene,
                    reason=reason,
                    n_observed=n_obs,
                    n_candidates=n_cand,
                    prefix_len=prefix_len,
                    n_rows_affected=n_aff,
                    exception_message=(rec.message if rec else ""),
                )
                predictions[gene] = pd.Series(fill, index=X_out.index[miss])
                continue

            pipe, predictors = packed
            pred_block = X_pred_source.loc[miss, predictors]
            row_ok = ~pred_block.isna().any(axis=1)
            y_hat_full = pd.Series(index=X_out.index[miss], dtype=float)

            if int(row_ok.sum()) > 0:
                idx_ok = pred_block.index[row_ok]
                y_hat = pipe.predict(pred_block.loc[idx_ok].to_numpy(dtype=float))
                y_hat_full.loc[idx_ok] = y_hat

            if int((~row_ok).sum()) > 0:
                fill = float(self._fallback_values_.get(gene, 0.0))
                bad_idx = pred_block.index[~row_ok]
                y_hat_full.loc[bad_idx] = fill
                self._append_fallback(
                    gene=gene,
                    reason="incomplete_predictors_on_impute_rows",
                    n_observed=n_obs,
                    n_candidates=n_cand,
                    prefix_len=prefix_len,
                    n_rows_affected=int(len(bad_idx)),
                )

            predictions[gene] = y_hat_full

        for gene, series in predictions.items():
            X_out.loc[series.index, gene] = series.to_numpy()

        for gene in self.feature_names_in_:
            if X_out[gene].isna().any():
                n_left = int(X_out[gene].isna().sum())
                fill = float(self._fallback_values_.get(gene, 0.0))
                X_out[gene] = X_out[gene].fillna(fill)
                self._append_fallback(
                    gene=gene,
                    reason="residual_nan_after_transform",
                    n_observed=int(X_pred_source[gene].notna().sum()),
                    n_candidates=0,
                    prefix_len=None,
                    n_rows_affected=n_left,
                )

        impute_s = time.perf_counter() - t0
        if hasattr(self, "audit_") and self.audit_.genes:
            self.audit_.genes[-1].impute_seconds = impute_s
        self._last_transform_seconds_ = impute_s
        return X_out

    def fallback_rate(self) -> float:
        """Fraction of genes-with-missing that used model-level mean fallback.

        Row-level fills due to incomplete predictors under MCAR are tracked
        separately via ``incomplete_predictor_fallback_rate`` and do not count
        here (SVR was fitted; only some impute rows lacked predictors).
        """
        if not hasattr(self, "audit_"):
            return 0.0
        n_imp = sum(1 for g in self.audit_.genes if g.n_missing > 0)
        if n_imp == 0:
            return 0.0
        model_reasons = {
            "no_fitted_svr_model",
            "residual_nan_after_transform",
        }
        genes_fb = {
            e.gene
            for e in self.audit_.fallback_events
            if e.reason in model_reasons or e.reason.startswith("skipped")
        }
        genes_miss = {g.gene for g in self.audit_.genes if g.n_missing > 0}
        return float(len(genes_fb & genes_miss) / n_imp)

    def incomplete_predictor_fallback_rate(self) -> float:
        """Fraction of genes that needed mean on some rows due to NaN predictors."""
        if not hasattr(self, "audit_"):
            return 0.0
        n_imp = sum(1 for g in self.audit_.genes if g.n_missing > 0)
        if n_imp == 0:
            return 0.0
        genes_fb = {
            e.gene
            for e in self.audit_.fallback_events
            if e.reason == "incomplete_predictors_on_impute_rows"
        }
        genes_miss = {g.gene for g in self.audit_.genes if g.n_missing > 0}
        return float(len(genes_fb & genes_miss) / n_imp)

    def get_audit_dict(self) -> dict[str, Any]:
        if not hasattr(self, "audit_"):
            return {}
        d = self.audit_.to_dict()
        d["fallback_rate"] = self.fallback_rate()
        d["incomplete_predictor_fallback_rate"] = (
            self.incomplete_predictor_fallback_rate()
        )
        d["input_protocol"] = str(self.input_protocol)
        d["mask_source"] = MASK_SOURCE if self._is_target_wise() else "multivariate"
        d["predictor_values"] = (
            PREDICTOR_VALUES if self._is_target_wise() else "masked_matrix"
        )
        n_mask = int(getattr(self, "_n_target_masked_cells_transform_", 0) or 0)
        n_svr = int(getattr(self, "_n_svr_imputed_cells_", 0) or 0)
        d["n_target_masked_cells"] = n_mask
        d["n_svr_imputed_cells"] = n_svr
        d["svr_coverage"] = (float(n_svr) / n_mask) if n_mask else 1.0
        return d
