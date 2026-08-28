# Methodology (as executed)

Canonical freeze: **`v0.3.1-original-rfeca-targetwise`**.

This page summarizes **only** what the current code and freeze config execute.
The detailed executed-protocol write-up is
`artifacts/final_analysis/methods_freeze.md`.
Do not treat older `RFECA_SVR(k=5|10|20)` campaigns as the principal method.

This is a **research benchmark**, not a clinical diagnostic system.

## Datasets

| Cohort | Role | n samples | n genes | Labels |
|---|---|---|---|---|
| METABRIC PAM50 (processed) | Principal freeze + fair baselines | 1608 | 50 | LumA, LumB, Her2, Basal |
| Discovery laboratory PAM50 | Historical six-imputer campaign | 117 | 50 | same 4-class |

See `data/README.md` for acquisition and redistribution limits.

## Controlled missingness

Implemented in `src/bcimpute/missingness.py`.

- **MCAR:** exact cell count `round(rate × n_eligible)`, sampled without replacement.
- **MAR:** leave-one-out row mean → column z-score → `|z| + Gumbel`; top-n cells (exact count).
- Eligible cells: `originally_observed_only` when an observation mask exists.
- Freeze grid: MCAR and MAR × {5%, 10%, 20%, 30%} × 5 replicates (40 slots).
- Seeds: scheme **v2**, `base_seed=42`.

Masks are shared across imputers within each `(mechanism, rate, replicate)` for fair comparison.

## Gene estimation — OriginalRFECA TARGET-WISE (principal)

Implemented in `src/bcimpute/imputation_original/` (`rfeca.py`, `selection.py`, `target_wise.py`, `base.py`).

Per target gene:

1. Artificial NaNs are applied **only** to that gene's masked cells.
2. Predictors are the **original complete** values of other genes (no chaining; imputed values are never reused as predictors).
3. Candidates are ranked by absolute Pearson correlation on **observed** target rows.
4. The candidate pool is capped at `max_candidates=49` (all other PAM50 genes).
5. For each prefix length 1…pool size, inner 5-fold CV re-ranks on the train fold, applies sklearn **RFE + linear SVR**, and scores OOF RMSE.
6. The winning prefix is refit on all observed rows; only masked cells are filled.
7. RFE's default `n_features_to_select` is approximately **half the prefix length** (sklearn). The number of predictors is therefore **dynamic per gene**, not a global fixed k.

Config (freeze): `use_scaler=False`, `selection_protocol=leakage_safe`, `inner_cv=5`, `evaluation_protocol=repeated_mask_holdout`.

Fallback if no SVR can be fit: column mean of the original matrix. The freeze records **zero** fallback events and SVR coverage 1.0 on all 40 slots.

Missing predictors at impute time are **not** mean-filled; the TARGET-WISE path asserts predictor finiteness.

## Baselines (fair RMSE comparison)

Mean, KNN (k=5), and MissForest-like (`IterativeImputer` + ExtraTrees) evaluated on the **same freeze masks** under `repeated_mask_holdout` (`experiments/run_fair_baseline_holdout_vs_rfeca.py`).

A separate older campaign nested those imputers **inside** Stratified 5-fold CV (`experiments/run_full_metabric.py`). That campaign also included legacy `RFECA_SVR(k=*)`. Those k-fixed variants are **not** the principal method.

## Downstream subtype classification

Labels: PAM50 4-class (`LumA`, `LumB`, `Her2`, `Basal`).

**OriginalRFECA path** (`experiments/run_original_rfeca_classification.py`): reconstruct the dense matrix, then StratifiedKFold(5) with an identity imputer (no re-estimation inside folds). Classifiers: SVC, logistic regression, random forest, gradient boosting, and **EnsembleSoft** (soft vote of the four). Primary reported classifier: EnsembleSoft. CV `random_state=42`.

**Baseline CV path:** imputer is inside the classification pipeline (different nesting). Macro-F1 numbers from the two paths are **not** a paired confirmatory contrast. See `artifacts/final_analysis/central/caveats.md`.

Metrics: holdout RMSE/MAE on masked cells (estimation); macro-F1 and balanced accuracy (classification).

## What this document does not change

Predictor selection, missingness generation, leakage constraints, and evaluation protocols are defined by the code and freeze config. Repository cleanup must not alter them.
