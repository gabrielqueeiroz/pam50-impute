# REPORT_RFECA_REDUCED_FINAL

## Classification: **A**

## Methodological justification

Devido ao elevado custo computacional do OriginalRFECA, sua avaliação foi restrita à taxa intermediária de 20% de ausência, sob os mecanismos MCAR e MAR, com cinco réplicas independentes. O experimento foi conduzido como análise complementar target-wise, utilizando preditores completos da matriz original e avaliando exclusivamente os valores artificialmente mascarados do gene-alvo.

## Protocol

- method = OriginalRFECA
- evaluation_protocol = repeated_mask_holdout
- input_protocol = target_wise_complete_predictors
- predictor_values = original_complete_matrix
- selection_protocol = leakage_safe
- use_scaler = false
- max_candidates = 49
- No RFACA, no outer CV, no auxiliary imputation, no chaining

## Scope justification

Only MCAR and MAR at 20% missingness with 5 independent replicates (10 slots × 50 genes).

## MCAR vs MAR at 20%

| Mechanism | RMSE mean±std | MAE mean | Wall (h) | Cov min | Fallbacks | Class |
|-----------|---------------|----------|----------|---------|-----------|-------|
| MCAR | 0.614263±0.004707 | 0.405514 | 5.31 | 1.0000 | 0 | A |
| MAR | 0.639723±0.005242 | 0.419278 | 5.85 | 1.0000 | 0 | A |

## Variability

MCAR RMSE range [0.608457, 0.621039];
MAR RMSE range [0.631468, 0.644256].

## Limitations

- Restricted to a single missingness rate (20%) due to compute cost.
- Serial gene execution; wall clock is large (~days).
- Target-wise protocol does not model joint multivariate missingness at prediction time.

Artifacts root: `<repo-root>\artifacts\original_rfeca_reduced_metabric`
