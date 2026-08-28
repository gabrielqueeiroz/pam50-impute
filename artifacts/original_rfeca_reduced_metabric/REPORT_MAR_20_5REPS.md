# REPORT MAR 20% × 5 réplicas

## Status

- Classification: **A**
- Replicates done: 5 / 5
- Genes completed per rep: [50, 50, 50, 50, 50]
- Failures per rep: [0, 0, 0, 0, 0]
- SVR coverage (min): 1.000000
- Total fallback_count: 0
- Wall clock total: 5.85 h

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
- inner_validation = KFold(5)
- RFE = SVR(kernel=linear), step=1, n_features_to_select=None

## Aggregates across replicates

| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| RMSE | 0.639723 | 0.005242 | 0.641713 | 0.631468 | 0.644256 |
| MAE | 0.419278 | 0.002234 | — | — | — |

## Prefix / predictor distributions

- Prefix length: mean=42.71, median=46.0, min=8, max=49
- Final predictors: mean=21.11, median=23.0, min=4, max=24


## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
| 0 | 0.641713 | 0.419772 | 50 | 0 | 1.0000 | 0 | 4236.4 | A |
| 1 | 0.631468 | 0.416225 | 50 | 0 | 1.0000 | 0 | 4307.9 | A |
| 2 | 0.637785 | 0.417790 | 50 | 0 | 1.0000 | 0 | 4179.7 | A |
| 3 | 0.643394 | 0.421270 | 50 | 0 | 1.0000 | 0 | 4183.7 | A |
| 4 | 0.644256 | 0.421332 | 50 | 0 | 1.0000 | 0 | 4158.0 | A |

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mar\rate_0.20`
