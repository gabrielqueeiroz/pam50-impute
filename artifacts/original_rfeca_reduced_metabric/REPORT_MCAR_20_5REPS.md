# REPORT MCAR 20% × 5 réplicas

## Status

- Classification: **A**
- Replicates done: 5 / 5
- Genes completed per rep: [50, 50, 50, 50, 50]
- Failures per rep: [0, 0, 0, 0, 0]
- SVR coverage (min): 1.000000
- Total fallback_count: 0
- Wall clock total: 5.31 h

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
| RMSE | 0.614263 | 0.004707 | 0.614433 | 0.608457 | 0.621039 |
| MAE | 0.405514 | 0.002326 | — | — | — |

## Prefix / predictor distributions

- Prefix length: mean=43.59, median=46.0, min=6, max=49
- Final predictors: mean=21.56, median=23.0, min=3, max=24


## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
| 0 | 0.621039 | 0.408064 | 50 | 0 | 1.0000 | 0 | 3268.4 | A |
| 1 | 0.614433 | 0.406529 | 50 | 0 | 1.0000 | 0 | 2880.8 | A |
| 2 | 0.611654 | 0.403038 | 50 | 0 | 1.0000 | 0 | 4360.5 | A |
| 3 | 0.608457 | 0.403046 | 50 | 0 | 1.0000 | 0 | 4351.5 | A |
| 4 | 0.615731 | 0.406891 | 50 | 0 | 1.0000 | 0 | 4262.4 | A |

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mcar\rate_0.20`
