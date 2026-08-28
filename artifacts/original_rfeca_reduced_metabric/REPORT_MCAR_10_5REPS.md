# REPORT MCAR 10% × 5 réplicas

## Status

- Classification: **A**
- Replicates done: 5 / 5
- Genes completed per rep: [50, 50, 50, 50, 50]
- Failures per rep: [0, 0, 0, 0, 0]
- SVR coverage (min): 1.000000
- Total fallback_count: 0
- Wall clock total: 6.99 h

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
| RMSE | 0.614648 | 0.007803 | 0.613987 | 0.605792 | 0.625205 |
| MAE | 0.406857 | 0.003057 | — | — | — |

## Prefix / predictor distributions

- Prefix length: mean=44.48, median=47.0, min=6, max=49
- Final predictors: mean=21.98, median=23.0, min=3, max=24


## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
| 0 | 0.625205 | 0.410870 | 50 | 0 | 1.0000 | 0 | 5214.8 | A |
| 1 | 0.619262 | 0.406688 | 50 | 0 | 1.0000 | 0 | 4955.4 | A |
| 2 | 0.608996 | 0.403443 | 50 | 0 | 1.0000 | 0 | 4905.5 | A |
| 3 | 0.605792 | 0.404464 | 50 | 0 | 1.0000 | 0 | 4893.9 | A |
| 4 | 0.613987 | 0.408821 | 50 | 0 | 1.0000 | 0 | 5186.9 | A |

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mcar\rate_0.10`
