# REPORT MAR 5% × 5 réplicas

## Status

- Classification: **A**
- Replicates done: 5 / 5
- Genes completed per rep: [50, 50, 50, 50, 50]
- Failures per rep: [0, 0, 0, 0, 0]
- SVR coverage (min): 1.000000
- Total fallback_count: 0
- Wall clock total: 7.96 h

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
| RMSE | 0.641425 | 0.014924 | 0.641643 | 0.622403 | 0.663562 |
| MAE | 0.420758 | 0.005661 | — | — | — |

## Prefix / predictor distributions

- Prefix length: mean=44.08, median=47.0, min=6, max=49
- Final predictors: mean=21.78, median=23.0, min=3, max=24


## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
| 0 | 0.663562 | 0.427575 | 50 | 0 | 1.0000 | 0 | 5709.6 | A |
| 1 | 0.635670 | 0.419643 | 50 | 0 | 1.0000 | 0 | 5573.3 | A |
| 2 | 0.622403 | 0.412747 | 50 | 0 | 1.0000 | 0 | 5608.7 | A |
| 3 | 0.643845 | 0.419300 | 50 | 0 | 1.0000 | 0 | 5884.0 | A |
| 4 | 0.641643 | 0.424525 | 50 | 0 | 1.0000 | 0 | 5866.6 | A |

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mar\rate_0.05`
