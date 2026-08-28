# REPORT MAR 10% × 5 réplicas

## Status

- Classification: **A**
- Replicates done: 5 / 5
- Genes completed per rep: [50, 50, 50, 50, 50]
- Failures per rep: [0, 0, 0, 0, 0]
- SVR coverage (min): 1.000000
- Total fallback_count: 0
- Wall clock total: 7.06 h

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
| RMSE | 0.638834 | 0.011372 | 0.639831 | 0.622027 | 0.653505 |
| MAE | 0.418484 | 0.005505 | — | — | — |

## Prefix / predictor distributions

- Prefix length: mean=43.62, median=46.0, min=6, max=49
- Final predictors: mean=21.58, median=23.0, min=3, max=24


## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
| 0 | 0.653505 | 0.427525 | 50 | 0 | 1.0000 | 0 | 5105.7 | A |
| 1 | 0.622027 | 0.413580 | 50 | 0 | 1.0000 | 0 | 5119.1 | A |
| 2 | 0.636411 | 0.416212 | 50 | 0 | 1.0000 | 0 | 5129.7 | A |
| 3 | 0.642395 | 0.419615 | 50 | 0 | 1.0000 | 0 | 5076.4 | A |
| 4 | 0.639831 | 0.415486 | 50 | 0 | 1.0000 | 0 | 4980.3 | A |

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mar\rate_0.10`
