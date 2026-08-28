# REPORT MCAR 30% × 5 réplicas

## Status

- Classification: **A**
- Replicates done: 5 / 5
- Genes completed per rep: [50, 50, 50, 50, 50]
- Failures per rep: [0, 0, 0, 0, 0]
- SVR coverage (min): 1.000000
- Total fallback_count: 0
- Wall clock total: 4.83 h

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
| RMSE | 0.621264 | 0.004157 | 0.622379 | 0.616045 | 0.626485 |
| MAE | 0.409487 | 0.002450 | — | — | — |

## Prefix / predictor distributions

- Prefix length: mean=42.85, median=46.0, min=8, max=49
- Final predictors: mean=21.18, median=23.0, min=4, max=24


## Per-replicate summaries

| Rep | RMSE | MAE | Genes OK | Failures | Cov | Fallbacks | Wall (s) | Class |
|-----|------|-----|----------|----------|-----|-----------|----------|-------|
| 0 | 0.616045 | 0.406795 | 50 | 0 | 1.0000 | 0 | 3484.8 | A |
| 1 | 0.626485 | 0.411195 | 50 | 0 | 1.0000 | 0 | 3459.8 | A |
| 2 | 0.618180 | 0.407004 | 50 | 0 | 1.0000 | 0 | 3488.3 | A |
| 3 | 0.623232 | 0.412132 | 50 | 0 | 1.0000 | 0 | 3498.1 | A |
| 4 | 0.622379 | 0.410311 | 50 | 0 | 1.0000 | 0 | 3456.0 | A |

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mcar\rate_0.30`
