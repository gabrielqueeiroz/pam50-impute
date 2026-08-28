# Discovery smoke-test report

- **Status:** PASS
- **Ready for full discovery benchmark:** True
- **Run ID:** 20260719_213142
- **Output:** `<repo-root>\artifacts\discovery_smoke_20260719_213142`

## Cohort
- n: **117**
- Class distribution: `{'LumA': 57, 'Basal': 29, 'LumB': 17, 'Her2': 14}`
- Provenance confidence: **Moderate**
- Sample-ID status: `{'status': 'synthetic_row_ids', 'biological_ids_available': False, 'scheme': 'discovery_row_XXXX (1-based synthetic)'}`

## Fingerprint
- values+labels: `9994c4ed68d2c3c299aeb2a4d609c8edc091768b46d2dca77ce3f5c669838bea`

## Assertions
- `mask_feature_only`: **PASS**
- `complete_matrix_not_mutated`: **PASS**
- `shared_masks_across_imputers`: **PASS**
- `metrics_on_masked_cells_only`: **PASS**
- `train_val_no_overlap`: **PASS**
- `no_shared_imputer_or_scaler_or_corr_across_folds`: **PASS**
- `rfeca_training_fold_correlation_only`: **PASS**
- `loader_schema_parity_with_metabric`: **PASS**

## Imputation
- **KNN(k=5,dist)**: RMSE=2.1444±0.1666, MAE=1.4512±0.0975
- **RFECA_SVR(k=5)**: RMSE=2.1832±0.1765, MAE=1.4311±0.1115
- **SimpleMean**: RMSE=2.9562±0.1059, MAE=2.1530±0.0912

## Classification (EnsembleSoft)
- **KNN(k=5,dist)**: F1=0.9366±0.0463, BalAcc=0.9167±0.0624
- **RFECA_SVR(k=5)**: F1=0.9262±0.0269, BalAcc=0.9021±0.0353
- **SimpleMean**: F1=0.9480±0.0413, BalAcc=0.9271±0.0581

## Metric parity vs historical Colab
- **KNN(k=5,dist)**: smoke F1=0.9366 vs hist 0.9226 (Δ=+0.0140); BalAcc Δ=+0.0027. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- **RFECA_SVR(k=5)**: smoke F1=0.9262 vs hist 0.9195 (Δ=+0.0068); BalAcc Δ=-0.0061. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- **SimpleMean**: smoke F1=0.9480 vs hist 0.9259 (Δ=+0.0222); BalAcc Δ=+0.0118. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- Note: Historical run: n_reps=10, n_splits=5 (smoke uses 2 reps / 3 splits).
- Note: Historical RFECA used precomputed full-cohort correlation CSV (leakage).
- Note: Corrected pipeline recomputes correlations inside each training fold.
- Note: Therefore exact numeric parity is NOT expected; deviations are informative.

## Runtime
- Wall clock: **17.50s**

**Stopped after discovery smoke test — full benchmark not started.**