# Discovery smoke-test report

- **Status:** PASS
- **Ready for full discovery benchmark:** True
- **Run ID:** 20260720_030707
- **Output:** `<repo-root>\artifacts\discovery_smoke_20260720_030707`

## Cohort
- n: **117**
- Class distribution: `{'LumA': 57, 'Basal': 29, 'LumB': 17, 'Her2': 14}`
- Provenance confidence: **High — CPTAC-derived laboratory cohort**
- Sample-ID status: `{'status': 'recovered_Patient_ID', 'biological_ids_available': True, 'mapping_audit': '<repo-root>\\artifacts\\discovery_preparation\\discovery_sample_id_mapping.csv'}`

## Fingerprint
- values+labels: `282a34939979593ddf95ff172be9da60c6e2c552bbb104b25fb866c9b01b35ff`

## Assertions
- `mask_feature_only`: **PASS**
- `complete_matrix_not_mutated`: **PASS**
- `shared_masks_across_imputers`: **PASS**
- `artificial_mask_observed_only`: **PASS**
- `biological_ids_recovered`: **PASS**
- `observation_mask_aligned`: **PASS**
- `metrics_on_masked_cells_only`: **PASS**
- `no_legacy_imputed_in_metric_target`: **PASS**
- `train_val_no_overlap`: **PASS**
- `no_shared_imputer_or_scaler_or_corr_across_folds`: **PASS**
- `rfeca_training_fold_correlation_only`: **PASS**
- `loader_schema_parity_with_metabric`: **PASS**

## Imputation
- **KNN(k=5,dist)**: RMSE=1.9504±0.1836, MAE=1.3336±0.1341
- **RFECA_SVR(k=5)**: RMSE=1.9528±0.2449, MAE=1.3143±0.1431
- **SimpleMean**: RMSE=2.7897±0.3075, MAE=2.0483±0.2085

## Classification (EnsembleSoft)
- **KNN(k=5,dist)**: F1=0.9236±0.0409, BalAcc=0.9103±0.0477
- **RFECA_SVR(k=5)**: F1=0.9188±0.0378, BalAcc=0.8992±0.0415
- **SimpleMean**: F1=0.8973±0.0284, BalAcc=0.8774±0.0274

## Prior corrected smoke comparison
- Eligible cells now: **5804** (prior matrix cells: 5850)
- Masked @10% now: **580** (prior approx: 585)
- Cause: Prior smoke sampled artificial missingness from all 117x50 cells; current primary policy restricts masking and RMSE/MAE to originally observed cells only (legacy-imputed cells excluded).
- **KNN(k=5,dist)**: RMSE 2.1444->1.9504 (d=-0.1940); F1 0.9366->0.9236 (d=-0.0130)
- **RFECA_SVR(k=5)**: RMSE 2.1832->1.9528 (d=-0.2304); F1 0.9262->0.9188 (d=-0.0074)
- **SimpleMean**: RMSE 2.9562->2.7897 (d=-0.1665); F1 0.9480->0.8973 (d=-0.0507)

## Metric parity vs historical Colab
- **KNN(k=5,dist)**: smoke F1=0.9236 vs hist 0.9226 (delta=+0.0010); BalAcc delta=-0.0037. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- **RFECA_SVR(k=5)**: smoke F1=0.9188 vs hist 0.9195 (delta=-0.0006); BalAcc delta=-0.0090. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- **SimpleMean**: smoke F1=0.8973 vs hist 0.9259 (delta=-0.0286); BalAcc delta=-0.0378. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- Note: Historical run: n_reps=10, n_splits=5 (smoke uses 2 reps / 3 splits).
- Note: Historical RFECA used precomputed full-cohort correlation CSV (leakage).
- Note: Corrected pipeline recomputes correlations inside each training fold.
- Note: Therefore exact numeric parity is NOT expected; deviations are informative.

## Runtime
- Wall clock: **15.19s**

**Stopped after discovery smoke test — full benchmark not started.**