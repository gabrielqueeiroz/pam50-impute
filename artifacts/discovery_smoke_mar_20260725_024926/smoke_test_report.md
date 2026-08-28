# Discovery smoke-test report

- **Status:** PASS
- **Ready for full discovery benchmark:** True
- **Run ID:** 20260725_024926
- **Output:** `<repo-root>\artifacts\discovery_smoke_mar_20260725_024926`

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
- **KNN(k=5,dist)**: RMSE=2.2464±0.3915, MAE=1.5757±0.2166, corr_RV=0.9913±0.0041, corr_Frel=0.1302
- **MissForest**: RMSE=2.0623±0.4003, MAE=1.4061±0.2154, corr_RV=0.9922±0.0049, corr_Frel=0.1224
- **RFECA_SVR(k=5)**: RMSE=2.1185±0.3980, MAE=1.4694±0.2479, corr_RV=0.9920±0.0043, corr_Frel=0.1243
- **SimpleMean**: RMSE=3.0181±0.3718, MAE=2.2129±0.2250, corr_RV=0.9844±0.0045, corr_Frel=0.1871

## Classification (EnsembleSoft)
- **KNN(k=5,dist)**: F1=0.9250±0.0437, BalAcc=0.9103±0.0486
- **MissForest**: F1=0.9259±0.0488, BalAcc=0.9178±0.0600
- **RFECA_SVR(k=5)**: F1=0.9191±0.0623, BalAcc=0.9045±0.0731
- **SimpleMean**: F1=0.9047±0.0655, BalAcc=0.8853±0.0757

## Prior corrected smoke comparison
- Eligible cells now: **5804** (prior matrix cells: 5850)
- Masked @10% now: **580** (prior approx: 585)
- Cause: Prior smoke sampled artificial missingness from all 117x50 cells; current primary policy restricts masking and RMSE/MAE to originally observed cells only (legacy-imputed cells excluded).
- **KNN(k=5,dist)**: RMSE 2.1444->2.2464 (d=+0.1020); F1 0.9366->0.9250 (d=-0.0116)
- **RFECA_SVR(k=5)**: RMSE 2.1832->2.1185 (d=-0.0647); F1 0.9262->0.9191 (d=-0.0071)
- **SimpleMean**: RMSE 2.9562->3.0181 (d=+0.0619); F1 0.9480->0.9047 (d=-0.0433)

## Metric parity vs historical Colab
- **KNN(k=5,dist)**: smoke F1=0.9250 vs hist 0.9226 (delta=+0.0025); BalAcc delta=-0.0037. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- **MissForest**: no historical reference row found.
- **RFECA_SVR(k=5)**: smoke F1=0.9191 vs hist 0.9195 (delta=-0.0004); BalAcc delta=-0.0037. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- **SimpleMean**: smoke F1=0.9047 vs hist 0.9259 (delta=-0.0211); BalAcc delta=-0.0300. Cause: Different CV budget (smoke 2x3 vs historical 10x5) and/or RFECA correlation leakage fix (training-fold only).
- Note: Historical run: n_reps=10, n_splits=5 (smoke uses 2 reps / 3 splits).
- Note: Historical RFECA used precomputed full-cohort correlation CSV (leakage).
- Note: Corrected pipeline recomputes correlations inside each training fold.
- Note: Therefore exact numeric parity is NOT expected; deviations are informative.

## Runtime
- Wall clock: **99.26s**

**Stopped after discovery smoke test — full benchmark not started.**