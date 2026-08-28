# Discovery full-benchmark report

- **Status:** PASS
- **Run ID:** 20260727_201003
- **Output:** `<repo-root>\artifacts\discovery_full_mar_rfeca_20260727_201003`

## Cohort
- n: **117**
- Class distribution: `{'LumA': 57, 'Basal': 29, 'LumB': 17, 'Her2': 14}`
- Provenance: **High — CPTAC-derived laboratory cohort**

## Protocol
- Rates: `[0.0, 0.05, 0.1, 0.2, 0.3]`
- Reps × folds: 10 × 5
- Imputers: `['RFECA_SVR(k=5)', 'RFECA_SVR(k=10)', 'RFECA_SVR(k=20)']`
- Target cell policy: `originally_observed_only`
- Execution: **sequential** (n_jobs forced to 1)

## Assertions
- `mask_feature_only`: **PASS**
- `complete_matrix_not_mutated`: **PASS**
- `shared_masks_across_imputers`: **PASS**
- `artificial_mask_observed_only`: **PASS**
- `metrics_on_masked_cells_only`: **PASS**
- `no_legacy_imputed_in_metric_target`: **PASS**
- `train_val_no_overlap`: **PASS**
- `no_shared_imputer_or_scaler_or_corr_across_folds`: **PASS**
- `rfeca_training_fold_correlation_only`: **PASS**

## Imputation (by imputer × rate)
- **RFECA_SVR(k=10)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=10)** @ 0.05: RMSE=1.9827±0.4881, MAE=1.3381±0.2832, corr_RV=0.9946±0.0035, corr_Frel=0.1009
- **RFECA_SVR(k=10)** @ 0.1: RMSE=2.1289±0.4652, MAE=1.4202±0.2811, corr_RV=0.9893±0.0045, corr_Frel=0.1460
- **RFECA_SVR(k=10)** @ 0.2: RMSE=2.1716±0.4273, MAE=1.4563±0.2528, corr_RV=0.9787±0.0070, corr_Frel=0.2103
- **RFECA_SVR(k=10)** @ 0.3: RMSE=2.2875±0.3485, MAE=1.5302±0.2062, corr_RV=0.9632±0.0109, corr_Frel=0.2798
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=2.0654±0.5462, MAE=1.4075±0.3324, corr_RV=0.9940±0.0042, corr_Frel=0.1052
- **RFECA_SVR(k=20)** @ 0.1: RMSE=2.2301±0.4988, MAE=1.5066±0.3078, corr_RV=0.9882±0.0055, corr_Frel=0.1513
- **RFECA_SVR(k=20)** @ 0.2: RMSE=2.3048±0.3906, MAE=1.5631±0.2374, corr_RV=0.9766±0.0072, corr_Frel=0.2180
- **RFECA_SVR(k=20)** @ 0.3: RMSE=2.4727±0.3254, MAE=1.6606±0.1972, corr_RV=0.9580±0.0108, corr_Frel=0.2931
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=1.9406±0.4678, MAE=1.3053±0.2763, corr_RV=0.9949±0.0029, corr_Frel=0.0981
- **RFECA_SVR(k=5)** @ 0.1: RMSE=2.1161±0.4988, MAE=1.4154±0.3030, corr_RV=0.9893±0.0047, corr_Frel=0.1446
- **RFECA_SVR(k=5)** @ 0.2: RMSE=2.1521±0.4272, MAE=1.4441±0.2536, corr_RV=0.9791±0.0069, corr_Frel=0.2062
- **RFECA_SVR(k=5)** @ 0.3: RMSE=2.2657±0.3543, MAE=1.5147±0.2064, corr_RV=0.9642±0.0104, corr_Frel=0.2721

## Classification EnsembleSoft (by imputer × rate)
- **RFECA_SVR(k=10)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=10)** @ 0.05: F1=0.9234±0.0498, BalAcc=0.9133±0.0596
- **RFECA_SVR(k=10)** @ 0.1: F1=0.9148±0.0481, BalAcc=0.9066±0.0560
- **RFECA_SVR(k=10)** @ 0.2: F1=0.9255±0.0561, BalAcc=0.9170±0.0629
- **RFECA_SVR(k=10)** @ 0.3: F1=0.9047±0.0442, BalAcc=0.8967±0.0542
- **RFECA_SVR(k=20)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=20)** @ 0.05: F1=0.9233±0.0453, BalAcc=0.9142±0.0548
- **RFECA_SVR(k=20)** @ 0.1: F1=0.9152±0.0508, BalAcc=0.9042±0.0583
- **RFECA_SVR(k=20)** @ 0.2: F1=0.9199±0.0516, BalAcc=0.9108±0.0593
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8938±0.0637, BalAcc=0.8833±0.0709
- **RFECA_SVR(k=5)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=5)** @ 0.05: F1=0.9179±0.0458, BalAcc=0.9087±0.0536
- **RFECA_SVR(k=5)** @ 0.1: F1=0.9123±0.0500, BalAcc=0.9045±0.0567
- **RFECA_SVR(k=5)** @ 0.2: F1=0.9174±0.0595, BalAcc=0.9095±0.0644
- **RFECA_SVR(k=5)** @ 0.3: F1=0.9078±0.0530, BalAcc=0.9008±0.0621

## Runtime
- Wall clock: **932.3s** (15.5 min)
- Completed slots: 50 / 50
