# Discovery full-benchmark report

- **Status:** PASS
- **Run ID:** 20260727_161656
- **Output:** `<repo-root>\artifacts\discovery_full_rfeca_20260727_161656`

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
- **RFECA_SVR(k=10)** @ 0.05: RMSE=1.8216±0.3984, MAE=1.2198±0.1967, corr_RV=0.9964±0.0015, corr_Frel=0.0836
- **RFECA_SVR(k=10)** @ 0.1: RMSE=1.8153±0.2770, MAE=1.2166±0.1253, corr_RV=0.9922±0.0026, corr_Frel=0.1260
- **RFECA_SVR(k=10)** @ 0.2: RMSE=2.0086±0.2004, MAE=1.3408±0.1142, corr_RV=0.9813±0.0047, corr_Frel=0.2003
- **RFECA_SVR(k=10)** @ 0.3: RMSE=2.1225±0.1939, MAE=1.4271±0.1278, corr_RV=0.9693±0.0077, corr_Frel=0.2641
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=1.8814±0.3972, MAE=1.2846±0.2059, corr_RV=0.9958±0.0021, corr_Frel=0.0896
- **RFECA_SVR(k=20)** @ 0.1: RMSE=1.9105±0.2914, MAE=1.2978±0.1507, corr_RV=0.9912±0.0029, corr_Frel=0.1328
- **RFECA_SVR(k=20)** @ 0.2: RMSE=2.0881±0.1879, MAE=1.4160±0.1154, corr_RV=0.9794±0.0053, corr_Frel=0.2080
- **RFECA_SVR(k=20)** @ 0.3: RMSE=2.2512±0.1945, MAE=1.5365±0.1284, corr_RV=0.9672±0.0076, corr_Frel=0.2678
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=1.8019±0.3956, MAE=1.2200±0.1984, corr_RV=0.9965±0.0015, corr_Frel=0.0834
- **RFECA_SVR(k=5)** @ 0.1: RMSE=1.8083±0.2909, MAE=1.2170±0.1323, corr_RV=0.9923±0.0024, corr_Frel=0.1249
- **RFECA_SVR(k=5)** @ 0.2: RMSE=1.9808±0.2002, MAE=1.3182±0.1107, corr_RV=0.9823±0.0045, corr_Frel=0.1934
- **RFECA_SVR(k=5)** @ 0.3: RMSE=2.0729±0.1674, MAE=1.3931±0.1090, corr_RV=0.9710±0.0063, corr_Frel=0.2524

## Classification EnsembleSoft (by imputer × rate)
- **RFECA_SVR(k=10)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=10)** @ 0.05: F1=0.9192±0.0448, BalAcc=0.9117±0.0572
- **RFECA_SVR(k=10)** @ 0.1: F1=0.9217±0.0471, BalAcc=0.9133±0.0571
- **RFECA_SVR(k=10)** @ 0.2: F1=0.9023±0.0611, BalAcc=0.8962±0.0710
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8986±0.0612, BalAcc=0.8904±0.0667
- **RFECA_SVR(k=20)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=20)** @ 0.05: F1=0.9204±0.0513, BalAcc=0.9121±0.0613
- **RFECA_SVR(k=20)** @ 0.1: F1=0.9272±0.0513, BalAcc=0.9192±0.0608
- **RFECA_SVR(k=20)** @ 0.2: F1=0.9068±0.0556, BalAcc=0.9008±0.0669
- **RFECA_SVR(k=20)** @ 0.3: F1=0.9003±0.0599, BalAcc=0.8925±0.0689
- **RFECA_SVR(k=5)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=5)** @ 0.05: F1=0.9210±0.0529, BalAcc=0.9146±0.0648
- **RFECA_SVR(k=5)** @ 0.1: F1=0.9303±0.0486, BalAcc=0.9217±0.0579
- **RFECA_SVR(k=5)** @ 0.2: F1=0.9027±0.0649, BalAcc=0.8975±0.0742
- **RFECA_SVR(k=5)** @ 0.3: F1=0.9017±0.0616, BalAcc=0.8928±0.0686

## Runtime
- Wall clock: **884.4s** (14.7 min)
- Completed slots: 50 / 50
