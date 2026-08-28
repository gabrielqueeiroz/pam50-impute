# METABRIC full-benchmark report

- **Status:** PASS
- **Run ID:** 20260727_163213
- **Output:** `<repo-root>\artifacts\metabric_full_rfeca_20260727_163213`

## Cohort
- n: **1608**
- Class distribution: `{'LumA': 700, 'LumB': 475, 'Her2': 224, 'Basal': 209}`

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
- **RFECA_SVR(k=10)** @ 0.05: RMSE=0.6634±0.0269, MAE=0.4420±0.0161, corr_RV=0.9994±0.0001, corr_Frel=0.0362
- **RFECA_SVR(k=10)** @ 0.1: RMSE=0.6738±0.0257, MAE=0.4481±0.0143, corr_RV=0.9987±0.0002, corr_Frel=0.0591
- **RFECA_SVR(k=10)** @ 0.2: RMSE=0.6954±0.0178, MAE=0.4635±0.0086, corr_RV=0.9963±0.0004, corr_Frel=0.1095
- **RFECA_SVR(k=10)** @ 0.3: RMSE=0.7321±0.0178, MAE=0.4890±0.0093, corr_RV=0.9918±0.0012, corr_Frel=0.1767
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=0.6488±0.0263, MAE=0.4319±0.0150, corr_RV=0.9995±0.0001, corr_Frel=0.0349
- **RFECA_SVR(k=20)** @ 0.1: RMSE=0.6588±0.0258, MAE=0.4376±0.0139, corr_RV=0.9988±0.0002, corr_Frel=0.0563
- **RFECA_SVR(k=20)** @ 0.2: RMSE=0.6836±0.0177, MAE=0.4551±0.0082, corr_RV=0.9968±0.0004, corr_Frel=0.1055
- **RFECA_SVR(k=20)** @ 0.3: RMSE=0.7243±0.0180, MAE=0.4834±0.0093, corr_RV=0.9925±0.0011, corr_Frel=0.1757
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=0.6842±0.0267, MAE=0.4561±0.0157, corr_RV=0.9994±0.0001, corr_Frel=0.0371
- **RFECA_SVR(k=5)** @ 0.1: RMSE=0.6930±0.0275, MAE=0.4620±0.0147, corr_RV=0.9986±0.0002, corr_Frel=0.0594
- **RFECA_SVR(k=5)** @ 0.2: RMSE=0.7166±0.0175, MAE=0.4778±0.0081, corr_RV=0.9961±0.0005, corr_Frel=0.1057
- **RFECA_SVR(k=5)** @ 0.3: RMSE=0.7466±0.0182, MAE=0.4995±0.0091, corr_RV=0.9918±0.0010, corr_Frel=0.1609

## Classification EnsembleSoft (by imputer × rate)
- **RFECA_SVR(k=10)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=10)** @ 0.05: F1=0.8823±0.0136, BalAcc=0.8721±0.0142
- **RFECA_SVR(k=10)** @ 0.1: F1=0.8778±0.0160, BalAcc=0.8688±0.0162
- **RFECA_SVR(k=10)** @ 0.2: F1=0.8705±0.0149, BalAcc=0.8617±0.0169
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8627±0.0178, BalAcc=0.8543±0.0203
- **RFECA_SVR(k=20)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=20)** @ 0.05: F1=0.8829±0.0123, BalAcc=0.8725±0.0126
- **RFECA_SVR(k=20)** @ 0.1: F1=0.8777±0.0152, BalAcc=0.8684±0.0145
- **RFECA_SVR(k=20)** @ 0.2: F1=0.8735±0.0155, BalAcc=0.8644±0.0169
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8640±0.0162, BalAcc=0.8555±0.0189
- **RFECA_SVR(k=5)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=5)** @ 0.05: F1=0.8835±0.0133, BalAcc=0.8736±0.0134
- **RFECA_SVR(k=5)** @ 0.1: F1=0.8770±0.0155, BalAcc=0.8679±0.0148
- **RFECA_SVR(k=5)** @ 0.2: F1=0.8708±0.0161, BalAcc=0.8620±0.0184
- **RFECA_SVR(k=5)** @ 0.3: F1=0.8609±0.0194, BalAcc=0.8521±0.0209

## Runtime
- Wall clock: **12649.9s** (3.51 h)
- Completed slots: 50 / 50
