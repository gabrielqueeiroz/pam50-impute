# METABRIC full-benchmark report

- **Status:** PASS
- **Run ID:** 20260727_202556
- **Output:** `<repo-root>\artifacts\metabric_full_mar_rfeca_20260727_202556`

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
- **RFECA_SVR(k=10)** @ 0.05: RMSE=0.6967±0.0355, MAE=0.4612±0.0174, corr_RV=0.9993±0.0002, corr_Frel=0.0404
- **RFECA_SVR(k=10)** @ 0.1: RMSE=0.7217±0.0321, MAE=0.4760±0.0167, corr_RV=0.9982±0.0003, corr_Frel=0.0663
- **RFECA_SVR(k=10)** @ 0.2: RMSE=0.7545±0.0211, MAE=0.5015±0.0107, corr_RV=0.9954±0.0006, corr_Frel=0.1086
- **RFECA_SVR(k=10)** @ 0.3: RMSE=0.7957±0.0220, MAE=0.5274±0.0111, corr_RV=0.9912±0.0010, corr_Frel=0.1498
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=0.6827±0.0356, MAE=0.4496±0.0184, corr_RV=0.9993±0.0001, corr_Frel=0.0389
- **RFECA_SVR(k=20)** @ 0.1: RMSE=0.7079±0.0315, MAE=0.4660±0.0164, corr_RV=0.9984±0.0003, corr_Frel=0.0634
- **RFECA_SVR(k=20)** @ 0.2: RMSE=0.7438±0.0225, MAE=0.4933±0.0107, corr_RV=0.9958±0.0006, corr_Frel=0.1054
- **RFECA_SVR(k=20)** @ 0.3: RMSE=0.7857±0.0227, MAE=0.5209±0.0113, corr_RV=0.9921±0.0009, corr_Frel=0.1474
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=0.7204±0.0339, MAE=0.4787±0.0173, corr_RV=0.9992±0.0002, corr_Frel=0.0413
- **RFECA_SVR(k=5)** @ 0.1: RMSE=0.7466±0.0332, MAE=0.4928±0.0173, corr_RV=0.9981±0.0003, corr_Frel=0.0659
- **RFECA_SVR(k=5)** @ 0.2: RMSE=0.7828±0.0229, MAE=0.5194±0.0127, corr_RV=0.9950±0.0007, corr_Frel=0.1050
- **RFECA_SVR(k=5)** @ 0.3: RMSE=0.8261±0.0240, MAE=0.5461±0.0132, corr_RV=0.9906±0.0010, corr_Frel=0.1421

## Classification EnsembleSoft (by imputer × rate)
- **RFECA_SVR(k=10)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=10)** @ 0.05: F1=0.8817±0.0138, BalAcc=0.8717±0.0142
- **RFECA_SVR(k=10)** @ 0.1: F1=0.8781±0.0122, BalAcc=0.8687±0.0140
- **RFECA_SVR(k=10)** @ 0.2: F1=0.8707±0.0164, BalAcc=0.8621±0.0176
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8609±0.0144, BalAcc=0.8527±0.0178
- **RFECA_SVR(k=20)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=20)** @ 0.05: F1=0.8814±0.0148, BalAcc=0.8714±0.0147
- **RFECA_SVR(k=20)** @ 0.1: F1=0.8777±0.0117, BalAcc=0.8683±0.0135
- **RFECA_SVR(k=20)** @ 0.2: F1=0.8716±0.0167, BalAcc=0.8627±0.0180
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8593±0.0156, BalAcc=0.8513±0.0190
- **RFECA_SVR(k=5)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=5)** @ 0.05: F1=0.8798±0.0149, BalAcc=0.8702±0.0153
- **RFECA_SVR(k=5)** @ 0.1: F1=0.8783±0.0124, BalAcc=0.8689±0.0135
- **RFECA_SVR(k=5)** @ 0.2: F1=0.8692±0.0144, BalAcc=0.8605±0.0162
- **RFECA_SVR(k=5)** @ 0.3: F1=0.8592±0.0147, BalAcc=0.8503±0.0177

## Runtime
- Wall clock: **12490.7s** (3.47 h)
- Completed slots: 50 / 50
