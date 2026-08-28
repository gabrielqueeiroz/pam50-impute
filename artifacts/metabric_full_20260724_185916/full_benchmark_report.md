# METABRIC full-benchmark report

- **Status:** PASS
- **Run ID:** 20260724_185916
- **Output:** `<repo-root>\artifacts\metabric_full_20260724_185916`

## Cohort
- n: **1608**
- Class distribution: `{'LumA': 700, 'LumB': 475, 'Her2': 224, 'Basal': 209}`

## Protocol
- Rates: `[0.0, 0.05, 0.1, 0.2, 0.3]`
- Reps × folds: 10 × 5
- Imputers: `['SimpleMean', 'KNN(k=5,dist)', 'RFECA_SVR(k=5)', 'RFECA_SVR(k=10)', 'RFECA_SVR(k=20)', 'MissForest']`
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
- **KNN(k=5,dist)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **KNN(k=5,dist)** @ 0.05: RMSE=0.6795±0.0315, MAE=0.4634±0.0200, corr_RV=0.9994±0.0001, corr_Frel=0.0348
- **KNN(k=5,dist)** @ 0.1: RMSE=0.6915±0.0209, MAE=0.4690±0.0123, corr_RV=0.9987±0.0002, corr_Frel=0.0544
- **KNN(k=5,dist)** @ 0.2: RMSE=0.7096±0.0187, MAE=0.4818±0.0105, corr_RV=0.9970±0.0005, corr_Frel=0.0892
- **KNN(k=5,dist)** @ 0.3: RMSE=0.7336±0.0175, MAE=0.4992±0.0100, corr_RV=0.9945±0.0008, corr_Frel=0.1242
- **MissForest** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **MissForest** @ 0.05: RMSE=0.6172±0.0318, MAE=0.4148±0.0180, corr_RV=0.9995±0.0001, corr_Frel=0.0327
- **MissForest** @ 0.1: RMSE=0.6236±0.0178, MAE=0.4167±0.0100, corr_RV=0.9990±0.0001, corr_Frel=0.0517
- **MissForest** @ 0.2: RMSE=0.6371±0.0167, MAE=0.4260±0.0088, corr_RV=0.9977±0.0003, corr_Frel=0.0856
- **MissForest** @ 0.3: RMSE=0.6532±0.0145, MAE=0.4367±0.0077, corr_RV=0.9958±0.0005, corr_Frel=0.1215
- **RFECA_SVR(k=10)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=10)** @ 0.05: RMSE=0.6778±0.0391, MAE=0.4502±0.0213, corr_RV=0.9994±0.0001, corr_Frel=0.0361
- **RFECA_SVR(k=10)** @ 0.1: RMSE=0.6891±0.0240, MAE=0.4561±0.0119, corr_RV=0.9987±0.0002, corr_Frel=0.0577
- **RFECA_SVR(k=10)** @ 0.2: RMSE=0.7059±0.0175, MAE=0.4700±0.0091, corr_RV=0.9966±0.0005, corr_Frel=0.0986
- **RFECA_SVR(k=10)** @ 0.3: RMSE=0.7293±0.0179, MAE=0.4876±0.0089, corr_RV=0.9936±0.0007, corr_Frel=0.1383
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=0.6734±0.0346, MAE=0.4479±0.0199, corr_RV=0.9994±0.0001, corr_Frel=0.0352
- **RFECA_SVR(k=20)** @ 0.1: RMSE=0.6859±0.0230, MAE=0.4545±0.0112, corr_RV=0.9987±0.0002, corr_Frel=0.0555
- **RFECA_SVR(k=20)** @ 0.2: RMSE=0.7059±0.0181, MAE=0.4705±0.0094, corr_RV=0.9968±0.0004, corr_Frel=0.0933
- **RFECA_SVR(k=20)** @ 0.3: RMSE=0.7361±0.0174, MAE=0.4933±0.0086, corr_RV=0.9941±0.0006, corr_Frel=0.1290
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=0.6931±0.0399, MAE=0.4604±0.0202, corr_RV=0.9994±0.0001, corr_Frel=0.0371
- **RFECA_SVR(k=5)** @ 0.1: RMSE=0.7058±0.0230, MAE=0.4668±0.0115, corr_RV=0.9985±0.0002, corr_Frel=0.0594
- **RFECA_SVR(k=5)** @ 0.2: RMSE=0.7231±0.0183, MAE=0.4816±0.0090, corr_RV=0.9963±0.0005, corr_Frel=0.0996
- **RFECA_SVR(k=5)** @ 0.3: RMSE=0.7433±0.0179, MAE=0.4970±0.0090, corr_RV=0.9931±0.0010, corr_Frel=0.1375
- **SimpleMean** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **SimpleMean** @ 0.05: RMSE=1.0667±0.0413, MAE=0.7511±0.0268, corr_RV=0.9986±0.0002, corr_Frel=0.0674
- **SimpleMean** @ 0.1: RMSE=1.0633±0.0334, MAE=0.7468±0.0214, corr_RV=0.9966±0.0004, corr_Frel=0.1142
- **SimpleMean** @ 0.2: RMSE=1.0675±0.0235, MAE=0.7506±0.0140, corr_RV=0.9903±0.0009, corr_Frel=0.2075
- **SimpleMean** @ 0.3: RMSE=1.0688±0.0168, MAE=0.7539±0.0100, corr_RV=0.9796±0.0017, corr_Frel=0.3013

## Classification EnsembleSoft (by imputer × rate)
- **KNN(k=5,dist)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **KNN(k=5,dist)** @ 0.05: F1=0.8812±0.0131, BalAcc=0.8717±0.0145
- **KNN(k=5,dist)** @ 0.1: F1=0.8757±0.0130, BalAcc=0.8661±0.0135
- **KNN(k=5,dist)** @ 0.2: F1=0.8691±0.0158, BalAcc=0.8596±0.0175
- **KNN(k=5,dist)** @ 0.3: F1=0.8582±0.0171, BalAcc=0.8484±0.0182
- **MissForest** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **MissForest** @ 0.05: F1=0.8812±0.0122, BalAcc=0.8717±0.0134
- **MissForest** @ 0.1: F1=0.8787±0.0155, BalAcc=0.8693±0.0159
- **MissForest** @ 0.2: F1=0.8742±0.0147, BalAcc=0.8646±0.0164
- **MissForest** @ 0.3: F1=0.8662±0.0145, BalAcc=0.8574±0.0163
- **RFECA_SVR(k=10)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=10)** @ 0.05: F1=0.8822±0.0122, BalAcc=0.8725±0.0137
- **RFECA_SVR(k=10)** @ 0.1: F1=0.8781±0.0143, BalAcc=0.8684±0.0149
- **RFECA_SVR(k=10)** @ 0.2: F1=0.8700±0.0152, BalAcc=0.8608±0.0160
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8639±0.0166, BalAcc=0.8548±0.0172
- **RFECA_SVR(k=20)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=20)** @ 0.05: F1=0.8822±0.0134, BalAcc=0.8723±0.0144
- **RFECA_SVR(k=20)** @ 0.1: F1=0.8788±0.0144, BalAcc=0.8690±0.0154
- **RFECA_SVR(k=20)** @ 0.2: F1=0.8749±0.0161, BalAcc=0.8655±0.0163
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8627±0.0188, BalAcc=0.8543±0.0195
- **RFECA_SVR(k=5)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=5)** @ 0.05: F1=0.8812±0.0125, BalAcc=0.8716±0.0136
- **RFECA_SVR(k=5)** @ 0.1: F1=0.8771±0.0145, BalAcc=0.8672±0.0150
- **RFECA_SVR(k=5)** @ 0.2: F1=0.8700±0.0175, BalAcc=0.8605±0.0178
- **RFECA_SVR(k=5)** @ 0.3: F1=0.8612±0.0163, BalAcc=0.8521±0.0166
- **SimpleMean** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **SimpleMean** @ 0.05: F1=0.8787±0.0139, BalAcc=0.8695±0.0152
- **SimpleMean** @ 0.1: F1=0.8748±0.0154, BalAcc=0.8653±0.0159
- **SimpleMean** @ 0.2: F1=0.8661±0.0128, BalAcc=0.8569±0.0143
- **SimpleMean** @ 0.3: F1=0.8551±0.0169, BalAcc=0.8453±0.0185

## Runtime
- Wall clock: **35401.2s** (9.83 h)
- Completed slots: 50 / 50
