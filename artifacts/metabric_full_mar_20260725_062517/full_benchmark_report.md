# METABRIC full-benchmark report

- **Status:** PASS
- **Run ID:** 20260725_062517
- **Output:** `<repo-root>\artifacts\metabric_full_mar_20260725_062517`

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
- **KNN(k=5,dist)** @ 0.05: RMSE=0.6997±0.0278, MAE=0.4777±0.0171, corr_RV=0.9993±0.0001, corr_Frel=0.0385
- **KNN(k=5,dist)** @ 0.1: RMSE=0.7099±0.0249, MAE=0.4831±0.0147, corr_RV=0.9985±0.0002, corr_Frel=0.0591
- **KNN(k=5,dist)** @ 0.2: RMSE=0.7507±0.0246, MAE=0.5090±0.0144, corr_RV=0.9962±0.0006, corr_Frel=0.0956
- **KNN(k=5,dist)** @ 0.3: RMSE=0.8567±0.0316, MAE=0.5750±0.0162, corr_RV=0.9924±0.0014, corr_Frel=0.1263
- **MissForest** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **MissForest** @ 0.05: RMSE=0.6291±0.0278, MAE=0.4261±0.0178, corr_RV=0.9994±0.0001, corr_Frel=0.0357
- **MissForest** @ 0.1: RMSE=0.6428±0.0230, MAE=0.4307±0.0135, corr_RV=0.9988±0.0002, corr_Frel=0.0559
- **MissForest** @ 0.2: RMSE=0.6721±0.0236, MAE=0.4453±0.0117, corr_RV=0.9971±0.0005, corr_Frel=0.0905
- **MissForest** @ 0.3: RMSE=0.7053±0.0256, MAE=0.4650±0.0125, corr_RV=0.9950±0.0008, corr_Frel=0.1254
- **RFECA_SVR(k=10)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=10)** @ 0.05: RMSE=0.7055±0.0305, MAE=0.4698±0.0171, corr_RV=0.9993±0.0001, corr_Frel=0.0398
- **RFECA_SVR(k=10)** @ 0.1: RMSE=0.7192±0.0258, MAE=0.4798±0.0137, corr_RV=0.9984±0.0003, corr_Frel=0.0611
- **RFECA_SVR(k=10)** @ 0.2: RMSE=0.7638±0.0204, MAE=0.5064±0.0121, corr_RV=0.9959±0.0006, corr_Frel=0.0955
- **RFECA_SVR(k=10)** @ 0.3: RMSE=0.8100±0.0220, MAE=0.5382±0.0117, corr_RV=0.9920±0.0012, corr_Frel=0.1311
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=0.7003±0.0282, MAE=0.4683±0.0176, corr_RV=0.9993±0.0001, corr_Frel=0.0387
- **RFECA_SVR(k=20)** @ 0.1: RMSE=0.7172±0.0266, MAE=0.4789±0.0152, corr_RV=0.9984±0.0003, corr_Frel=0.0592
- **RFECA_SVR(k=20)** @ 0.2: RMSE=0.7673±0.0203, MAE=0.5105±0.0108, corr_RV=0.9960±0.0006, corr_Frel=0.0928
- **RFECA_SVR(k=20)** @ 0.3: RMSE=0.8156±0.0226, MAE=0.5433±0.0125, corr_RV=0.9922±0.0013, corr_Frel=0.1273
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=0.7213±0.0323, MAE=0.4808±0.0187, corr_RV=0.9992±0.0001, corr_Frel=0.0404
- **RFECA_SVR(k=5)** @ 0.1: RMSE=0.7414±0.0247, MAE=0.4942±0.0142, corr_RV=0.9982±0.0003, corr_Frel=0.0626
- **RFECA_SVR(k=5)** @ 0.2: RMSE=0.7873±0.0208, MAE=0.5212±0.0132, corr_RV=0.9953±0.0007, corr_Frel=0.0983
- **RFECA_SVR(k=5)** @ 0.3: RMSE=0.8351±0.0234, MAE=0.5524±0.0128, corr_RV=0.9910±0.0014, corr_Frel=0.1351
- **SimpleMean** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **SimpleMean** @ 0.05: RMSE=1.1689±0.0405, MAE=0.8279±0.0239, corr_RV=0.9978±0.0004, corr_Frel=0.0904
- **SimpleMean** @ 0.1: RMSE=1.1602±0.0268, MAE=0.8191±0.0183, corr_RV=0.9943±0.0008, corr_Frel=0.1551
- **SimpleMean** @ 0.2: RMSE=1.1491±0.0215, MAE=0.8113±0.0151, corr_RV=0.9831±0.0021, corr_Frel=0.2687
- **SimpleMean** @ 0.3: RMSE=1.1430±0.0177, MAE=0.8029±0.0138, corr_RV=0.9664±0.0040, corr_Frel=0.3624

## Classification EnsembleSoft (by imputer × rate)
- **KNN(k=5,dist)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **KNN(k=5,dist)** @ 0.05: F1=0.8803±0.0138, BalAcc=0.8703±0.0144
- **KNN(k=5,dist)** @ 0.1: F1=0.8779±0.0141, BalAcc=0.8680±0.0147
- **KNN(k=5,dist)** @ 0.2: F1=0.8722±0.0146, BalAcc=0.8621±0.0168
- **KNN(k=5,dist)** @ 0.3: F1=0.8482±0.0176, BalAcc=0.8381±0.0201
- **MissForest** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **MissForest** @ 0.05: F1=0.8806±0.0131, BalAcc=0.8707±0.0143
- **MissForest** @ 0.1: F1=0.8792±0.0132, BalAcc=0.8692±0.0139
- **MissForest** @ 0.2: F1=0.8708±0.0146, BalAcc=0.8614±0.0169
- **MissForest** @ 0.3: F1=0.8606±0.0159, BalAcc=0.8506±0.0186
- **RFECA_SVR(k=10)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=10)** @ 0.05: F1=0.8812±0.0145, BalAcc=0.8713±0.0154
- **RFECA_SVR(k=10)** @ 0.1: F1=0.8769±0.0138, BalAcc=0.8679±0.0148
- **RFECA_SVR(k=10)** @ 0.2: F1=0.8709±0.0172, BalAcc=0.8624±0.0179
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8576±0.0164, BalAcc=0.8487±0.0191
- **RFECA_SVR(k=20)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=20)** @ 0.05: F1=0.8816±0.0126, BalAcc=0.8716±0.0141
- **RFECA_SVR(k=20)** @ 0.1: F1=0.8780±0.0142, BalAcc=0.8685±0.0164
- **RFECA_SVR(k=20)** @ 0.2: F1=0.8710±0.0167, BalAcc=0.8627±0.0180
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8586±0.0148, BalAcc=0.8498±0.0171
- **RFECA_SVR(k=5)** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **RFECA_SVR(k=5)** @ 0.05: F1=0.8801±0.0130, BalAcc=0.8706±0.0136
- **RFECA_SVR(k=5)** @ 0.1: F1=0.8788±0.0128, BalAcc=0.8690±0.0148
- **RFECA_SVR(k=5)** @ 0.2: F1=0.8709±0.0159, BalAcc=0.8621±0.0174
- **RFECA_SVR(k=5)** @ 0.3: F1=0.8557±0.0166, BalAcc=0.8462±0.0183
- **SimpleMean** @ 0.0: F1=0.8859±0.0139, BalAcc=0.8752±0.0131
- **SimpleMean** @ 0.05: F1=0.8807±0.0143, BalAcc=0.8710±0.0159
- **SimpleMean** @ 0.1: F1=0.8743±0.0141, BalAcc=0.8652±0.0165
- **SimpleMean** @ 0.2: F1=0.8636±0.0160, BalAcc=0.8540±0.0170
- **SimpleMean** @ 0.3: F1=0.8480±0.0202, BalAcc=0.8379±0.0215

## Runtime
- Wall clock: **32662.4s** (9.07 h)
- Completed slots: 50 / 50
