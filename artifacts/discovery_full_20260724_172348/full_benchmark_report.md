# Discovery full-benchmark report

- **Status:** PASS
- **Run ID:** 20260724_172348
- **Output:** `<repo-root>\artifacts\discovery_full_20260724_172348`

## Cohort
- n: **117**
- Class distribution: `{'LumA': 57, 'Basal': 29, 'LumB': 17, 'Her2': 14}`
- Provenance: **High — CPTAC-derived laboratory cohort**

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
- **KNN(k=5,dist)** @ 0.05: RMSE=2.0170±0.3386, MAE=1.3827±0.1899, corr_RV=0.9957±0.0015, corr_Frel=0.0915
- **KNN(k=5,dist)** @ 0.1: RMSE=2.0851±0.2420, MAE=1.4127±0.1363, corr_RV=0.9910±0.0030, corr_Frel=0.1328
- **KNN(k=5,dist)** @ 0.2: RMSE=2.0783±0.1550, MAE=1.4204±0.1027, corr_RV=0.9807±0.0052, corr_Frel=0.1970
- **KNN(k=5,dist)** @ 0.3: RMSE=2.1542±0.1852, MAE=1.4698±0.1042, corr_RV=0.9678±0.0093, corr_Frel=0.2552
- **MissForest** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **MissForest** @ 0.05: RMSE=1.7611±0.3020, MAE=1.1937±0.1825, corr_RV=0.9965±0.0012, corr_Frel=0.0824
- **MissForest** @ 0.1: RMSE=1.8334±0.2610, MAE=1.2276±0.1436, corr_RV=0.9930±0.0021, corr_Frel=0.1186
- **MissForest** @ 0.2: RMSE=1.8573±0.1452, MAE=1.2359±0.0855, corr_RV=0.9847±0.0045, corr_Frel=0.1781
- **MissForest** @ 0.3: RMSE=1.9253±0.1778, MAE=1.2800±0.0967, corr_RV=0.9760±0.0064, corr_Frel=0.2256
- **RFECA_SVR(k=10)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=10)** @ 0.05: RMSE=2.1477±0.4371, MAE=1.4476±0.2231, corr_RV=0.9950±0.0018, corr_Frel=0.0992
- **RFECA_SVR(k=10)** @ 0.1: RMSE=2.2507±0.3254, MAE=1.5391±0.1756, corr_RV=0.9887±0.0034, corr_Frel=0.1497
- **RFECA_SVR(k=10)** @ 0.2: RMSE=2.3227±0.1964, MAE=1.5917±0.1297, corr_RV=0.9724±0.0080, corr_Frel=0.2352
- **RFECA_SVR(k=10)** @ 0.3: RMSE=2.5592±0.2753, MAE=1.7459±0.1206, corr_RV=0.9461±0.0174, corr_Frel=0.3254
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=2.3880±0.4026, MAE=1.6445±0.2417, corr_RV=0.9933±0.0023, corr_Frel=0.1141
- **RFECA_SVR(k=20)** @ 0.1: RMSE=2.4792±0.3007, MAE=1.7028±0.1666, corr_RV=0.9858±0.0044, corr_Frel=0.1670
- **RFECA_SVR(k=20)** @ 0.2: RMSE=2.6261±0.1947, MAE=1.8189±0.1368, corr_RV=0.9655±0.0092, corr_Frel=0.2605
- **RFECA_SVR(k=20)** @ 0.3: RMSE=2.8258±0.2487, MAE=1.9609±0.1391, corr_RV=0.9332±0.0195, corr_Frel=0.3586
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=2.0406±0.3615, MAE=1.3645±0.1922, corr_RV=0.9955±0.0014, corr_Frel=0.0943
- **RFECA_SVR(k=5)** @ 0.1: RMSE=2.0960±0.2935, MAE=1.4247±0.1681, corr_RV=0.9903±0.0029, corr_Frel=0.1394
- **RFECA_SVR(k=5)** @ 0.2: RMSE=2.1743±0.1904, MAE=1.4834±0.1138, corr_RV=0.9771±0.0066, corr_Frel=0.2152
- **RFECA_SVR(k=5)** @ 0.3: RMSE=2.3204±0.1738, MAE=1.5942±0.1157, corr_RV=0.9577±0.0135, corr_Frel=0.2903
- **SimpleMean** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **SimpleMean** @ 0.05: RMSE=2.8081±0.4575, MAE=2.0590±0.3254, corr_RV=0.9914±0.0028, corr_Frel=0.1328
- **SimpleMean** @ 0.1: RMSE=2.8483±0.2896, MAE=2.0975±0.2025, corr_RV=0.9811±0.0040, corr_Frel=0.2003
- **SimpleMean** @ 0.2: RMSE=2.7961±0.2537, MAE=2.0629±0.1993, corr_RV=0.9560±0.0085, corr_Frel=0.3099
- **SimpleMean** @ 0.3: RMSE=2.7858±0.2198, MAE=2.0680±0.1743, corr_RV=0.9191±0.0157, corr_Frel=0.4155

## Classification EnsembleSoft (by imputer × rate)
- **KNN(k=5,dist)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **KNN(k=5,dist)** @ 0.05: F1=0.9245±0.0511, BalAcc=0.9200±0.0621
- **KNN(k=5,dist)** @ 0.1: F1=0.9250±0.0531, BalAcc=0.9158±0.0628
- **KNN(k=5,dist)** @ 0.2: F1=0.9122±0.0588, BalAcc=0.9041±0.0661
- **KNN(k=5,dist)** @ 0.3: F1=0.9084±0.0549, BalAcc=0.9011±0.0660
- **MissForest** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **MissForest** @ 0.05: F1=0.9224±0.0483, BalAcc=0.9158±0.0591
- **MissForest** @ 0.1: F1=0.9184±0.0492, BalAcc=0.9108±0.0564
- **MissForest** @ 0.2: F1=0.9083±0.0598, BalAcc=0.9049±0.0692
- **MissForest** @ 0.3: F1=0.9158±0.0476, BalAcc=0.9090±0.0570
- **RFECA_SVR(k=10)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=10)** @ 0.05: F1=0.9195±0.0465, BalAcc=0.9100±0.0576
- **RFECA_SVR(k=10)** @ 0.1: F1=0.9101±0.0478, BalAcc=0.9020±0.0575
- **RFECA_SVR(k=10)** @ 0.2: F1=0.8917±0.0528, BalAcc=0.8816±0.0609
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8952±0.0681, BalAcc=0.8891±0.0750
- **RFECA_SVR(k=20)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=20)** @ 0.05: F1=0.9186±0.0515, BalAcc=0.9104±0.0610
- **RFECA_SVR(k=20)** @ 0.1: F1=0.9181±0.0532, BalAcc=0.9124±0.0614
- **RFECA_SVR(k=20)** @ 0.2: F1=0.8939±0.0565, BalAcc=0.8853±0.0651
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8828±0.0627, BalAcc=0.8702±0.0731
- **RFECA_SVR(k=5)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=5)** @ 0.05: F1=0.9223±0.0479, BalAcc=0.9146±0.0615
- **RFECA_SVR(k=5)** @ 0.1: F1=0.9176±0.0501, BalAcc=0.9120±0.0603
- **RFECA_SVR(k=5)** @ 0.2: F1=0.8990±0.0529, BalAcc=0.8937±0.0614
- **RFECA_SVR(k=5)** @ 0.3: F1=0.8972±0.0696, BalAcc=0.8936±0.0746
- **SimpleMean** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **SimpleMean** @ 0.05: F1=0.9234±0.0493, BalAcc=0.9166±0.0577
- **SimpleMean** @ 0.1: F1=0.9173±0.0586, BalAcc=0.9054±0.0673
- **SimpleMean** @ 0.2: F1=0.8914±0.0590, BalAcc=0.8799±0.0659
- **SimpleMean** @ 0.3: F1=0.8886±0.0694, BalAcc=0.8791±0.0757

## Runtime
- Wall clock: **5663.3s** (94.4 min)
- Completed slots: 50 / 50
