# Discovery full-benchmark report

- **Status:** PASS
- **Run ID:** 20260725_045052
- **Output:** `<repo-root>\artifacts\discovery_full_mar_20260725_045052`

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
- **KNN(k=5,dist)** @ 0.05: RMSE=2.1042±0.3876, MAE=1.5072±0.2607, corr_RV=0.9943±0.0027, corr_Frel=0.1042
- **KNN(k=5,dist)** @ 0.1: RMSE=2.1978±0.3689, MAE=1.5410±0.2281, corr_RV=0.9882±0.0049, corr_Frel=0.1514
- **KNN(k=5,dist)** @ 0.2: RMSE=2.2023±0.2916, MAE=1.5363±0.1917, corr_RV=0.9767±0.0067, corr_Frel=0.2152
- **KNN(k=5,dist)** @ 0.3: RMSE=2.2845±0.2032, MAE=1.5592±0.1448, corr_RV=0.9633±0.0094, corr_Frel=0.2737
- **MissForest** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **MissForest** @ 0.05: RMSE=1.9185±0.4033, MAE=1.3277±0.2682, corr_RV=0.9954±0.0023, corr_Frel=0.0944
- **MissForest** @ 0.1: RMSE=2.0184±0.4385, MAE=1.3569±0.2706, corr_RV=0.9903±0.0045, corr_Frel=0.1375
- **MissForest** @ 0.2: RMSE=2.1149±0.4580, MAE=1.3883±0.2674, corr_RV=0.9819±0.0059, corr_Frel=0.1915
- **MissForest** @ 0.3: RMSE=2.1524±0.3307, MAE=1.4131±0.1924, corr_RV=0.9711±0.0085, corr_Frel=0.2463
- **RFECA_SVR(k=10)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=10)** @ 0.05: RMSE=2.2021±0.5093, MAE=1.5432±0.3224, corr_RV=0.9923±0.0049, corr_Frel=0.1195
- **RFECA_SVR(k=10)** @ 0.1: RMSE=2.4674±0.5242, MAE=1.6792±0.3520, corr_RV=0.9841±0.0071, corr_Frel=0.1747
- **RFECA_SVR(k=10)** @ 0.2: RMSE=2.5330±0.4323, MAE=1.7449±0.2757, corr_RV=0.9663±0.0103, corr_Frel=0.2575
- **RFECA_SVR(k=10)** @ 0.3: RMSE=2.8020±0.3647, MAE=1.9279±0.2271, corr_RV=0.9296±0.0268, corr_Frel=0.3687
- **RFECA_SVR(k=20)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=20)** @ 0.05: RMSE=2.4614±0.5587, MAE=1.7530±0.3799, corr_RV=0.9899±0.0068, corr_Frel=0.1351
- **RFECA_SVR(k=20)** @ 0.1: RMSE=2.7392±0.5618, MAE=1.9143±0.3962, corr_RV=0.9795±0.0101, corr_Frel=0.1966
- **RFECA_SVR(k=20)** @ 0.2: RMSE=2.8753±0.3868, MAE=1.9996±0.2766, corr_RV=0.9533±0.0137, corr_Frel=0.3008
- **RFECA_SVR(k=20)** @ 0.3: RMSE=3.0672±0.3182, MAE=2.1407±0.2101, corr_RV=0.9139±0.0242, corr_Frel=0.4064
- **RFECA_SVR(k=5)** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **RFECA_SVR(k=5)** @ 0.05: RMSE=2.1007±0.4606, MAE=1.4628±0.2792, corr_RV=0.9937±0.0036, corr_Frel=0.1092
- **RFECA_SVR(k=5)** @ 0.1: RMSE=2.2993±0.4764, MAE=1.5535±0.3090, corr_RV=0.9867±0.0057, corr_Frel=0.1602
- **RFECA_SVR(k=5)** @ 0.2: RMSE=2.4281±0.4238, MAE=1.6617±0.2680, corr_RV=0.9718±0.0089, corr_Frel=0.2359
- **RFECA_SVR(k=5)** @ 0.3: RMSE=2.5556±0.3467, MAE=1.7610±0.2239, corr_RV=0.9464±0.0236, corr_Frel=0.3226
- **SimpleMean** @ 0.0: RMSE=nan±nan, MAE=nan±nan, corr_RV=1.0000±0.0000, corr_Frel=0.0000
- **SimpleMean** @ 0.05: RMSE=3.0330±0.6149, MAE=2.2607±0.4304, corr_RV=0.9878±0.0061, corr_Frel=0.1567
- **SimpleMean** @ 0.1: RMSE=2.9894±0.4407, MAE=2.2151±0.2967, corr_RV=0.9767±0.0078, corr_Frel=0.2207
- **SimpleMean** @ 0.2: RMSE=2.9445±0.3828, MAE=2.2008±0.2639, corr_RV=0.9487±0.0117, corr_Frel=0.3331
- **SimpleMean** @ 0.3: RMSE=2.9320±0.2931, MAE=2.1695±0.2159, corr_RV=0.9117±0.0197, corr_Frel=0.4287

## Classification EnsembleSoft (by imputer × rate)
- **KNN(k=5,dist)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **KNN(k=5,dist)** @ 0.05: F1=0.9342±0.0486, BalAcc=0.9246±0.0586
- **KNN(k=5,dist)** @ 0.1: F1=0.9242±0.0500, BalAcc=0.9112±0.0597
- **KNN(k=5,dist)** @ 0.2: F1=0.9284±0.0589, BalAcc=0.9208±0.0669
- **KNN(k=5,dist)** @ 0.3: F1=0.8891±0.0688, BalAcc=0.8766±0.0750
- **MissForest** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **MissForest** @ 0.05: F1=0.9289±0.0461, BalAcc=0.9200±0.0584
- **MissForest** @ 0.1: F1=0.9260±0.0508, BalAcc=0.9203±0.0602
- **MissForest** @ 0.2: F1=0.9268±0.0533, BalAcc=0.9208±0.0647
- **MissForest** @ 0.3: F1=0.8961±0.0644, BalAcc=0.8891±0.0677
- **RFECA_SVR(k=10)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=10)** @ 0.05: F1=0.9269±0.0449, BalAcc=0.9179±0.0568
- **RFECA_SVR(k=10)** @ 0.1: F1=0.9178±0.0531, BalAcc=0.9058±0.0619
- **RFECA_SVR(k=10)** @ 0.2: F1=0.9092±0.0500, BalAcc=0.8967±0.0595
- **RFECA_SVR(k=10)** @ 0.3: F1=0.8802±0.0616, BalAcc=0.8741±0.0652
- **RFECA_SVR(k=20)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=20)** @ 0.05: F1=0.9233±0.0435, BalAcc=0.9142±0.0538
- **RFECA_SVR(k=20)** @ 0.1: F1=0.9193±0.0529, BalAcc=0.9125±0.0616
- **RFECA_SVR(k=20)** @ 0.2: F1=0.8933±0.0646, BalAcc=0.8812±0.0669
- **RFECA_SVR(k=20)** @ 0.3: F1=0.8739±0.0736, BalAcc=0.8658±0.0762
- **RFECA_SVR(k=5)** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **RFECA_SVR(k=5)** @ 0.05: F1=0.9190±0.0426, BalAcc=0.9100±0.0545
- **RFECA_SVR(k=5)** @ 0.1: F1=0.9191±0.0489, BalAcc=0.9091±0.0580
- **RFECA_SVR(k=5)** @ 0.2: F1=0.9044±0.0501, BalAcc=0.8921±0.0615
- **RFECA_SVR(k=5)** @ 0.3: F1=0.8902±0.0536, BalAcc=0.8783±0.0576
- **SimpleMean** @ 0.0: F1=0.9063±0.0349, BalAcc=0.8917±0.0555
- **SimpleMean** @ 0.05: F1=0.9220±0.0506, BalAcc=0.9137±0.0582
- **SimpleMean** @ 0.1: F1=0.9117±0.0559, BalAcc=0.9003±0.0612
- **SimpleMean** @ 0.2: F1=0.8978±0.0533, BalAcc=0.8850±0.0627
- **SimpleMean** @ 0.3: F1=0.8736±0.0545, BalAcc=0.8541±0.0620

## Runtime
- Wall clock: **5610.5s** (93.5 min)
- Completed slots: 50 / 50
