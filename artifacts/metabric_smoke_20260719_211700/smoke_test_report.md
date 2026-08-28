# METABRIC smoke-test report

- **Status:** PASS
- **Technically ready for full METABRIC benchmark:** True
- **Run ID:** 20260719_211700
- **Output:** `<repo-root>\artifacts\metabric_smoke_20260719_211700`

## Cohort
- Final n: **1608**
- Class distribution: `{'LumA': 700, 'LumB': 475, 'Her2': 224, 'Basal': 209}`
- Genes: 50

## Assertions
- `mask_feature_only`: **PASS**
- `complete_matrix_not_mutated`: **PASS**
- `shared_masks_across_imputers`: **PASS**
- `metrics_on_masked_cells_only`: **PASS**
- `train_val_no_overlap`: **PASS**
- `no_shared_imputer_or_scaler_or_corr_across_folds`: **PASS**
- `rfeca_training_fold_correlation_only`: **PASS**

## Masked cells
- rate=0.1 rep=0 KNN(k=5,dist): n_masked_test_values=8038
- rate=0.1 rep=0 RFECA_SVR(k=5): n_masked_test_values=8038
- rate=0.1 rep=0 SimpleMean: n_masked_test_values=8038
- rate=0.1 rep=1 KNN(k=5,dist): n_masked_test_values=7857
- rate=0.1 rep=1 RFECA_SVR(k=5): n_masked_test_values=7857
- rate=0.1 rep=1 SimpleMean: n_masked_test_values=7857

## Imputation (RMSE / MAE)
- **KNN(k=5,dist)**: RMSE=0.6957±0.0329, MAE=0.4736±0.0155
- **RFECA_SVR(k=5)**: RMSE=0.7055±0.0375, MAE=0.4684±0.0190
- **SimpleMean**: RMSE=1.0708±0.0238, MAE=0.7537±0.0183

## Classification (primary model)
- **KNN(k=5,dist)** / EnsembleSoft: macro-F1=0.8783±0.0069, BalAcc=0.8673±0.0107
- **RFECA_SVR(k=5)** / EnsembleSoft: macro-F1=0.8754±0.0037, BalAcc=0.8643±0.0060
- **SimpleMean** / EnsembleSoft: macro-F1=0.8731±0.0069, BalAcc=0.8625±0.0064

## Runtime
- Wall clock: **139.43s**
- imputation_rate0.1_rep0: 3.50s
- classification_rate0.1_rep0: 68.84s
- imputation_rate0.1_rep1: 2.94s
- classification_rate0.1_rep1: 64.12s

## Fold class distributions (sample)
- rep=0 fold=1 train={'Basal': 140, 'Her2': 149, 'LumA': 466, 'LumB': 317} test={'Basal': 69, 'Her2': 75, 'LumA': 234, 'LumB': 158}
- rep=0 fold=2 train={'Basal': 139, 'Her2': 150, 'LumA': 467, 'LumB': 316} test={'Basal': 70, 'Her2': 74, 'LumA': 233, 'LumB': 159}
- rep=0 fold=3 train={'Basal': 139, 'Her2': 149, 'LumA': 467, 'LumB': 317} test={'Basal': 70, 'Her2': 75, 'LumA': 233, 'LumB': 158}
- rep=0 fold=1 train={'Basal': 140, 'Her2': 149, 'LumA': 466, 'LumB': 317} test={'Basal': 69, 'Her2': 75, 'LumA': 234, 'LumB': 158}
- rep=0 fold=2 train={'Basal': 139, 'Her2': 150, 'LumA': 467, 'LumB': 316} test={'Basal': 70, 'Her2': 74, 'LumA': 233, 'LumB': 159}
- rep=0 fold=3 train={'Basal': 139, 'Her2': 149, 'LumA': 467, 'LumB': 317} test={'Basal': 70, 'Her2': 75, 'LumA': 233, 'LumB': 158}
- rep=0 fold=1 train={'Basal': 140, 'Her2': 149, 'LumA': 466, 'LumB': 317} test={'Basal': 69, 'Her2': 75, 'LumA': 234, 'LumB': 158}
- rep=0 fold=2 train={'Basal': 139, 'Her2': 150, 'LumA': 467, 'LumB': 316} test={'Basal': 70, 'Her2': 74, 'LumA': 233, 'LumB': 159}
- rep=0 fold=3 train={'Basal': 139, 'Her2': 149, 'LumA': 467, 'LumB': 317} test={'Basal': 70, 'Her2': 75, 'LumA': 233, 'LumB': 158}
- rep=1 fold=1 train={'Basal': 140, 'Her2': 149, 'LumA': 466, 'LumB': 317} test={'Basal': 69, 'Her2': 75, 'LumA': 234, 'LumB': 158}
- rep=1 fold=2 train={'Basal': 139, 'Her2': 150, 'LumA': 467, 'LumB': 316} test={'Basal': 70, 'Her2': 74, 'LumA': 233, 'LumB': 159}
- rep=1 fold=3 train={'Basal': 139, 'Her2': 149, 'LumA': 467, 'LumB': 317} test={'Basal': 70, 'Her2': 75, 'LumA': 233, 'LumB': 158}

## Warnings / degenerate predictions
- Warning count: 0
- Degenerate imputation folds: 0
- Degenerate classification folds: 0
- None recorded.

## Readiness notes
- Smoke protocol completed with leakage-safety assertions PASS.
- RFECA correlations computed from training-fold X.corr() only.
- StandardScaler fitted inside CV folds via sklearn Pipeline clone.
- Full benchmark will use config.full_benchmark_config('metabric') (rates 0-30%, 10 reps, 5 folds) and will be substantially slower.
- Smoke wall time: 139.4s on n=1608.

**Stopped after smoke test — full METABRIC benchmark not started.**