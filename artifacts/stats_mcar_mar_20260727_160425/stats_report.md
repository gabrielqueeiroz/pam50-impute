# Consolidated stats report (MCAR + MAR)

- Generated (UTC): `2026-07-27T16:05:20.601200+00:00`

## Method
- Unit: mean over 5 CV folds within each replicate (n_blocks=n_pairs=10).
- Omnibus: Friedman chi-square across imputers (blocks = replicates).
- Post-hoc: Wilcoxon signed-rank (paired) + matched-pairs rank-biserial.
- Effect CI: percentile bootstrap 95% for mean paired delta (n_boot=5000, replicate-level).
- Multiple testing: Holm within each (cohort × mechanism × metric × rate).
- Primary family: MissForest / RFECA(k) vs SimpleMean / KNN.
- Friedman tables: (a) all 6 imputers; (b) reduced set Mean/KNN/RFECA(k=20)/MissForest.

## Highlights
- METABRIC MCAR F1 primary contrasts with Holm p<0.05: **10** (of 32 tested).
- METABRIC MAR F1 primary contrasts with Holm p<0.05: **12** (of 32 tested).
- Friedman reduced | METABRIC MAR f1_macro @ 5%: chi2=0.840, p=0.8399, n_blocks=10.
- Friedman reduced | METABRIC MAR f1_macro @ 10%: chi2=10.200, p=0.01694, n_blocks=10.
- Friedman reduced | METABRIC MAR f1_macro @ 20%: chi2=16.680, p=0.0008223, n_blocks=10.
- Friedman reduced | METABRIC MAR f1_macro @ 30%: chi2=18.360, p=0.0003707, n_blocks=10.
- Friedman reduced | METABRIC MCAR f1_macro @ 5%: chi2=6.000, p=0.1116, n_blocks=10.
- Friedman reduced | METABRIC MCAR f1_macro @ 10%: chi2=9.000, p=0.02929, n_blocks=10.
- Friedman reduced | METABRIC MCAR f1_macro @ 20%: chi2=21.360, p=8.862e-05, n_blocks=10.
- Friedman reduced | METABRIC MCAR f1_macro @ 30%: chi2=15.000, p=0.001817, n_blocks=10.
- Friedman reduced | METABRIC MAR rmse @ 5%: chi2=27.120, p=5.556e-06, n_blocks=10.
- Friedman reduced | METABRIC MAR rmse @ 10%: chi2=28.920, p=2.328e-06, n_blocks=10.
- Friedman reduced | METABRIC MAR rmse @ 20%: chi2=28.920, p=2.328e-06, n_blocks=10.
- Friedman reduced | METABRIC MAR rmse @ 30%: chi2=30.000, p=1.38e-06, n_blocks=10.
- Friedman reduced | METABRIC MCAR rmse @ 5%: chi2=27.480, p=4.669e-06, n_blocks=10.
- Friedman reduced | METABRIC MCAR rmse @ 10%: chi2=28.080, p=3.494e-06, n_blocks=10.
- Friedman reduced | METABRIC MCAR rmse @ 20%: chi2=27.480, p=4.669e-06, n_blocks=10.
- Friedman reduced | METABRIC MCAR rmse @ 30%: chi2=28.080, p=3.494e-06, n_blocks=10.
- ΔRMSE MissForest vs Mean | MAR @ 5%: -0.5398 [-0.5504, -0.5297] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MAR @ 10%: -0.5174 [-0.5226, -0.5124] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MAR @ 20%: -0.4770 [-0.4821, -0.4725] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MAR @ 30%: -0.4377 [-0.4415, -0.4335] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MCAR @ 5%: -0.4495 [-0.4603, -0.4378] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MCAR @ 10%: -0.4397 [-0.4499, -0.4302] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MCAR @ 20%: -0.4304 [-0.4350, -0.4257] (bootstrap CI95, n=10).
- ΔRMSE MissForest vs Mean | MCAR @ 30%: -0.4156 [-0.4189, -0.4118] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MAR @ 5%: -0.4686 [-0.4759, -0.4608] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MAR @ 10%: -0.4430 [-0.4482, -0.4379] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MAR @ 20%: -0.3818 [-0.3860, -0.3775] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MAR @ 30%: -0.3274 [-0.3303, -0.3244] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MCAR @ 5%: -0.3933 [-0.4025, -0.3833] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MCAR @ 10%: -0.3774 [-0.3859, -0.3703] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MCAR @ 20%: -0.3616 [-0.3644, -0.3586] (bootstrap CI95, n=10).
- ΔRMSE RFECA_SVR(k=20) vs Mean | MCAR @ 30%: -0.3327 [-0.3366, -0.3291] (bootstrap CI95, n=10).
- MissForest vs Mean | MAR @ 5%: ΔF1=-0.0001, r_rb=-0.127, p=0.7695, p_Holm=1.
- MissForest vs Mean | MAR @ 10%: ΔF1=+0.0048, r_rb=0.782, p=0.02734, p_Holm=0.2188.
- MissForest vs Mean | MAR @ 20%: ΔF1=+0.0072, r_rb=0.964, p=0.003906, p_Holm=0.02344.
- MissForest vs Mean | MAR @ 30%: ΔF1=+0.0126, r_rb=1.000, p=0.001953, p_Holm=0.01562.
- MissForest vs Mean | MCAR @ 5%: ΔF1=+0.0025, r_rb=0.527, p=0.1602, p_Holm=0.6543.
- MissForest vs Mean | MCAR @ 10%: ΔF1=+0.0039, r_rb=0.709, p=0.04883, p_Holm=0.2441.
- MissForest vs Mean | MCAR @ 20%: ΔF1=+0.0081, r_rb=1.000, p=0.001953, p_Holm=0.01562.
- MissForest vs Mean | MCAR @ 30%: ΔF1=+0.0111, r_rb=1.000, p=0.001953, p_Holm=0.01562.

## Artifacts
- `descriptives_by_imputer.csv`
- `pairwise_all.csv`
- `primary_contrasts.csv`
- `primary_metabric_f1_sig_holm05.csv`
- `friedman_all_imputers.csv`
- `friedman_reduced_imputers.csv`
- `primary_metabric_rmse_vs_mean.csv`
