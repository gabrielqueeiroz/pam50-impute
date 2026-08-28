# Consolidated stats report (MCAR + MAR)

- Generated (UTC): `2026-07-27T15:58:57.126676+00:00`

## Method
- Unit: mean over 5 CV folds within each replicate (n_blocks=n_pairs=10).
- Omnibus: Friedman chi-square across imputers (blocks = replicates).
- Post-hoc: Wilcoxon signed-rank (paired) + matched-pairs rank-biserial.
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
