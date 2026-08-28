# Consolidated stats report (MCAR + MAR)

- Generated (UTC): `2026-07-25T16:31:05.718058+00:00`

## Method
- Unit: mean over 5 CV folds within each replicate (n_pairs=10).
- Test: Wilcoxon signed-rank (paired).
- Effect size: matched-pairs rank-biserial r (positive ⇒ method_a > method_b).
- Multiple testing: Holm within each (cohort × mechanism × metric × rate).
- Primary family: MissForest / RFECA(k) vs SimpleMean / KNN.

## Highlights
- METABRIC MCAR F1 primary contrasts with Holm p<0.05: **10** (of 32 tested).
- METABRIC MAR F1 primary contrasts with Holm p<0.05: **12** (of 32 tested).
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
