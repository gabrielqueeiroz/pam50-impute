# Comparison + RFECA stats package

Generated: `2026-08-03T16:14:32.183031+00:00`

## Layout
- `comparison/` — METABRIC figures with **Mean, KNN, MissForest, RFECA**
  (RFECA = OriginalRFECA TARGET-WISE; **RFECA-k5/k10/k20 excluded**)
- `stats/` — Wilcoxon/Holm/Friedman on the six-imputer campaign (legacy RFECA_SVR(k=*))

## Comparison methods
| Display name | Source |
|---|---|
| Mean, KNN, MissForest | Shared-mask CV METABRIC (10 reps); F1 = EnsembleSoft imputer-within-CV |
| **RFECA** | OriginalRFECA TARGET-WISE (mask-holdout, 5 reps); F1 = post-impute identity CV |

## Key figures (`comparison/`)
- `fig01_metabric_rmse_by_missingness` — RMSE lines including **RFECA** (5/10/20/30%)
- `fig02_metabric_rv_by_missingness` — RV (baselines only; no RFECA-k*)
- `fig03_metabric_macrof1_by_missingness` — Macro-F1 including **RFECA**
- `fig04_rmse_vs_macrof1` — baselines + RFECA
- `fig05_metabric_rmse_bars_5_10_20_30` — RMSE bars with **RFECA**
- `fig06_metabric_macrof1_bars_5_10_20_30` — F1 bars with **RFECA**

## Protocol note
RFECA classification uses already-imputed matrices (identity imputer in StratifiedKFold).
Baselines keep imputer-within-CV from the METABRIC full campaign.

## Stats summary
```json
{
  "n_rfeca_pairwise": 288,
  "n_rfeca_primary": 144,
  "n_rfeca_vs_mean": 72,
  "n_rfeca_vs_mean_holm05": 58,
  "n_k20_vs_mean_rows": 24
}
```

Copied non-figure paper artifacts: 9 files.
