# OriginalRFECA TARGET-WISE — figure/table captions

Protocol: METABRIC PAM50; TARGET-WISE complete predictors; `repeated_mask_holdout`;
leakage-safe Pearson prefixes + RFE + linear SVR; `max_candidates=49`; `use_scaler=false`;
`seed_scheme=v2`; `base_seed=42`; 5 replications; rates 10/20/30%; freeze `v0.3.0-original-rfeca-targetwise`.

Rounding: RMSE/MAE displayed to **3 decimals** (half-up). Full precision in `*_full.csv`.

## Figures
- **Fig. 1** `fig01_rmse_by_missingness` — mean±SD RMSE vs missingness (MCAR/MAR).
- **Fig. 2** `fig02_mae_by_missingness` — mean±SD MAE vs missingness.
- **Fig. 3** `fig03_rmse_boxplot_by_rate` — replicate RMSE spread.
- **Fig. 4** `fig04_per_gene_rmse_mcar20` — per-gene mean RMSE at MCAR 20%.
- **Fig. 5** `fig05_delta_mar_minus_mcar` — MAR−MCAR RMSE gap.

## Tables
- `table_original_rfeca_metabric.tex` / `table_paper_rmse_mae_display.csv` — main RMSE/MAE grid.
- `slot_level_full.csv` — 30 slots (seeds, mask hashes, metrics).
- `summary_by_mech_rate_{full,display}.csv` — aggregated means.
- `per_gene_rmse_by_mech_rate_full.csv` — gene-level means.
- `table_top10_genes_mcar20_lowest_rmse.csv` / `table_bottom10_genes_mcar20_highest_rmse.csv`.
