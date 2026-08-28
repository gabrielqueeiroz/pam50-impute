# Baseline gene-level metrics — prepared run

## Scenario
- **Methods:** Mean (`SimpleMean`), KNN(k=5,dist), MissForest
- **Mechanisms:** MCAR + MAR
- **Rate:** 20%
- **Replicates:** 0–4
- **Seeds:** `legacy` (frozen `metabric_full_*`)
  - MCAR 20%: 1062–1066
  - MAR 20%: 18062–18066
- **CV:** StratifiedKFold(5, shuffle=True, `random_state=42`)
- **Imputation-only** (no classification)
- **Export:** RMSE/MAE per gene on masked test-fold cells

## Commands

```bash
# Audit seeds / write PLAN only
python experiments/run_baseline_gene_metrics.py --dry-run

# Launch (≈5–15 min wall with --workers 5 on this machine)
python experiments/run_baseline_gene_metrics.py --confirm --workers 5 --missforest-n-jobs 2
```

## Outputs
`artifacts/baseline_gene_metrics_<timestamp>/`
- `per_gene_summary.csv`, `gene_method_rmse_wide.csv`
- `masks/` + `seed_audit.csv`
- `sanity_vs_protocol_means.csv` (global RMSE check)
