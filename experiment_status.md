# Experiment status

**Date:** 2026-08-03  
**Status:** OriginalRFECA TARGET-WISE METABRIC freeze **PASS** → `v0.3.1-original-rfeca-targetwise`

## Frozen: OriginalRFECA TARGET-WISE (principal method)

- Protocol: target-wise complete predictors, `repeated_mask_holdout`, leakage-safe RFE+SVR
- Grid: MCAR + MAR × {5%, 10%, 20%, 30%} × 5 replicates (40 slots)
- Config: `max_candidates=49`, `use_scaler=False`, `seed_scheme=v2`, `base_seed=42`, `gene_workers=16`
- Artifacts: `artifacts/original_rfeca_reduced_metabric/` (+ `FREEZE/` manifest, mask hashes, pinned requirements)
- Pin file: `requirements-freeze-v0.3.txt`

Mean RMSE from `FREEZE/README.md` (slot reports):

| mean RMSE | 5% | 10% | 20% | 30% |
|-----------|-----|-----|-----|-----|
| MCAR | 0.6126 | 0.6146 | 0.6143 | 0.6213 |
| MAR | 0.6414 | 0.6388 | 0.6397 | 0.6395 |

Fair paired comparison vs Mean/KNN/MissForest (same masks):
`artifacts/final_analysis/fair_imputation_comparison_display.csv`.

## Prior freezes

- `v0.3.0-original-rfeca-targetwise` — same method, rates 10/20/30% only (superseded)
- `v0.2.0-full-benchmarks` — Discovery + METABRIC multi-imputer fulls
- `v0.1.0-discovery-ready` — discovery-ready pipeline
