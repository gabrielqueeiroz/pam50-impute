# Freeze v0.3.1-original-rfeca-targetwise

Frozen UTC: `2026-08-03T16:14:14.931076+00:00`

## Protocol
- Method: **OriginalRFECA** TARGET-WISE
- Cohort: METABRIC PAM50 (50 genes)
- Evaluation: `repeated_mask_holdout`
- Predictors: original complete matrix (no SimpleImputer, no chaining)
- Selection: leakage-safe Pearson prefixes + RFE + linear SVR
- `max_candidates=49`, `use_scaler=False`, `inner_cv=5`, `seed_scheme=v2`, `base_seed=42`
- Grid: MCAR+MAR × {0.05, 0.10, 0.20, 0.30} × reps 0–4 (40 slots)

## Results (mean RMSE)
| | 5% | 10% | 20% | 30% |
|--|----|-----|-----|-----|
| MCAR | 0.6126 | 0.6146 | 0.6143 | 0.6213 |
| MAR | 0.6414 | 0.6388 | 0.6397 | 0.6395 |

All 40 slots classification **A**; SVR coverage 1.0; fallbacks 0.

## Files
- `manifest.json` — full freeze record (seeds, mask hashes, config, env)
- `mask_hashes.csv` — one row per slot
- `config_snapshot.json` — protocol knobs
- `requirements.txt` — pinned packages

## Reproduce
```bash
pip install -r requirements-freeze-v0.3.txt
python experiments/run_original_rfeca_targetwise.py --confirm --phase mcar --replicates 0 1 2 3 4 --resume --auto-continue --gene-workers 16 --rates 0.05 0.10 0.20 0.30 --evaluation repeated_mask_holdout
```

Joblib gene models are **not** in git (~1GB); regenerate with the command above
(`--resume` skips completed slots if masks/DONE present).
