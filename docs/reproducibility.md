# Reproducibility

Freeze: **`v0.3.1-original-rfeca-targetwise`**.  
Pin file: `requirements-freeze-v0.3.txt`.  
Manifest: `artifacts/original_rfeca_reduced_metabric/FREEZE/`.

The full OriginalRFECA TARGET-WISE grid is expensive (gene-level RFE+SVR × 50 genes × 40 slots). Do **not** treat a laptop smoke run as a reproduction of the freeze.

## Environment

```powershell
python -m pip install -r requirements.txt
```

For bit-for-bit alignment with the freeze runtime, use `requirements-freeze-v0.3.txt` (the freeze was recorded on Python 3.13.5 / Windows).

Add `src/` to `PYTHONPATH` or insert it as the experiment scripts do.

## Data

See `data/README.md`. The METABRIC 50-gene matrix used by the freeze is
**not shipped** in this public snapshot. After `python scripts\prepare_metabric.py`
it should be at `data/processed/metabric/metabric_pam50_4class.csv`.

Validate that matrix (does not reload the 657 MB microarray):

```powershell
python scripts\test_prepare_metabric.py
```

If you need to rebuild it from a cBioPortal download:

```powershell
python scripts\prepare_metabric.py
python scripts\test_prepare_metabric.py
```

## Lightweight checks (no full benchmark)

```powershell
python experiments\run_original_rfeca_targetwise.py --help
python experiments\run_full_metabric.py --help
python scripts\test_mask_holdout_leakage.py
python scripts\test_rfeca_inductive_and_seeds.py
python scripts\test_original_rfeca_rfaca.py
```

Discovery tests require the local processed discovery matrix (gitignored; create with `python scripts\prepare_discovery.py`). `scripts\test_prepare_discovery.py` currently expects synthetic `discovery_row_*` IDs; a later local prepare that recovered CPTAC Patient_IDs will fail that ID-format check. Values/fingerprints are unchanged.

Smoke tests (`scripts\run_smoke_metabric.py`, `scripts\run_smoke_discovery.py`) run a **reduced** protocol. They do not reproduce freeze metrics.

## Reproduce the freeze (opt-in, hours)

```powershell
pip install -r requirements-freeze-v0.3.txt
python experiments\run_original_rfeca_targetwise.py --confirm --phase mcar --replicates 0 1 2 3 4 --resume --auto-continue --gene-workers 16 --rates 0.05 0.10 0.20 0.30 --evaluation repeated_mask_holdout
```

Then the MAR phase with `--phase mar` (same flags). `--resume` skips completed slots when masks/`DONE.json` are present.

Joblib gene models are not in Git (~1 GB). Mask hashes for completed slots are in `FREEZE/mask_hashes.csv`.

Rebuild the freeze bundle from completed slots:

```powershell
python scripts\build_original_rfeca_freeze.py
```

## Fair baseline RMSE (same masks)

```powershell
python experiments\run_fair_baseline_holdout_vs_rfeca.py --help
```

This script is opt-in and reuses freeze masks. Aggregates already produced:
`artifacts/fair_baseline_holdout_vs_rfeca/fair_imputation_comparison.md`
and `artifacts/final_analysis/fair_imputation_comparison_display.csv`.

## Seeds

| Item | Value |
|---|---|
| Missingness `base_seed` | 42 |
| Seed scheme | v2 |
| Classification CV | StratifiedKFold shuffle, `random_state=42` |
| OriginalRFECA inner KFold | `n_splits=5`, `random_state=42` |

v2 missingness seeds are **not** interchangeable with the legacy seed scheme used in some 2026-07 six-imputer artifacts. Do not merge those CSVs.

## Known documentation mismatches (do not “fix” silently)

- `experiments/run_original_rfeca_targetwise.py` still contains a `METHOD_JUSTIFICATION` string describing a 20%-only reduced analysis. The executed freeze is 5/10/20/30% × 40 slots (`FREEZE/README.md`).
- `paper_results_original_rfeca/` captions sometimes say freeze **v0.3.0**; the canonical freeze id is **v0.3.1**.
