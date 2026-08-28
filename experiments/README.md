# Experiments

Entry points for **full** multi-cohort benchmarks and the principal
**OriginalRFECA TARGET-WISE** METABRIC freeze (`v0.3.1-original-rfeca-targetwise`).

Smoke tests and data preparation live in `../scripts/`.

| Script | Purpose |
|---|---|
| `run_original_rfeca_targetwise.py` | Principal OriginalRFECA TARGET-WISE (mask-holdout). |
| `run_original_rfeca_classification.py` | Post-imputation PAM50 classification on freeze slots. |
| `run_fair_baseline_holdout_vs_rfeca.py` | Mean/KNN/MissForest on the same freeze masks. |
| `run_baseline_gene_metrics.py` | Per-gene baseline metrics helper. |
| `run_metabric_multiclf.py` | Nested multi-classifier campaign (not the freeze). |
| `run_maxcand_sensitivity.py` | `max_candidates` sensitivity (not freeze grid). |
| `orchestrate_reduced_rfeca.py` | Resume MCAR→MAR @20% reduced grid. |
| `orchestrate_expanded_rates.py` | After 20%: additional rates. |
| `consolidate_reduced_rfeca.py` | Consolidate reduced reports. |
| `run_full_discovery.py` | Full Discovery protocol (opt-in). |
| `run_full_metabric.py` | Full METABRIC six-imputer CV protocol (opt-in). |
| `cli_common.py` | Shared flags: `--only-rfeca`, `--seed-scheme`, `--mechanism`. |

Freeze bundle: `artifacts/original_rfeca_reduced_metabric/FREEZE/` + `requirements-freeze-v0.3.txt`.

## Flags

```bash
# Dry-run (prints config, does not start)
python experiments/run_full_discovery.py --only-rfeca --seed-scheme v2

# Launch RFECA-only (inductive RFECA; collision-free seeds)
python experiments/run_full_discovery.py --only-rfeca --seed-scheme v2 --confirm
python experiments/run_full_discovery.py --only-rfeca --mechanism mar --seed-scheme v2 --confirm
python experiments/run_full_metabric.py --only-rfeca --seed-scheme v2 --confirm
python experiments/run_full_metabric.py --only-rfeca --mechanism mar --seed-scheme v2 --confirm
```

- `--only-rfeca`: imputers = RFECA(k=5/10/20) only; artifact prefix `*_full_rfeca_*`.
- `--seed-scheme v2` (default): collision-free seeds. **Masks differ** from 2026-07 frozen runs — do not merge those CSVs.
- `--seed-scheme legacy`: reproduce old masks (only if you must merge with Mean/KNN/MissForest from those artifacts).

RFECA in `run_full_*` is the **legacy inductive k-fixed** path. The principal method is TARGET-WISE OriginalRFECA (`run_original_rfeca_targetwise.py`).

Post-run analysis: `../scripts/run_stats_posthoc.py`, `../scripts/analyze_full_benchmark_results.py`.
