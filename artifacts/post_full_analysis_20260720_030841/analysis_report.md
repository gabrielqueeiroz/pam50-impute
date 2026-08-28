# Post-full-benchmark analysis

- Generated (UTC): `2026-07-20T03:08:43.397615+00:00`
- Discovery run: `<repo-root>\artifacts\discovery_full_20260719_230651`
- METABRIC run: `<repo-root>\artifacts\metabric_full_20260719_234633`

## Executive findings
- New Discovery EnsembleSoft F1 is on average **-0.013** vs legacy paper (mean abs delta=0.015); drop expected from fold-local correlation + stricter mask policy.
- METABRIC: RFECA reduces RMSE vs SimpleMean by **34.5%** on average across rates (best at 5%: 36.9%).
- Ranking consistency (RMSE vs F1): METABRIC same-winner rates = 3/4; Discovery = 4/4 — better imputation does not always mean better classification.
- METABRIC F1 Wilcoxon (RFECA vs Mean/KNN, p<0.05): **14** significant pairs (see `wilcoxon_pairwise.csv`).
- Stability: Discovery CV(F1) is ~0.057 median vs METABRIC ~0.016 — METABRIC estimates are far tighter.
- Per-subtype F1/precision/recall columns are now implemented in `evaluation.py`; re-run smoke/full to populate `exp2_classification_per_class.csv`.

## Protocol deltas (new vs legacy paper)
- New: fold-local RFECA correlation; `originally_observed_only` mask policy; EnsembleSoft only.
- Legacy paper/Colab: precomputed full-cohort correlation CSV; broader classifier suite in raw archive.
- Absolute F1 levels are therefore **not** expected to match exactly; focus on relative ranking.

## Artifacts
- `new_vs_legacy_discovery.csv`
- `cross_cohort_classification.csv`
- `cross_cohort_imputation.csv`
- `wilcoxon_pairwise.csv`
- `wilcoxon_metabric_rfeca_vs_baselines_sig.csv`
- `ranking_consistency.csv`
- `fold_stability.csv`
- `best_by_rate.csv`
- `metabric_rfeca_rmse_uplift_vs_mean.csv`
