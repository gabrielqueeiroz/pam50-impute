# Artifacts

Executed experiment outputs. **Do not treat every directory as the current
manuscript method.**

Canonical freeze: `original_rfeca_reduced_metabric/FREEZE/`
(`v0.3.1-original-rfeca-targetwise`).

Writing / comparison hub: `final_analysis/`
(index: `final_analysis/INDEX.md`).

## Classification of trees

| Directory | Class | Notes |
|---|---|---|
| `original_rfeca_reduced_metabric/` | **A — current methodology** | TARGET-WISE freeze; slot summaries + masks; parquet gene models gitignored |
| `original_rfeca_reduced_metabric/FREEZE/` | **A** | Manifest, mask hashes, pinned requirements |
| `fair_baseline_holdout_vs_rfeca/` | **A / supporting** | Mean/KNN/MissForest on the **same freeze masks** |
| `final_analysis/` | **A / supporting** | Tables, figures, methods SSoT for writing |
| `paper_results_original_rfeca/` | **A / supporting** | TARGET-WISE paper figures/tables (some captions still say v0.3.0) |
| `paper_results/` | **C — historical / baseline arm** | Six-imputer CV campaign figures (legacy RFECA-k* included in some tables) |
| `metabric_full_*` / `discovery_full_*` | **C** | Full CV campaigns; later dates are MAR / RFECA-only variants |
| `original_rfeca_reduced/` | **C** | Earlier reduced run (not the METABRIC 40-slot freeze) |
| `*_smoke_*` | **E — local / validation** | Reduced protocol; not freeze metrics |
| `post_full_analysis_*` / `stats_*` | **B — supporting analysis** | Post-hoc stats on campaigns |
| `discovery_cptac_provenance_audit/` | **B** | Provenance notes; **sample-map CSVs gitignored** |
| `parallel_benchmark/` | **B** | Worker-count timing (not scientific metrics) |
| `archive/` (repo root) | **C** | Colab / conference-era materials |
| `profiling/`, `*_leakage_safe_*`, `original_rfeca_rfaca/` runs | **E** | Local intermediates; gitignored except the RFACA README |
| `metabric_multiclf_*` | **D / E** | Large nested-classifier dumps; aggregates copied into `paper_results/` |

A = current methodology · B = supporting · C = historical · D = generated
aggregates (keep if safe) · E = local/temporary · F = restricted (gitignored).

## Intentionally not in Git

- `*.parquet` reconstructed matrices and `*.joblib` gene models
- `imputer_checkpoint/` / `checkpoints/` trees
- Clinical `.tsi`, full transcriptome `legacy/cptac.csv`
- Recovered CPTAC ID maps (see `data/README.md`)
