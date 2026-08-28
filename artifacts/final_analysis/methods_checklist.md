# methods_checklist.md

Checklist de reprodutibilidade — protocolo congelado `v0.3.1-original-rfeca-targetwise`  
e baselines METABRIC full (MCAR/MAR).

Legenda: ✓ presente e verificável nos artefactos | △ parcial / caveat | ✗ ausente

---

## Seeds

- [x] ✓ `base_seed=42` (freeze + configs full)
- [x] ✓ Seed scheme **v2** documentado e `all_seeds_match_v2_formula=true` (freeze)
- [x] ✓ Seeds por slot em `mask.npz` / progress / REPORT
- [x] ✓ Baselines: seed scheme **legacy** nos artefactos principais (documentar explicitamente — não misturar com v2)
- [x] ✓ `random_state=42` StratifiedKFold / KFold / classifiers / MissForest IterativeImputer

## Máscaras

- [x] ✓ Persistência `mask.npz` por slot OriginalRFECA
- [x] ✓ `mask_hash` + `FREEZE/mask_hashes.csv`
- [x] ✓ Shared-mask assert entre imputers (braço baselines)
- [x] ✓ Política `originally_observed_only`
- [x] ✓ Exact-count MCAR/MAR

## Checkpoints

- [x] ✓ Checkpoints por gene / `gene_models.joblib` (TARGET-WISE)
- [x] ✓ Progress / slot DONE flags no freeze
- [x] ✓ Classification artefacts pós-imputação

## Ambiente

- [x] ✓ Python 3.13.5 (manifest freeze)
- [x] ✓ OS Windows 11 (manifest)
- [x] ✓ Pacotes registados no freeze requirements / environment snapshot

## Requirements

- [x] ✓ Requirements do freeze presentes em `FREEZE/`
- [x] ✓ Código em `src/bcimpute/` alinhado ao freeze (não reexecutar)

## Hashes

- [x] ✓ Mask hashes CSV
- [x] ✓ Manifest freeze_id `v0.3.1-original-rfeca-targetwise`
- [x] ✓ Parallel fingerprint-identical (autotune microbenchmark)

## Paralelização determinística

- [x] ✓ `gene_workers=16`, BLAS threads=1
- [x] ✓ Um gene por processo
- [x] ✓ Microbenchmark: speedup com fingerprints idênticos (até 8 workers recomendados no autotune; produção usou 16)

## Versionamento

- [x] ✓ Freeze id + config_snapshot.json
- [x] ✓ Config snapshots baselines `metabric_full_*`
- [x] △ Paper package flowchart ainda marca v0.3.0 no header — usar v0.3.1 como canónico

## Freeze

- [x] ✓ `artifacts/original_rfeca_reduced_metabric/FREEZE/`
- [x] ✓ 40/40 slots class A (MCAR+MAR × 4 rates × 5 reps)
- [x] ✓ `svr_coverage=1.0`, `fallback_rate=0`

## Artefactos

- [x] ✓ REPORT_* OriginalRFECA
- [x] ✓ Classification OriginalRFECA
- [x] ✓ Comparison figures `paper_results_original_rfeca/comparison/`
- [x] ✓ `artifacts/final_analysis/` (tabelas, auditoria, custo)
- [x] ✓ Baselines raw/summary CSVs campanhas full MCAR/MAR

---

## Uso deste checklist ao reescrever Methods

1. Citar freeze_id e caminhos de artefactos.
2. Separar seeds v2 vs legacy.
3. Não afirmar RV para OriginalRFECA.
4. Não afirmar Wilcoxon confirmatório OriginalRFECA vs MissForest.
5. Distinguir protocols A (baselines CV) vs B (TARGET-WISE).
