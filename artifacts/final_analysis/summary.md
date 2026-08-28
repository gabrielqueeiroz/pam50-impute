# Relatório científico consolidado — OriginalRFECA (dissertação)

Gerado: `2026-08-03T22:33:56.089129+00:00`  
Freeze: **`v0.3.1-original-rfeca-targetwise`**  
Fonte: artefatos existentes (sem reexecução de experimentos).

---

## PARTE 1 — Auditoria final

| Item | Status |
|---|---|
| Experimentos OriginalRFECA METABRIC | **Concluídos** — 40/40 slots DONE, class A |
| Relatórios `REPORT_*_5REPS` | MCAR/MAR × 5/10/20/30% presentes |
| Classificação PAM50 pós-imputação | 40 slots × EnsembleSoft+SVC+LogReg+RF+GB |
| Baselines Mean/KNN/MissForest | `metabric_full_*` (MCAR+MAR, 10 reps) |
| Freeze | `FREEZE/manifest.json` — seeds v2 OK |
| Código alinhado aos resultados | SIM — artefatos batem com `imputation_original/` + runner TARGET-WISE |

**Métodos na comparação principal:** Mean, KNN, MissForest, OriginalRFECA (display “RFECA” em algumas figuras).  
**Excluídos das figuras comparison:** RFECA-k5/k10/k20 (legado).

**Mecanismos:** MCAR, MAR  
**Taxas:** 5%, 10%, 20%, 30%  
**Réplicas OriginalRFECA:** 0–4 (n=5)  
**Réplicas baselines:** 10  
**Seeds OriginalRFECA:** scheme `v2`, `base_seed=42` (lista em freeze)  
**Protocolo OriginalRFECA:** `repeated_mask_holdout` + `target_wise_complete_predictors` + `leakage_safe` + `use_scaler=false` + `max_candidates=49` + 16 gene-workers  
**Protocolo baselines:** shared-mask CV (imputer-within-fold), seed legacy  

**Versão final:** freeze `v0.3.1-original-rfeca-targetwise` (inclui 5%).

---

## Índice para escrita

### Resultados
1. `summary.md` (este ficheiro) — panorama
2. `results_tables.csv` / `results_tables_display.csv` — RMSE/MAE/RV + gaps
3. `original_rfeca_slot_level.csv` — réplicas cruas
4. `stability_by_method_rate.csv` — CV entre réplicas
5. `statistical_analysis.md` + `statistical_rfeca_vs_missforest_descriptive.csv`
6. `rfeca_internal_analysis.md` + CSVs `rfeca_*`
7. `computational_cost.md`
8. `figures/` — figuras regeneradas/copiadas
9. `key_findings.md`
10. `methodology_audit.md`

### Discussão
1. `discussion_points.md`
2. `threats_to_validity.md`
3. `statistical_analysis.md` (limites inferenciais)
4. `rfeca_internal_analysis.md` (porquê seleção/estabilidade)

### Conclusão
1. `conclusion_points.md` (“suportam” / “não permitem”)
2. `key_findings.md`
3. `future_work.md`

---

## Figuras em `figures/`

- `fig_rmse_by_missingness.png` / `.pdf`
- `fig_mae_by_missingness.png` / `.pdf`
- `fig_rv_by_missingness_baselines.png` / `.pdf`
- `fig_rmse_bars_5_10_20_30.png` / `.pdf`
- `fig_stability_rmse_cv.png` / `.pdf`
- `fig_macrof1_by_missingness.png` / `.pdf`
- `copied_comparison_fig01_rmse.png` / `.pdf`
- `copied_comparison_fig03_f1.png` / `.pdf`
- `copied_comparison_fig05_rmse_bars.png` / `.pdf`
- `copied_comparison_fig06_f1_bars.png` / `.pdf`

---

## Estabilidade (resumo)

- CV médio RMSE (réplicas): Mean 0.0105 · OriginalRFECA 0.0124 · MissForest 0.0146 · KNN 0.0161
- **Nota:** Mean tem CV baixo porque o RMSE está saturado (~1.06–1.17); baixa variabilidade ≠ bom método.
- Entre métodos competitivos, **OriginalRFECA** é o mais estável (CV e estabilidade *entre taxas*).
- KNN é o que mais varia (e degrada forte em MAR 30%).
- OriginalRFECA consistente? **SIM** — baixa variação entre taxas; MAR ~0.02–0.03 pior que MCAR de forma estável; 40/40 A.

---

## Ranking RMSE por célula

Ver coluna `ranking_rmse` em `results_tables.csv`.  
Vitórias (1º lugar): {"OriginalRFECA": 7, "MissForest": 1}
