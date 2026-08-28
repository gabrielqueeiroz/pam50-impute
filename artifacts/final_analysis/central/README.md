# Resultados centralizados — comparação final

Métodos incluídos: **Mean · KNN · MissForest · OriginalRFECA**  
Excluídos: RFECA-k5 / k10 / k20 (legado; sem sentido com TARGET-WISE)

Não reexecuta experimentos — apenas agrega artefactos já congelados.

---

## Começar por aqui

| Prioridade | Ficheiro | Uso |
|---:|---|---|
| 1 | `headline_rmse_f1.csv` | Tabela única RMSE + Macro-F1 (mean ± SD) |
| 2 | `wins_summary.csv` | Quem ganha por célula + deltas vs MF/KNN/Mean |
| 3 | `comparison_display.csv` | Versão completa (RMSE/MAE/RV/F1 + rankings) |
| 4 | `imputation_long.csv` | Imputação em formato longo |
| 5 | `classification_long.csv` | Classificação EnsembleSoft em formato longo |
| 6 | `summary.json` | Contagens de vitórias |
| 7 | `caveats.md` | Limitações de comparação entre protocolos |

---

## Panorama (8 células: MCAR/MAR × 5/10/20/30%)

| Métrica | OriginalRFECA | MissForest | Outros |
|---|---:|---:|---|
| Vitórias RMSE (↓) | **7/8** | 1/8 (MAR 5%) | 0 |
| Vitórias Macro-F1 (↑) | **6/8** | 2/8 (MCAR 10%, MAR 10%) | 0 |

---

## Fontes

- Imputação: `../results_tables.csv`
- F1 baselines: `../../paper_results_original_rfeca/comparison/table06_metabric_classification.csv`
- F1 OriginalRFECA: `../../paper_results_original_rfeca/comparison/original_rfeca_classification_summary.csv`

Regenerar: `python scripts/build_central_comparison.py`
