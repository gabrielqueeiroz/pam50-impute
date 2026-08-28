# Índice rápido — o que abrir para escrever

## Results hub (comparação final — sem k5/k10/k20)
| Ordem | Ficheiro | Uso |
|---:|---|---|
| ★ | `central/README.md` | Índice do hub |
| ★ | `central/headline_rmse_f1.csv` | RMSE + Macro-F1 lado a lado |
| ★ | `central/wins_summary.csv` | Vitórias e deltas vs MF/KNN/Mean |
| ★ | `central/comparison_display.csv` | Tabela completa para análise |
| ★ | `central/caveats.md` | Limites de comparação entre protocolos |
| ★ | `one_page_results.md` | Resumo 1 página para Results do artigo |
| ★ | `stats_final/README.md` | Stats A (válidas) vs B (descritivas) |

Métodos: Mean · KNN · MissForest · OriginalRFECA.  
Excluídos: RFECA-k*. Regenerar: `python scripts/build_central_comparison.py` · `python scripts/build_final_stats_package.py`

## Writing enrichment (derived)
| Ordem | Ficheiro | Uso |
|---:|---|---|
| ★ | `one_page_results.md` | Results one-pager |
| ★ | `gene_difficulty_report.md` | Dificuldade por gene |
| ★ | `predictor_gene_biological_notes.md` | Notas biológicas top preditores |
| ★ | `limitations_vs_strengths.md` | Strengths / Limitations |
| ★ | `reviewer_results_questions.md` | Q&A de revisor (resultados) |
| ★ | `figures_inventory.md` | Inventário de figuras |
| ★ | `article_readiness.md` | Checklist de prontidão |
| ★ | `figures/gene_method_heatmap_rmse.png` | Heatmap gene×método |
| ★ | `gene_method_heatmap.csv` | Dados do heatmap (NaNs explícitos) |
| ★ | `fair_imputation_comparison.md` | Comparação justa holdout (máscaras freeze) |
| ★ | `fair_imputation_comparison_display.csv` | Tabela RMSE/MAE pareada |
| ★ | `figures/gene_method_heatmap_rmse_fair.png` | Heatmap 4 métodos (fair) |
| ★ | `fair_per_gene_winners.csv` | Vitórias por gene |

## Methods (single source of truth — protocolo executado)
| Ordem | Ficheiro | Uso |
|---:|---|---|
| 0 | `methods_freeze.md` | **Fonte canónica** para reescrever Methods |
| 0b | `experimental_protocols.md` | Protocolos A (baselines) vs B (TARGET-WISE) |
| 0c | `hyperparameters_table.csv` | Todos os hiperparâmetros efetivos |
| 0d | `methods_checklist.md` | Checklist de reprodutibilidade |
| 0e | `methods_vs_article.md` | Plano de atualização vs Methods do paper package |

## Documentos-base (dissertação / artigo)
| Ordem | Ficheiro | Uso |
|---:|---|---|
| 1 | `experiment_story.md` | História científica / hipótese / contribuições |
| 2 | `results_narrative.md` | Roteiro da seção Resultados (figura a figura) |
| 3 | `discussion_outline.md` | Roteiro da Discussão (tópicos) |
| 4 | `contribution_matrix.md` | Contribuição × evidência × defesa |
| 5 | `reviewer_questions.md` | Q&A de revisor (≥40 perguntas) |

## Resultados (evidência numérica)
| Ordem | Ficheiro | Conteúdo |
|---:|---|---|
| 1 | `summary.md` | Auditoria + panorama |
| 2 | `results_tables.csv` | Tabela mestra RMSE/MAE/RV/gaps |
| 3 | `results_tables_display.csv` | Versão para colar em texto |
| 4 | `figures/fig_rmse_by_missingness.png` | Figura principal RMSE |
| 5 | `figures/fig_mae_by_missingness.png` | MAE |
| 6 | `figures/fig_macrof1_by_missingness.png` | F1 |
| 7 | `stability_by_method_rate.csv` | Estabilidade |
| 8 | `statistical_analysis.md` | Significância (limites) |
| 9 | `rfeca_internal_analysis.md` | Seleção de features |
| 10 | `computational_cost.md` | Tempos/paralelismo |
| 11 | `methodology_audit.md` | Checklist leakage |
| 12 | `key_findings.md` | 10 respostas objetivas |

## Discussão
| Ordem | Ficheiro |
|---:|---|
| 1 | `discussion_outline.md` |
| 2 | `discussion_points.md` |
| 3 | `threats_to_validity.md` |
| 4 | `statistical_analysis.md` |
| 5 | `rfeca_internal_analysis.md` |

## Conclusão
| Ordem | Ficheiro |
|---:|---|
| 1 | `conclusion_points.md` |
| 2 | `key_findings.md` |
| 3 | `contribution_matrix.md` |
| 4 | `future_work.md` |
