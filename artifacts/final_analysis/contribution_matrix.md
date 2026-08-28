# contribution_matrix.md

Matriz contribuição → evidência → artefacto → defesa.

---

## C1 — Protocolo OriginalRFECA TARGET-WISE leakage-safe

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Definição e implementação de imputação TARGET-WISE com seleção `leakage_safe`, sem SimpleImputer/chaining |
| **Evidência experimental** | 40/40 slots: `leakage_or_protocol_fail=false`, `svr_coverage=1.0`, `fallback_rate=0`, `n_predictor_nans=0` |
| **Arquivo** | `artifacts/original_rfeca_reduced_metabric/FREEZE/manifest.json`; `*/DONE.json`; `methodology_audit.md` |
| **Figura** | flowchart OriginalRFECA (pacote paper_results_original_rfeca) |
| **Tabela** | checklist em `methodology_audit.md` |
| **Importância** | Alta — condição de validade de todos os RMSE |
| **Risco de crítica** | “TARGET-WISE não é imputação multivariada real” |
| **Defesa** | Concordar no escopo: é imputação condicional por gene com preditores completos; vantagem é anti-leakage/anti-chaining; comparar honestamente com MissForest multivariado |

---

## C2 — RMSE competitivo/superior vs MissForest no METABRIC

| Campo | Conteúdo |
|---|---|
| **Contribuição** | OriginalRFECA com menor RMSE médio em 7/8 células mecanismo×taxa |
| **Evidência** | Rankings em `results_tables.csv`; Δ RFECA−MF de −0.005 a −0.066 (exceto MAR 5% +0.012) |
| **Arquivo** | `results_tables.csv`, `results_tables_display.csv`, `original_rfeca_slot_level.csv` |
| **Figura** | `figures/fig_rmse_by_missingness.png`, `fig_rmse_bars_5_10_20_30.png` |
| **Tabela** | `results_tables_display.csv` |
| **Importância** | Alta — resultado principal de desempenho |
| **Risco de crítica** | Protocolos diferentes (5 vs 10 reps; holdout vs CV; v2 vs legacy) |
| **Defesa** | Declarar caveat; apresentar como evidência descritiva convergente; não reivindicar Wilcoxon pareado; oferecer redesenho paired como trabalho futuro |

---

## C3 — Estabilidade do RMSE entre taxas

| Campo | Conteúdo |
|---|---|
| **Contribuição** | RMSE OriginalRFECA quase invariante entre 5–30% (esp. MAR) |
| **Evidência** | Span MCAR ≈ 0.009; MAR ≈ 0.003 (`discussion_points.md` / slot means) |
| **Arquivo** | `original_rfeca_slot_level.csv`, `results_tables.csv` |
| **Figura** | `fig_rmse_by_missingness.png` (linhas planas RFECA) |
| **Tabela** | médias por taxa em `results_tables_display.csv` |
| **Importância** | Alta — diferencia o método dos baselines |
| **Risco de crítica** | “Plano porque já saturado” |
| **Defesa** | Comparar com MissForest/KNN que sobem no mesmo eixo; saturado seria também gaps nulos vs MF — não é o caso a 20–30% |

---

## C4 — Robustez relativa sob MAR

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Em MAR, vantagem vs KNN/MissForest aumenta com a taxa |
| **Evidência** | MAR 30%: RFECA 0.639 vs MF 0.705 vs KNN 0.857 |
| **Arquivo** | `results_tables.csv` |
| **Figura** | `fig_rmse_by_missingness.png` (painel MAR), barras MAR 30% |
| **Tabela** | linha MAR 30% em display |
| **Importância** | Alta para relevância prática |
| **Risco de crítica** | MAR simulado artificial |
| **Defesa** | Padrão da literatura de simulação; declarar limite MNAR; mostrar MCAR+MAR consistentemente |

---

## C5 — Separação clara vs Mean

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Mean é baseline inferior (~40%+ RMSE relativo maior) |
| **Evidência** | Mean RMSE 1.06–1.17 vs RFECA ~0.61–0.64 |
| **Arquivo** | `results_tables.csv` |
| **Figura** | todas as figs RMSE |
| **Tabela** | display |
| **Importância** | Média — sanity check |
| **Risco de crítica** | Baseline demasiado fraca |
| **Defesa** | Mean é controlo negativo padrão; o contraste forte é vs MissForest/KNN |

---

## C6 — Classificação PAM50 como métrica secundária

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Mostrar que F1 discrimina pouco a 5–10% e mais a 20–30% |
| **Evidência** | `table_f1_compact_with_rfeca.csv` / `fig_macrof1_*`; F1 ~0.88→~0.85–0.87 |
| **Arquivo** | `artifacts/original_rfeca_reduced_metabric/classification/`; comparison F1 |
| **Figura** | `fig_macrof1_by_missingness.png` |
| **Tabela** | F1 compact / unified_imputer_clf |
| **Importância** | Média |
| **Risco de crítica** | Nesting diferente favorece RFECA |
| **Defesa** | Caveat explícito; não usar F1 como claim principal; RMSE é eixo |

---

## C7 — Anatomia de seleção (preditores / estabilidade)

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Caracterização: ~22 preditores; Jaccard ~0.68 |
| **Evidência** | `rfeca_internal_analysis.md`, `rfeca_n_predictors_distribution.csv`, `rfeca_subset_jaccard_by_gene.csv` |
| **Arquivo** | mesmos + `gene_summary.csv` por slot |
| **Figura** | (opcional histogramas a partir da distribuição; não obrigatório) |
| **Tabela** | `rfeca_top_predictors.csv` |
| **Importância** | Média — interpretabilidade operacional |
| **Risco de crítica** | Overclaim biológico |
| **Defesa** | Reportar como estatística de seleção, não como descoberta de biomarcador |

---

## C8 — Paralelismo methodology-preserving

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Speedup com fingerprints idênticos ao serial |
| **Evidência** | Autotune 1–8 workers; speedup 4.70× @ 8; valid=True |
| **Arquivo** | `artifacts/parallel_benchmark/benchmark_workers.md`; `parallel_benchmark_snapshot.csv` |
| **Figura** | `artifacts/parallel_benchmark/benchmark_workers.png` (se citar) |
| **Tabela** | tabela workers no `computational_cost.md` |
| **Importância** | Média-alta — factibilidade |
| **Risco de crítica** | Produção usou 16 ≠ ótimo 8 |
| **Defesa** | Autotune em 8 genes; produção 50 genes favorece mais workers; declarar que 16 não foi re-otimizado formalmente |

---

## C9 — Freeze reprodutível

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Pacote freeze com seeds, hashes, pins, 40 slots A |
| **Evidência** | `FREEZE/manifest.json`: `all_seeds_match_v2_formula=true`, `all_classification_A=true` |
| **Arquivo** | `FREEZE/*` |
| **Figura** | n/a |
| **Tabela** | `mask_hashes.csv` |
| **Importância** | Alta para banca |
| **Risco de crítica** | Joblibs grandes fora do git |
| **Defesa** | Política explícita: regenerar com `--resume`; hashes de máscara no freeze |

---

## C10 — Pacote de evidência para escrita (`final_analysis/`)

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Consolidação auditada sem reexecução |
| **Evidência** | Este diretório + INDEX |
| **Arquivo** | `artifacts/final_analysis/**` |
| **Figura** | `figures/*` |
| **Tabela** | `results_tables.csv` |
| **Importância** | Prática |
| **Risco de crítica** | “Análise post-hoc seletiva” |
| **Defesa** | Escopo pré-definido Mean/KNN/MF/OriginalRFECA; exclusão de RFECA-k* justificada (legado ≠ método principal) |

---

## C11 — O que explicitamente NÃO se contribui (anti-claims)

| Campo | Conteúdo |
|---|---|
| **Contribuição** | Transparência de limites |
| **Evidência** | `conclusion_points.md` (“NÃO permitem”); `statistical_analysis.md` |
| **Arquivo** | mesmos |
| **Figura** | n/a |
| **Tabela** | Δ descritivos com nota DESCRIPTIVE_ONLY |
| **Importância** | Alta para credibilidade |
| **Risco de crítica** | Parecer “fraco” sem p-value |
| **Defesa** | Melhor que p-value inválido; propor desenho pareado futuro |
