# Strengths vs Limitations

## Strengths

| contribuição | evidência experimental | figura | tabela | importância para o artigo |
|---|---|---|---|---|
| Protocolo OriginalRFECA TARGET-WISE leakage-safe (sem SimpleImputer/chaining) | 40/40 slots: coverage=1.0, fallback=0, n_predictor_nans=0 | flowchart OriginalRFECA | `methodology_audit.md`, FREEZE manifest | Alta — validade metodológica |
| RMSE competitivo/superior vs Mean/KNN/MissForest (descritivo) | Melhor RMSE em 7/8 células; Δ vs MF até −9.3% (MAR 30%) | `fig_rmse_by_missingness.png`, bars | `central/headline_rmse_f1.csv`, effect sizes | Alta — resultado principal |
| Estabilidade do RMSE entre taxas (esp. MAR) | Span RMSE OriginalRFECA pequeno vs degradação de KNN/MF | `fig_rmse_by_missingness.png` | `results_tables_display.csv` | Alta — diferenciação |
| Seleção interpretável (preditores + Jaccard) | ~21.6 preditores/gene; Jaccard ~0.68; top proliferativos | (opcional heatmap z) | `rfeca_top_predictors.csv`, gene reports | Média — discussão |
| Reprodutibilidade / freeze | Seeds v2, mask hashes, config snapshot, requirements | — | FREEZE/, `methods_checklist.md` | Alta — Methods |
| Stats formais entre baselines (mesmo protocolo) | Friedman + Wilcoxon–Holm Mean/KNN/MF | — | `stats_final/A_*.csv` | Média — suporte baselines |
| Separação explícita válida vs descritiva | Sem p-values inválidos RFECA vs baselines | — | `stats_final/B_*.csv`, `one_page_results.md` | Alta — credibilidade |

## Limitations

| limitação | impacto potencial | mitigação adotada | possibilidade de trabalho futuro |
|---|---|---|---|
| Protocolos distintos (holdout TARGET-WISE vs CV shared-mask) | Impede inferência confirmatória RFECA vs baselines | Comparação descritiva; disclaimer; stats A só baselines | Redesign paired shared-mask / mesmo n_reps |
| Ausência de comparação estatística confirmatória para OriginalRFECA vs baselines | Revisores podem pedir p-values | Não reportar p inválidos; IC bootstrap descritivo | Experimento pareado dedicado |
| Custo computacional (~51.5 h, 16 workers) | Barreira de adoção / escala | Paralelismo gene-nível; fingerprint-identical | Early-stop RFE, cache correlações, approx. prefixes |
| Apenas PAM50 (50 genes) | Generalização a transcriptoma denso incerta | Escopo explícito; painel clinicamente usado | Extender a painéis maiores / RNA-seq |
| Apenas câncer de mama (METABRIC; CPTAC auxiliar) | Validade externa limitada | Declarar domínio | Outras histologias / multi-câncer |
| Ausência de validação clínica | Não prova utilidade terapêutica | Foco em erro de imputação + F1 PAM50 | Estudos clínicos / utilidade em pipelines reais |
| Ausência de OriginalRFACA no freeze principal | Comparação RFE vs RFA incompleta | RFACA só smoke/audit | Campanha RFACA TARGET-WISE completa |
| Protocolo target-wise (preditores da matriz completa) | Não modela missingness conjunta realista nos preditores | Contrato metodológico anti-leakage explícito | Variante com preditores parcialmente observados |
| Sem RMSE gene-level para baselines | Heatmap/análise gene×método incompleta | NaNs explícitos; sem interpolação | Exportar métricas por gene nos baselines |
| Sem RV para OriginalRFECA | Não compara preservação de correlação | Reportar RV só baselines | Calcular RV pós-imputação TARGET-WISE |
| Classificação: nesting diferente | F1 cross-protocol pouco comparável | Caveat; F1 secundário | Aninhar RFECA no CV de classificação |
| Missingness MCAR/MAR simulados (não MNAR clínico) | Mecanismos reais podem diferir | Dois mecanismos + exact-count | MNAR / missingness real de plataforma |
