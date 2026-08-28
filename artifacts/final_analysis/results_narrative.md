# results_narrative.md

Roteiro para a seção de Resultados (não é texto de dissertação).  
Figuras: `artifacts/final_analysis/figures/` + aliases `copied_comparison_*`.  
Tabelas: `results_tables.csv`, `results_tables_display.csv`, CSVs auxiliares.

---

# Blocos figura/tabela

=================================================
## Figura: `fig_rmse_by_missingness` (+ `copied_comparison_fig01_rmse`)

**Objetivo:** mostrar RMSE vs % missingness, MCAR e MAR, quatro métodos.

**Mensagem principal:** OriginalRFECA e MissForest dominam; Mean saturado; KNN intermédio e piora em MAR; RFECA quase plano nas taxas.

**Valores a destacar:**
- MCAR: RFECA 0.613→0.621; MissForest 0.617→0.653; Mean ~1.06
- MAR: RFECA ~0.64 estável; MissForest 0.629→0.705; KNN 0.700→0.857
- Única célula em que MissForest vence: MAR 5% (0.629 vs 0.641)

**Comparações:** RFECA vs MissForest em cada taxa; RFECA vs Mean; KNN vs MissForest em MAR 30%.

**Afirmações suportadas:** ranking descritivo RMSE; estabilidade RFECA entre taxas; degradação baselines em MAR.

**Afirmações NÃO suportadas:** significância formal pareada; superioridade em todos os datasets.

**Ligação seguinte:** MAE para confirmar que o padrão não é artefacto só do RMSE.

=================================================
## Figura: `fig_mae_by_missingness`

**Objetivo:** mesma comparação em MAE.

**Mensagem principal:** ordenação alinhada ao RMSE (Mean ≫ KNN > MissForest ≈/≲ RFECA).

**Valores:** RFECA MAE ~0.405–0.421; Mean ~0.75–0.83.

**Comparações:** coerência MAE–RMSE por método.

**Suportadas:** consistência da hierarquia de erro absoluto.

**Não suportadas:** implicações clínicas do MAE.

**Ligação seguinte:** barras RMSE para leitura por taxa discreta.

=================================================
## Figura: `fig_rmse_bars_5_10_20_30` (+ `copied_comparison_fig05_rmse_bars`)

**Objetivo:** RMSE agrupado por taxa com IC (visual).

**Mensagem principal:** gap RFECA–MissForest aumenta com a taxa (sobretudo MAR); Mean sempre pior.

**Valores:** gaps RFECA−MF ≈ −0.005 (MCAR 5%) … −0.066 (MAR 30%); gap 1º vs 2º até ~9% relativo em MAR 30%.

**Comparações:** barras lado a lado por mecanismo.

**Suportadas:** magnitude descritiva dos gaps por célula.

**Não suportadas:** p-values das barras como teste formal.

**Ligação seguinte:** RV só nos baselines (ausência RFECA).

=================================================
## Figura: `fig_rv_by_missingness_baselines`

**Objetivo:** RV (estrutura de correlação) para Mean/KNN/MissForest.

**Mensagem principal:** RV alto para todos os baselines; MissForest tipicamente melhor preservação; Mean pior em taxas altas.

**Valores:** RV ~0.966–1.000 conforme taxa/método (ver `results_tables_display.csv`).

**Comparações:** MissForest vs Mean em RV a 30%.

**Suportadas:** baselines preservam bem correlação no protocolo CV.

**Não suportadas:** qualquer afirmação de RV para OriginalRFECA (n/a no freeze).

**Ligação seguinte:** estabilidade entre réplicas.

=================================================
## Figura: `fig_stability_rmse_cv`

**Objetivo:** CV do RMSE entre réplicas vs taxa.

**Mensagem principal:** KNN mais variável; Mean CV baixo mas saturado; RFECA estável entre competitivos.

**Valores:** CV médio Mean 0.0105 · RFECA 0.0124 · MF 0.0146 · KNN 0.0161.

**Comparações:** não interpretar Mean como “melhor estável”.

**Suportadas:** ranking de dispersão relativa entre réplicas.

**Não suportadas:** que CV baixo = melhor imputador.

**Ligação seguinte:** F1 PAM50 (utilidade a jusante).

=================================================
## Figura: `fig_macrof1_by_missingness` (+ `copied_comparison_fig03_f1` / `fig06`)

**Objetivo:** Macro-F1 EnsembleSoft vs missingness.

**Mensagem principal:** a 5–10% métodos quase empatados (~0.88); a 20–30% RFECA/MissForest separam-se de Mean/KNN; RFECA relativamente forte a 30%.

**Valores (aprox. display F1 compact):** MCAR 30% Mean 0.855 / KNN 0.858 / MF 0.866 / RFECA 0.872; MAR 30% Mean/KNN ~0.848 / MF 0.861 / RFECA 0.872.

**Comparações:** F1 vs RMSE (desacoplamento a taxas baixas).

**Suportadas:** pequena discriminação F1 a baixa missingness; vantagem relativa RFECA a alta taxa (descritiva).

**Não suportadas:** F1 RFECA comparável formalmente ao nesting dos baselines; superioridade EnsembleSoft vs SVC.

**Ligação seguinte:** tabela mestra numérica.

=================================================
## Tabela: `results_tables.csv` / `results_tables_display.csv`

**Objetivo:** números exatos RMSE/MAE/RV + ranking + gaps.

**Mensagem principal:** suporte quantitativo a todas as figuras de imputação.

**Valores a citar na escrita:** colunas `ranking_rmse`, `OriginalRFECA_minus_MissForest`, `gap_1_vs_2_pct`, IC quando disponíveis.

**Comparações:** célula a célula.

**Suportadas:** 7/8 vitórias RFECA; única derrota MAR 5%.

**Não suportadas:** inferência causal além do desenho.

**Ligação seguinte:** slots crús / estabilidade tabular.

=================================================
## Tabela: `original_rfeca_slot_level.csv` + `stability_by_method_rate.csv`

**Objetivo:** dispersão entre réplicas; seeds; wall.

**Mensagem principal:** freeze completo e estável operacionalmente (A, coverage 1.0).

**Valores:** 40 linhas; wall por slot; RMSE por rep.

**Suportadas:** reprodutibilidade operacional do grid.

**Não suportadas:** que 5 reps esgotam a variabilidade amostral.

**Ligação seguinte:** análise interna de seleção.

=================================================
## Tabela/texto: `rfeca_internal_analysis.md` + `rfeca_top_predictors.csv` + Jaccard CSV

**Objetivo:** o que o método seleciona.

**Mensagem principal:** ~22 preditores/gene; prefixos longos; subconjuntos moderadamente estáveis (Jaccard ~0.68).

**Valores:** n_pred mean 21.56 [3,24]; prefix mean 43.62; Jaccard 0.677.

**Suportadas:** descrição do comportamento de seleção no freeze.

**Não suportadas:** interpretabilidade biológica causal dos preditores sem análise externa.

**Ligação seguinte:** custo.

=================================================
## Texto/tabela: `computational_cost.md` + `parallel_benchmark_snapshot.csv`

**Objetivo:** wall time e paralelismo.

**Mensagem principal:** grid ~51.5 h; speedup microbenchmark até 4.7× @ 8 workers; produção usou 16.

**Valores:** wall total 51.46 h; tabela workers 1→8.

**Suportadas:** custo e preservação de fingerprint no autotune.

**Não suportadas:** 16 = ótimo global.

**Ligação seguinte:** limites estatísticos.

=================================================
## Texto: `statistical_analysis.md`

**Objetivo:** declarar o que NÃO se pode testar formalmente + Δ descritivos.

**Mensagem principal:** sem Wilcoxon pareado válido RFECA vs MissForest; Δ descritivos maiores a 20–30%.

**Valores:** tabela Δ / Cohen's d / IC boot (descritivo).

**Suportadas:** impossibilidade do contraste formal; magnitudes descritivas.

**Não suportadas:** usar Welch p como evidência confirmatória.

**Ligação seguinte:** (fecha resultados; abre discussão).

=================================================
## Texto: `methodology_audit.md`

**Objetivo:** checklist leakage-safe (pode ir a Métodos ou apêndice de Resultados/QA).

**Mensagem principal:** 9 itens SIM no freeze TARGET-WISE.

**Suportadas:** ausência de SimpleImputer/chaining/fallback nos 40 slots.

**Não suportadas:** equivalência de protocolo com baselines.

---

# Sequência lógica da seção Resultados

**Resultado 1 — Protocolo e completude do freeze**  
40 slots A; coverage 1.0; fallback 0; seeds v2.  
→ estabelece que os números seguintes são de um pipeline válido.

↓

**Resultado 2 — RMSE vs missingness (figura linhas)**  
Hierarquia Mean ≪ KNN < MissForest ≲ OriginalRFECA; RFECA estável.  
→ define o ranking principal.

↓

**Resultado 3 — MAE**  
Confirma a mesma hierarquia.  
→ robustez da métrica de erro.

↓

**Resultado 4 — Barras + tabela de gaps**  
Magnitude Δ vs MissForest/Mean; MAR 30% como cenário de maior ganho relativo.  
→ quantifica “quanto”.

↓

**Resultado 5 — RV baselines (+ nota ausência RFECA)**  
Baselines preservam correlação; RFECA sem RV no freeze.  
→ delimita o que se mediu sobre estrutura.

↓

**Resultado 6 — Estabilidade entre réplicas**  
CV; nota sobre Mean saturado; RFECA estável entre competitivos.  
→ fiabilidade das médias.

↓

**Resultado 7 — Macro-F1 PAM50**  
Empate a baixa taxa; separação a 20–30%; caveat nesting.  
→ utilidade a jusante (secundária).

↓

**Resultado 8 — Anatomia do OriginalRFECA**  
n predictores; Jaccard; tempo/gene.  
→ o método não é caixa-preta operacional.

↓

**Resultado 9 — Custo e paralelismo**  
51.5 h; speedup fingerprint-safe.  
→ factibilidade.

↓

**Resultado 10 — Limite estatístico explícito**  
Sem teste pareado válido vs MissForest; stats legado ≠ OriginalRFECA.  
→ honestidade antes da Discussão.

---

# Notas de escrita (operacionais)

- Sempre declarar diferença de protocolo baselines vs OriginalRFECA numa frase de rodapé/nota.
- Preferir “menor RMSE médio” a “significativamente melhor”.
- CPTAC 2C: se mencionado nos Resultados, como coorte auxiliar legado — não misturar com freeze TARGET-WISE.
