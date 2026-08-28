# Possíveis perguntas de um revisor relacionadas aos resultados

Respostas baseadas **exclusivamente** nos resultados/artefactos obtidos. Sem evidências inventadas.

---

### L1 — Protocolos distintos
**Pergunta:** Como podem comparar OriginalRFECA com MissForest se os protocolos de avaliação diferem?

**Resposta:** Não reivindicamos equivalência formal. A comparação é **descritiva** (ΔRMSE/MAE, rankings, win counts, IC bootstrap). Testes confirmatórios (Friedman/Wilcoxon–Holm) aplicam-se apenas entre Mean/KNN/MissForest no protocolo CV shared-mask (`stats_final/A_*`). OriginalRFECA usa `repeated_mask_holdout` TARGET-WISE com 5 reps e seeds v2.

---

### L2 — Sem p-values RFECA vs MissForest
**Pergunta:** Por que não há Wilcoxon pareado OriginalRFECA vs MissForest?

**Resposta:** Unidades experimentais não são emparelháveis (máscaras/seeds/n_reps/protocolos diferentes). Reportar p-values seria metodologicamente inválido; por isso `p_value=NOT_REPORTED` em `stats_final/B_*`.

---

### L3 — Custo computacional
**Pergunta:** O método é praticável dado ~51.5 h de wall time?

**Resposta:** O grid completo (40 slots, 50 genes, RFE+SVR) custou ~51.5 h com 16 gene-workers nesta máquina. Microbenchmarks mostraram speedup até ~4.7× com fingerprints idênticos. É custo de evidência metodológica; otimizações (early-stop, cache) ficam como trabalho futuro — não foram aplicadas ao freeze.

---

### L4 — Apenas PAM50
**Pergunta:** Os resultados generalizam para transcriptomas densos?

**Resposta:** Não foi testado. Todos os resultados de imputação do freeze são em 50 genes PAM50 METABRIC. Não há evidência experimental no pacote final para p≫50.

---

### L5 — Apenas mama
**Pergunta:** Funciona fora de câncer de mama?

**Resposta:** O freeze OriginalRFECA é METABRIC. CPTAC 2C aparece na campanha six-imputer legado, não como freeze TARGET-WISE. Não há validação multi-câncer no pacote final.

---

### L6 — Validação clínica
**Pergunta:** A melhoria de RMSE traduz-se em benefício clínico?

**Resposta:** Não avaliámos outcomes clínicos. Métricas: RMSE/MAE (e F1 PAM50 secundário). Qualquer afirmação de utilidade clínica estaria além dos dados.

---

### L7 — Ausência de RFACA
**Pergunta:** Por que não comparar OriginalRFACA?

**Resposta:** O freeze principal é OriginalRFECA apenas. RFACA existe no código/smoke, mas não no grid final de 40 slots. Não há números RFACA TARGET-WISE comparáveis no pacote final.

---

### L8 — TARGET-WISE “não é imputação real”
**Pergunta:** Usar a matriz completa como preditores não é irrealista?

**Resposta:** É uma escolha metodológica explícita (`input_protocol=target_wise_complete_predictors`) para evitar leakage/chaining. Avalia erro nas células mascaradas do alvo com preditores observados. Não pretende simular missingness conjunta em todos os genes; isso é limitação declarada, não um bug oculto.

---

### L9 — Heatmap gene×método incompleto
**Pergunta:** Por que Mean/KNN/MissForest estão em cinza no heatmap?

**Resposta:** Os artefactos finais dos baselines não armazenam RMSE por gene — só agregados por fold. Marcámos NaN sem interpolar. A dificuldade gene-level reportada refere-se ao OriginalRFECA.

---

### L10 — MAR 5% MissForest melhor
**Pergunta:** O método falha sob MAR a baixa taxa?

**Resposta:** Em MAR 5%, MissForest tem RMSE médio 0.629 vs 0.641 do OriginalRFECA (Δ descritivo +0.012). É a única célula RMSE em que MissForest vence. Em MAR ≥10% e em todo MCAR do grid, OriginalRFECA tem menor RMSE médio.

---

### L11 — F1 quase empatado
**Pergunta:** Se F1 quase não muda, qual a relevância prática?

**Resposta:** Em 5–10% as diferenças de Macro-F1 são pequenas; a 20–30% o OriginalRFECA aparece relativamente melhor, com caveat de nesting de classificação diferente. O resultado primário do estudo é **erro de imputação (RMSE)**, não F1.

---

### L12 — Reprodutibilidade
**Pergunta:** Os resultados são reprodutíveis?

**Resposta:** Freeze `v0.3.1-original-rfeca-targetwise` com mask hashes, seeds v2 auditados, config snapshot, requirements e 40/40 slots completos. Paralelismo gene-nível foi validado com fingerprints idênticos ao serial no microbenchmark.
