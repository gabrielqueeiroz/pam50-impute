# reviewer_questions.md

Perguntas de um revisor rigoroso (bioinformática / ML omics) e respostas **apenas** com evidência disponível.  
Se a evidência não existir: **EVIDÊNCIA INSUFICIENTE**.

---

# Metodologia

### Q1. O protocolo TARGET-WISE não é imputação multivariada. Por que comparar com MissForest?
**Resposta:** Concordamos no tipo de modelo: RFECA é condicional por gene com preditores da matriz completa. A comparação é pragmática (erro de preenchimento nas mesmas células mascaradas). O caveat de paradigma está em `threats_to_validity.md` / `methodology_audit.md`.

### Q2. Como garantem ausência de leakage?
**Resposta:** Preditores de `X` original; máscara só no alvo; seleção `leakage_safe`; flags `n_predictor_nans=0`, `svr_coverage=1.0`, `fallback_rate=0` em 40/40 `DONE.json`; checklist em `methodology_audit.md`.

### Q3. Houve SimpleImputer ou chaining?
**Resposta:** Não no caminho TARGET-WISE final (auditoria SIM). Fallback de média só se sem modelo SVR — ocorrências 0 no freeze.

### Q4. A correlação/RFE usa o conjunto de teste?
**Resposta:** Não — `leakage_safe` / fit por gene só em observações reais do alvo (`methodology_audit.md` itens 5–8).

### Q5. Por que `use_scaler=false`?
**Resposta:** Configuração do freeze (`FREEZE/manifest.json`). **EVIDÊNCIA INSUFICIENTE** neste pacote para um sweep completo scaler on/off no grid 40 slots (há menção a sensibilidade noutros scripts, não consolidada aqui como resultado principal).

### Q6. `max_candidates=49` é arbitrário?
**Resposta:** Fixado no protocolo (PAM50 ⇒ no máx. 49 outros genes). Justificação de desenho; sensibilidade completa **EVIDÊNCIA INSUFICIENTE** como tabela principal do freeze.

### Q7. Avaliação `repeated_mask_holdout` vs CV aninhado — porquê?
**Resposta:** Custo de refitar OriginalRFECA por fold. Documentado no plano/protocolo de classificação pós-imputação. Trade-off explícito.

### Q8. O target mascarado entra no treino?
**Resposta:** Não (`methodology_audit.md` item 8).

---

# Experimentos

### Q9. Por que só METABRIC PAM50 como eixo principal?
**Resposta:** Freeze TARGET-WISE é METABRIC. CPTAC 2C existe na campanha six-imputer legada (n=117) mas não como freeze OriginalRFECA. Limitação de escopo declarada.

### Q10. Por que não MNAR?
**Resposta:** Só MCAR/MAR simulados. **EVIDÊNCIA INSUFICIENTE** para MNAR.

### Q11. 5 réplicas bastam?
**Resposta:** Suficiente para médias/IC t com df=4 no freeze; não prova cobertura total da variabilidade. Baselines usam 10. Limitação declarada.

### Q12. Taxas 5–30% são realistas?
**Resposta:** Grelha padrão do estudo; realismo clínico depende do ensaio. Não há calibração a missingness empírica METABRIC neste pacote. **EVIDÊNCIA INSUFICIENTE** para “taxa clínica típica”.

### Q13. Por que excluir RFECA-k5/10/20 das figuras finais?
**Resposta:** São legado `RFECA_SVR(k=*)` da campanha six-imputer, não o método OriginalRFECA TARGET-WISE. Evitar confundir identidades.

### Q14. CPTAC sugere que o método falha em n pequeno?
**Resposta:** CPTAC mostra F1 instável e RFECA-k* legado fraco relativo — **não** mede OriginalRFECA TARGET-WISE. Não se pode concluir falha do método final em CPTAC. **EVIDÊNCIA INSUFICIENTE** (TARGET-WISE CPTAC).

### Q15. Classificação usa os mesmos folds/máscaras que a imputação?
**Resposta:** Baselines: imputer-within-CV shared-mask. RFECA: matriz já imputada + StratifiedKFold identity. Protocolos diferentes — caveat obrigatório.

---

# Estatística

### Q16. Onde está o teste de significância OriginalRFECA vs MissForest?
**Resposta:** Não há teste pareado válido. Motivos: n_reps, seeds, protocolo (`statistical_analysis.md`). Há Δ descritivos + Cohen's d / IC boot com nota DESCRIPTIVE_ONLY.

### Q17. Posso reportar os p Welch da tabela descritiva?
**Resposta:** Não como evidência confirmatória. O próprio pacote recomenda não usar.

### Q18. Os Wilcoxon/Holm do `stats/` validam o OriginalRFECA?
**Resposta:** Não — referem-se a `RFECA_SVR(k=*)` legado na campanha six-imputer.

### Q19. Múltiplos testes ao longo de 8 células — corrigiram?
**Resposta:** Para o contraste Original vs MissForest descritivo, **não** há família de testes formais. **EVIDÊNCIA INSUFICIENTE** / não aplicável a p inválidos.

### Q20. IC das figuras baselines vs RFECA são comparáveis?
**Resposta:** Ambos usam lógica de IC entre réplicas, mas n=10 vs n=5 e protocolos diferem — comparar visualmente com cautela.

---

# Implementação

### Q21. Paralelismo altera resultados?
**Resposta:** No microbenchmark (8 genes), fingerprints idênticos serial vs 2–8 workers (`benchmark_workers.md`).

### Q22. Por que 16 workers se o autotune recomenda 8?
**Resposta:** Autotune em 8 genes; produção 50 genes. 16 usado no plano de execução. Não há re-benchmark formal a 16 no CSV principal. **EVIDÊNCIA INSUFICIENTE** para otimalidade de 16.

### Q23. BLAS=1 é necessário?
**Resposta:** Sim no desenho methodology-preserving (evitar oversubscription com gene-workers). Documentado no runner/freeze.

### Q24. Código e artefatos estão alinhados?
**Resposta:** Freeze declara ambiente; `final_analysis` confirma 40 slots A e artefatos sob `imputation_original/`. Reexecução bit-a-bit nesta máquina não foi refeita neste passo de escrita.

### Q25. Modelos joblib estão versionados?
**Resposta:** Política de freeze: joblibs grandes fora do git; regeneráveis com `--resume`. Hashes de máscara sim.

---

# Comparabilidade

### Q26. Como podem comparar holdout 5 reps com CV 10 reps?
**Resposta:** Comparação descritiva entre campanhas publicadas do projeto; não é experimento pareado. Limitação central (`threats_to_validity.md`).

### Q27. Seeds v2 vs legacy — máscaras diferentes?
**Resposta:** Sim. Freeze: `all_seeds_match_v2_formula=true` para OriginalRFECA; baselines paper usam legacy.

### Q28. F1 RFECA não está enviesado por identity imputer?
**Resposta:** Possível viés relativo a imputer-in-CV. Declarado; F1 não é claim principal.

### Q29. MissForest hiperparâmetros foram otimizados?
**Resposta:** Usa configuração da campanha six-imputer do projeto. Sweep conjunto **EVIDÊNCIA INSUFICIENTE** no pacote final_analysis.

### Q30. KNN k=5 é justo?
**Resposta:** Identidade fixa da campanha (`KNN(k=5,dist)`). Alternativas de k **EVIDÊNCIA INSUFICIENTE** aqui.

---

# Limitações

### Q31. Generaliza para >50 genes?
**Resposta:** **EVIDÊNCIA INSUFICIENTE**.

### Q32. Generaliza para outros cancer types / plataformas?
**Resposta:** **EVIDÊNCIA INSUFICIENTE** (só METABRIC principal + CPTAC legado).

### Q33. RV do OriginalRFECA?
**Resposta:** Não calculado no freeze. **EVIDÊNCIA INSUFICIENTE**.

### Q34. Tempo vs MissForest head-to-head?
**Resposta:** Wall RFECA ~51.5 h no grid; MissForest wall comparável não consolidado slot-a-slot. Comparação qualitativa apenas (`computational_cost.md`).

### Q35. Fallback nunca ocorre — método frágil fora do METABRIC?
**Resposta:** Zero fallback no freeze METABRIC; comportamento noutros dados **EVIDÊNCIA INSUFICIENTE**.

---

# Validade

### Q36. RMSE em máscara artificial reflete erro real?
**Resposta:** É o construct padrão do estudo; não equivale a erro clínico. Limitação de construct validity.

### Q37. Melhor RMSE implica melhor decisão clínica PAM50?
**Resposta:** Não automaticamente — F1 quase empatada a 5–10%. Dados suportam desacoplamento parcial.

### Q38. A hipótese foi confirmada?
**Resposta:** Parcialmente: desempenho descritivo competitivo/superior e estabilidade — sim nos artefatos METABRIC; confirmação estatística formal vs MissForest — não.

### Q39. Há risco de HARKing / seleção de métricas?
**Resposta:** Eixo principal RMSE declarado; F1 secundário; exclusão RFECA-k* justificada por mudança de método. Risco mitigado pela documentação, não eliminado.

### Q40. O que mudaria a conclusão?
**Resposta:** (i) MissForest melhor em holdout pareado shared-mask; (ii) RV RFECA pior; (iii) F1 aninhada inverter ranking a 30%; (iv) falha de coverage/fallback fora de METABRIC. Nenhum destes foi observado no freeze atual — mas (i)–(iii) não foram testados no desenho ideal.

---

# Extra (implementação / narrativa)

### Q41. EnsembleSoft é necessário?
**Resposta:** Nos CSVs RFECA, EnsembleSoft ≈ SVC; GB/RF/LogReg piores. Não prova necessidade do ensemble. CPTAC sem multiclf. **EVIDÊNCIA INSUFICIENTE** para “ensemble obrigatório”.

### Q42. Por que MAE e RMSE contam a mesma história?
**Resposta:** Hierarquias alinhadas em `results_tables_display.csv` — coerência interna.

### Q43. A banca pode reproduzir os 40 slots?
**Resposta:** Com ambiente pinado + comando freeze + dados METABRIC + tempo (~50 h) / hardware adequado — sim em princípio; joblibs não precisam estar no git se `--resume`/recompute for aceite.
