# Conclusões extraídas dos dados

## Os dados suportam afirmar que...

1. O pipeline OriginalRFECA TARGET-WISE foi executado de ponta a ponta em METABRIC PAM50 sob MCAR e MAR nas taxas 5/10/20/30% × 5 réplicas (40 slots), todos com classificação operacional **A**, `svr_coverage=1.0` e `fallback_rate=0`.
2. Em RMSE, OriginalRFECA está entre os melhores métodos no conjunto Mean/KNN/MissForest/OriginalRFECA; venceu ou empatou no 1º lugar em **7/8** células mecanismo×taxa (ranking por RMSE médio).
3. OriginalRFECA teve RMSE menor que MissForest em **7/8** células (Δ = RFECA − MF < 0).
4. O RMSE do OriginalRFECA é **notavelmente estável entre taxas** dentro de cada mecanismo (variação entre 5–30% pequena face a KNN/MissForest em MAR).
5. Em MAR, a vantagem relativa do OriginalRFECA frente a KNN/MissForest aumenta com a taxa de missingness (especialmente vs KNN @ 30%).
6. Mean é consistentemente o pior imputador em RMSE (~1.06–1.17) no METABRIC.
7. Em Macro-F1 (EnsembleSoft), diferenças entre imputadores são pequenas a 5–10% e mais favoráveis ao OriginalRFECA a 20–30%, com a ressalva de protocolos de classificação distintos.
8. O custo wall do grid OriginalRFECA (40 slots, 16 gene-workers) foi da ordem de **51.5 h** nesta máquina.
9. Paralelismo gene-nível preservou fingerprints vs serial no benchmark e produziu speedups ~3–5× (até 8 workers no autotune).
10. O protocolo TARGET-WISE, tal como implementado e auditado nos artefatos, não apresentou eventos de leakage/fallback nos 40 slots.

## Os dados NÃO permitem afirmar que...

1. OriginalRFECA é estatisticamente superior a MissForest por teste pareado válido no mesmo protocolo (n/seeds/avaliação diferem).
2. OriginalRFECA generaliza a todos os painéis de expressão / todas as doenças.
3. O método é ótimo em CPTAC 2C (OriginalRFECA TARGET-WISE não foi o eixo principal lá; RFECA-k* legado comportou-se mal em F1/RMSE relativo).
4. EnsembleSoft é necessário (vs SVC sozinho) — no METABRIC são quase iguais; no CPTAC não há multiclf.
5. Melhor RMSE implica sempre melhor utilidade clínica / subtipagem.
6. O ganho justifica o custo em qualquer ambiente de produção (depende de orçamento e SLA).
7. Ausência de RV no freeze implica superioridade estrutural de correlação.
8. Resultados a 5 réplicas capturam toda a variabilidade amostral de máscaras.
9. O aninhamento pós-imputação da classificação RFECA é comparável formalmente ao imputer-within-CV dos baselines.
10. 16 workers é o ótimo global de paralelismo (autotune recomendou 8 no microbenchmark).
