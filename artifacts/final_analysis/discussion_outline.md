# discussion_outline.md

Roteiro de discussão em tópicos (sem parágrafos).

---

## D1 — RMSE: OriginalRFECA vs MissForest

**Resultado observado**
- RFECA menor RMSE em 7/8 células; perde só MAR 5%
- Gaps crescem com a taxa (máx. MAR 30% ≈ −0.066)

**Possíveis explicações**
- Preditores TARGET-WISE sem chaining
- Seleção gene-específica (prefixos + RFE) vs modelo global MissForest
- MAR estruturado penaliza mais métodos que misturam dependências sob missingness coordenada
- A taxas baixas, ambos saturam perto do ótimo local → gaps pequenos / empate

**Literatura relacionada**
- Imputação multivariada / random forests (MissForest)
- Seleção de features + SVR em bioexpressão
- MCAR vs MAR e robustez de imputadores
- Leakage em pipelines de missingness artificial

**Implicações**
- Em PAM50 correlacionado, TARGET-WISE supervisionado é competitível com SOTA interno
- Ganho prático maior quando missingness ↑ e/ou MAR

**Limitações**
- Protocolos de avaliação não idênticos
- Sem teste pareado válido
- Sem RV para RFECA

**Próxima pergunta**
- O ranking mantém-se se Mean/KNN/MissForest forem reavaliados sob o mesmo `repeated_mask_holdout` + seed v2?

---

## D2 — Estabilidade entre taxas

**Resultado observado**
- RMSE RFECA quase plano (5–30%)
- MissForest/KNN sobem com a taxa (esp. MAR)

**Possíveis explicações**
- Preditores sempre completos → sinal de entrada estável; só n de treino do alvo cai
- Baselines veem matriz cada vez mais degradada no CV

**Literatura relacionada**
- Sensibilidade de imputadores à fração missing
- Regimes de informação em painéis pequenos

**Implicações**
- Argumento central a favor do desenho TARGET-WISE (robustez operacional)

**Limitações**
- Estabilidade ≠ otimalidade absoluta
- Só PAM50 / METABRIC

**Próxima pergunta**
- Há limiar de missingness (>30%) onde o SVR por gene colapsa?

---

## D3 — MCAR vs MAR

**Resultado observado**
- RFECA: MAR ~+0.02–0.03 RMSE vs MCAR, mas estável
- KNN/MissForest degradam mais em MAR alto

**Possíveis explicações**
- MAR induz dependência missingness–valores → pior para imputadores que assumem aleatoriedade
- TARGET-WISE isola o alvo; preditores intactos mitigam parte do dano

**Literatura relacionada**
- Mecanismos de missingness (Rubin); simulações MAR em omics

**Implicações**
- Reportar sempre os dois mecanismos; não generalizar só de MCAR

**Limitações**
- MAR simulado ≠ MNAR clínico real

**Próxima pergunta**
- Como se comporta sob MNAR / missingness de plataforma real?

---

## D4 — Mean como baseline fraca mas F1 “não catastrófica”

**Resultado observado**
- Mean RMSE péssimo; F1 a 5% ainda ~0.88

**Possíveis explicações**
- Subtipo PAM50 com n grande: sinal de classe robusto a ruído de imputação leve
- RMSE mede células mascaradas; F1 mede decisão de classe

**Literatura relacionada**
- Desacoplamento métrica de reconstrução vs tarefa a jusante

**Implicações**
- Não usar só F1 para crowning de imputadores a baixa missingness

**Limitações**
- F1 depende do classificador (EnsembleSoft)

**Próxima pergunta**
- Quais tarefas a jusante (além de PAM50) são sensíveis ao RMSE?

---

## D5 — Macro-F1 e nesting

**Resultado observado**
- Empate F1 a 5–10%; RFECA melhor relativo a 20–30%
- Protocolo F1 RFECA = pós-imputação; baselines = imputer-in-CV

**Possíveis explicações**
- Nesting favorece RFECA (sem refit caro por fold)
- Ou vantagem real de matriz melhor a alta missingness

**Literatura relacionada**
- Nested CV / honest pipelines em ML biomédico

**Implicações**
- Tratar F1 como secundário e declarar caveat

**Limitações**
- Não comparável formalmente

**Próxima pergunta**
- Amostra de slots com RFECA aninhado no CV confirma o ranking F1?

---

## D6 — Anatomia da seleção (n predictores, Jaccard)

**Resultado observado**
- ~22 preditores; Jaccard ~0.68 entre reps

**Possíveis explicações**
- Prefixo longo + RFE reduz a um núcleo estável parcial
- Correlações PAM50 redundantes → múltiplos subconjuntos quase equivalentes

**Literatura relacionada**
- Estabilidade de seleção de features; coexpressão PAM50

**Implicações**
- Método interpretável ao nível de conjuntos, não de um único gene “mágico”

**Limitações**
- Jaccard ≠ validade biológica

**Próxima pergunta**
- Enrichment / overlap com vias conhecidas dos preditores frequentes?

---

## D7 — Custo computacional

**Resultado observado**
- ~51.5 h grid; speedup ~4.7× @ 8 workers; produção 16 workers

**Possíveis explicações**
- Gargalo RFE+SVR linear por gene
- Imbalance entre genes limita eficiência

**Literatura relacionada**
- Trade-offs accuracy–cost em imputação; parallel ML

**Implicações**
- Justificável para evidência científica; questionável para produção online

**Limitações**
- Hardware específico; sem benchmark MissForest wall-a-wall idêntico

**Próxima pergunta**
- Aproximações (early stop, fewer prefixes) preservam RMSE?

---

## D8 — CPTAC 2C / n pequeno (se discutir generalização)

**Resultado observado**
- n=117; F1 volátil; RFECA-k* legado não coroa F1
- OriginalRFECA TARGET-WISE não é o eixo do freeze CPTAC

**Possíveis explicações**
- Variância de fold com classes raras
- F1(5%)>F1(0%) como artefacto small-n

**Literatura relacionada**
- Small-sample ML; proteogenomics CPTAC

**Implicações**
- METABRIC = coorte principal; CPTAC = stress-test / auxiliar

**Limitações**
- Não misturar conclusões TARGET-WISE METABRIC com legado CPTAC

**Próxima pergunta**
- Replicar TARGET-WISE em CPTAC só em RMSE (sem F1 crowning)?

---

## D9 — Limites de inferência estatística

**Resultado observado**
- Stats Wilcoxon/Holm = legado k*
- Welch/boot vs MissForest = descritivo

**Possíveis explicações**
- Desenho experimental evoluiu mais rápido que a camada de testes

**Literatura relacionada**
- Multiple testing; paired designs; reporting guidelines

**Implicações**
- Transparência aumenta credibilidade perante banca

**Limitações**
- Não há “p oficial” Original vs MissForest

**Próxima pergunta**
- Redesign paired shared-mask holdout para todos os imputadores

---

## D10 — Contribuição científica global

**Resultado observado**
- Pipeline leakage-safe + evidência METABRIC competitiva/estável

**Possíveis explicações**
- Combinação protocolo + seleção adaptativa + preditores completos

**Literatura relacionada**
- Reproducible pipelines; method papers vs leaderboard papers

**Implicações**
- Contribuição = método + contrato de validade + evidência, não só um número RMSE

**Limitações**
- Escopo PAM50

**Próxima pergunta**
- Extensão a transcriptoma denso com candidatura/filtro prévio
