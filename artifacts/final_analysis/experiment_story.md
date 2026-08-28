# experiment_story.md

História científica dos experimentos (formato técnico).  
Fonte: freeze `v0.3.1-original-rfeca-targetwise` + `artifacts/final_analysis/` + campanhas METABRIC/CPTAC existentes.  
Não é texto de dissertação.

---

# Problema científico

## Qual problema motivou o trabalho?

- Preencher valores ausentes em matrizes de expressão gênica (PAM50 / subtipagem molecular) sem contaminar a avaliação com informação das células mascaradas.
- Comparar imputadores clássicos (média, KNN, MissForest) com um método baseado em seleção de preditores + SVR (RFECA), sob mecanismos MCAR e MAR.
- Garantir que o método “original/notebook-faithful” fosse **leakage-safe** e reprodutível o suficiente para dissertação/artigo.

## Por que imputação de expressão gênica é importante?

- Ensaios e coortes clínicas frequentemente têm células/genes incompletos; análises a jusante (subtipo PAM50, correlações, modelos) assumem matrizes densas.
- Erro de imputação propaga-se para classificação e para estimativas de coexpressão.
- Em painéis pequenos e correlacionados (PAM50, 50 genes), a estrutura entre genes é informativa — mas só se for explorada sem leakage.

## Quais limitações existiam nos métodos anteriores?

- **Mean:** RMSE alto (~1.06–1.17 no METABRIC); ignora coexpressão.
- **KNN:** melhor que Mean, mas degrada forte em MAR alto (RMSE 0.857 @ MAR 30%).
- **MissForest:** forte baseline multivariada; piora com taxa e especialmente sob MAR.
- **RFECA legado (`RFECA_SVR(k=*)`):** campanha six-imputer com k fixo; em CPTAC 2C o legado não liderava F1/RMSE de forma convincente; risco de desenho que mistura seleção/imputação de forma não TARGET-WISE.
- Limitações transversais: chaining / SimpleImputer em caminhos iniciais; custo de RFE+SVR; falta de protocolo unificado leakage-safe documentado.

---

# Hipótese

**O trabalho testou a hipótese de que** um imputador OriginalRFECA **TARGET-WISE** (preditores sempre da matriz original completa; máscara artificial só no gene-alvo; seleção `leakage_safe` com prefixos de correlação + RFE + SVR linear; avaliação `repeated_mask_holdout`) alcança **erro de imputação (RMSE/MAE) competitivo ou superior** a Mean, KNN e MissForest no METABRIC PAM50 sob MCAR e MAR em várias taxas de missingness, **sem** eventos de leakage/fallback, com custo computacional tornável tratável via paralelismo gene-nível.

Hipóteses auxiliares explícitas nos desenhos:

- A estabilidade do RMSE face à taxa de missingness é maior no TARGET-WISE do que em baselines que degradam com a taxa.
- Em MAR, a vantagem relativa vs KNN/MissForest aumenta com a taxa.
- Macro-F1 PAM50 (EnsembleSoft) é métrica secundária e pode ser menos discriminativa a taxas baixas.

---

# Evolução metodológica

Ordem lógica (conforme artefatos e módulos do repositório):

## 1. Implementação inicial (legado / notebook-faithful parcial)

- Existência de `RFECA_SVR(k=*)` na campanha six-imputer (METABRIC + CPTAC discovery).
- Pipeline de benchmark completo: missingness artificial → imputação → classificação EnsembleSoft.
- **Por quê:** estabelecer baseline experimental e métricas (RMSE, RV, F1).

## 2. Problemas encontrados

- Custo alto de RFE/SVR.
- Necessidade de alinhar o método “Original” ao protocolo científico (sem atalhos que invalidem a dissertação).
- Comparações com k fixo vs seleção adaptativa confundiam a narrativa do método principal.

## 3. Descoberta / risco de data leakage

- Identificação de caminhos onde preditores ou seleção podiam ver informação incompatível com holdout (ex.: chaining, SimpleImputer, uso de valores já imputados, seleção sem isolamento do alvo mascarado).
- **Por quê a mudança era necessária:** leakage torna RMSE/F1 não interpretáveis como generalização.

## 4. Reconstrução do algoritmo (`imputation_original/`)

- Módulo paralelo ao legado: `OriginalRFECAImputer` / base de correlação + RFE.
- Separação explícita de `selection_protocol`, `input_protocol`, auditoria por gene.
- **Por quê:** isolar o método principal do legado `RFECA_SVR(k=*)` e permitir auditoria.

## 5. Implementação leakage-safe

- `selection_protocol=leakage_safe`: correlação/RFE/refit só em observações reais do alvo.
- Asserções: `n_predictor_nans_at_impute=0`, `svr_coverage=1.0`, `fallback_rate=0` nos slots finais.
- **Por quê:** cumprir o contrato científico da hipótese.

## 6. Protocolo TARGET-WISE

- `input_protocol=target_wise_complete_predictors`: preditores = matriz original completa; NaNs artificiais só na coluna-alvo.
- Avaliação `repeated_mask_holdout` (não CV aninhado do imputer no freeze principal — custo).
- **Por quê:** eliminar chaining; alinhar com o desenho “um modelo por gene-alvo”; tornar leakage verificável por construção.

## 7. Paralelização

- Paralelismo **apenas entre genes**; BLAS=1; fingerprints idênticos vs serial no microbenchmark (`parallel_benchmark/`).
- Produção do freeze: 16 gene-workers.
- **Por quê:** RFE/SVR por gene é o gargalo (~wall dezenas de horas no grid); serial inviável para 40 slots.

## 8. Benchmark / freeze final

- Grid METABRIC: MCAR+MAR × {5,10,20,30}% × 5 reps → 40 slots.
- Freeze `v0.3.1-original-rfeca-targetwise` (seeds v2, mask hashes, class A).
- Classificação PAM50 pós-imputação (identity no CV) + pacote comparison vs Mean/KNN/MissForest.
- Pacote `artifacts/final_analysis/` para escrita.
- **Por quê:** fechar evidência reprodutível sem reexecuções ad hoc.

---

# Experimentos realizados

## Datasets

| Dataset | Uso no trabalho final |
|---|---|
| **METABRIC** (n=1608, 50 genes PAM50) | Eixo principal OriginalRFECA TARGET-WISE + baselines |
| **CPTAC 2C / discovery** (n=117, 50 genes) | Campanha six-imputer legado (RMSE/RV/F1); evidência de n pequeno — **não** é o freeze OriginalRFECA |

## Mecanismos

- MCAR, MAR (simulados sobre células elegíveis / `originally_observed_mask`).

## Taxas

- 0%, 5%, 10%, 20%, 30% nas campanhas full baselines.
- Freeze OriginalRFECA: **5%, 10%, 20%, 30%**.

## Réplicas

- OriginalRFECA: **5** (reps 0–4), seed scheme **v2**, `base_seed=42`.
- Baselines Mean/KNN/MissForest (paper): **10** reps, seed **legacy**, shared-mask CV.

## Métodos (comparação principal)

- Mean (`SimpleMean`)
- KNN (`KNN(k=5,dist)`)
- MissForest
- **OriginalRFECA** (display “RFECA” em algumas figuras)
- Excluídos das figuras comparison finais: `RFECA-k5/k10/k20` (legado)

## Métricas

- Primárias imputação: **RMSE**, **MAE** (holdout nas células mascaradas).
- Baselines também: **RV** / correlação (não disponível no freeze TARGET-WISE).
- Secundária: **Macro-F1 / bal_acc** PAM50 (EnsembleSoft; + SVC/LogReg/RF/GB nos CSVs RFECA).
- Operacionais: `svr_coverage`, `fallback_rate`, wall time, n predictores, Jaccard de subconjuntos.

---

# Evidências obtidas

Fatos sustentados pelos dados (sem interpretação narrativa):

1. 40/40 slots OriginalRFECA: classificação operacional **A**, `svr_coverage=1.0`, `fallback_rate=0`, `n_predictor_nans_at_impute=0`.
2. RMSE médio OriginalRFECA é o menor em **7/8** células mecanismo×taxa; MissForest vence só **MAR 5%** (0.629 vs 0.641).
3. Δ RMSE (RFECA − MissForest): negativo em 7/8 células; maior magnitude em MAR 30% (−0.066) e MCAR/MAR 20–30%.
4. RMSE OriginalRFECA quase plano nas taxas (span MCAR ≈ 0.009; MAR ≈ 0.003).
5. Mean RMSE ≈ 1.06–1.17 em todas as células; pior método.
6. KNN RMSE sobe até 0.857 em MAR 30%.
7. Macro-F1 EnsembleSoft: diferenças pequenas a 5–10%; OriginalRFECA relativamente melhor a 20–30% (com nesting diferente).
8. Média de preditores selecionados ≈ 21.6; Jaccard médio de subconjuntos entre réplicas ≈ 0.68.
9. Wall total grid OriginalRFECA ≈ **51.5 h** (16 workers).
10. Microbenchmark paralelismo: speedup até **4.70×** @ 8 workers; fingerprints idênticos ao serial.
11. Wilcoxon/Holm do pacote `stats/` aplica-se a **RFECA_SVR(k=*) legado**, não ao OriginalRFECA TARGET-WISE.
12. Contraste formal pareado OriginalRFECA vs MissForest **não** é válido com os protocolos atuais (5 vs 10 reps; v2 vs legacy; mask-holdout vs CV).

---

# O que foi respondido

## Perguntas científicas respondidas

- É possível implementar OriginalRFECA TARGET-WISE leakage-safe com cobertura SVR completa e sem fallback no METABRIC PAM50?
- Qual o ranking descritivo de RMSE vs Mean/KNN/MissForest sob MCAR/MAR e taxas 5–30%?
- O RMSE do método principal é estável entre taxas? Como se comporta MCAR vs MAR?
- A paralelização gene-nível preserva resultados vs serial?
- Qual a ordem de grandeza do custo wall do grid completo?
- A classificação PAM50 pós-imputação muda o ranking de forma dramática a taxas baixas? (resposta: não)

## Permaneceram abertas

- Superioridade estatística formal vs MissForest sob **protocolo idêntico**.
- RV / preservação de correlação para matrizes TARGET-WISE.
- Generalização além de PAM50 / além de METABRIC; MNAR real.
- F1 com OriginalRFECA **aninhado** no CV.
- Ótimo global de workers (produção 16 vs autotune 8).
- Necessidade do EnsembleSoft vs SVC sozinho (quase empate; CPTAC sem multiclf).
- Utilidade clínica.

---

# Contribuições

## Contribuições metodológicas

- Protocolo **TARGET-WISE** + `leakage_safe` documentado e auditável.
- Separação OriginalRFECA vs legado `RFECA_SVR(k=*)`.
- Contrato de asserções (coverage, fallback, predictor NaNs, mask match).

## Contribuições experimentais

- Freeze METABRIC 40 slots MCAR+MAR × 4 taxas × 5 reps.
- Tabelas/figuras comparison Mean/KNN/MissForest/OriginalRFECA.
- Classificação pós-imputação multi-classificador para RFECA.
- Pacote `final_analysis/` para escrita sem reexecução.

## Contribuições computacionais

- Paralelismo gene-nível methodology-preserving + benchmark de workers.
- Freeze com seeds/hashes/requirements.

## Contribuições práticas

- Evidência de que, em PAM50 METABRIC, seleção supervisionada por gene com preditores completos rivaliza MissForest em RMSE e é mais estável sob MAR/taxa.
- Trade-off explícito: qualidade/estabilidade vs custo RFE/SVR (~51 h no grid).
