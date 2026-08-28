# Gene difficulty report

## Disponibilidade dos dados

Nos artefactos finais, **apenas OriginalRFECA** possui RMSE/MAE por gene (`per_gene_all_*.csv`). Mean, KNN e MissForest reportam apenas métricas agregadas por fold/réplica. Colunas desses métodos no heatmap são **NaN explícitos** (sem interpolação).

Consequência: rankings cross-method, Δ melhor−pior e CV entre métodos **por gene** não são calculáveis a partir dos artefactos finais. A classificação abaixo usa a dificuldade sob OriginalRFECA e, quando aplicável, vitórias descritivas ao nível de cenário (não gene).

## RMSE / MAE por gene (OriginalRFECA)

### Mais difíceis (top 15)

| gene | RMSE | MAE | RMSE_std | n_pred_mean |
| --- | --- | --- | --- | --- |
| MMP11 | 1.2767 | 0.9958 | 0.0816 | 21.6750 |
| NAT1 | 1.2106 | 0.9372 | 0.0964 | 23.1000 |
| MAPT | 1.0643 | 0.8106 | 0.0539 | 23.2500 |
| SFRP1 | 0.9991 | 0.6938 | 0.1095 | 20.0750 |
| SLC39A6 | 0.9613 | 0.7646 | 0.0419 | 23.8750 |
| ESR1 | 0.9581 | 0.7524 | 0.0597 | 21.8000 |
| PHGDH | 0.9373 | 0.7343 | 0.0517 | 23.6750 |
| CDH3 | 0.8440 | 0.6232 | 0.0850 | 23.3750 |
| PGR | 0.8333 | 0.6365 | 0.0579 | 22.6000 |
| FGFR4 | 0.8310 | 0.5649 | 0.0788 | 23.7250 |
| MYC | 0.8282 | 0.6585 | 0.0521 | 23.7500 |
| KRT17 | 0.7894 | 0.5282 | 0.0747 | 5.7250 |
| FOXC1 | 0.7410 | 0.5394 | 0.0568 | 23.3250 |
| KRT14 | 0.6970 | 0.4157 | 0.0896 | 21.1250 |
| MLPH | 0.6629 | 0.4997 | 0.0523 | 22.6750 |

### Mais fáceis (bottom 10)

| gene | RMSE | MAE | RMSE_std | n_pred_mean |
| --- | --- | --- | --- | --- |
| ANLN | 0.3058 | 0.2284 | 0.0214 | 21.4500 |
| EXO1 | 0.3036 | 0.2344 | 0.0175 | 19.1750 |
| NUF2 | 0.2824 | 0.1961 | 0.0334 | 21.5750 |
| CEP55 | 0.2797 | 0.2124 | 0.0197 | 22.6750 |
| KIF2C | 0.2579 | 0.1992 | 0.0218 | 18.7250 |
| NDC80 | 0.2311 | 0.1775 | 0.0194 | 22.6000 |
| MDM2 | 0.2198 | 0.1530 | 0.0309 | 20.2750 |
| CDC6 | 0.2127 | 0.1384 | 0.0369 | 13.7250 |
| MKI67 | 0.2123 | 0.1603 | 0.0182 | 23.3500 |
| MYBL2 | 0.1655 | 0.1289 | 0.0105 | 13.9750 |

## Respostas pedidas

### Consistentemente difíceis para todos os métodos?

**Não verificável** nos artefactos finais (falta RMSE gene-level dos baselines). Sob OriginalRFECA, genes no quartil superior de dificuldade (Q4) incluem: MMP11, NAT1, MAPT, SFRP1, SLC39A6, ESR1, PHGDH, CDH3, PGR, FGFR4, MYC, KRT17, FOXC1. Estes são os melhores candidatos a 'difíceis', mas não se pode afirmar que Mean/KNN/MissForest falham nos mesmos genes sem os dados.

### Particularmente favorecidos pelo OriginalRFECA?

**Não calculável ao nível do gene** (sem RMSE gene-level de MissForest/KNN/Mean). Ao nível de **cenário** (descritivo): OriginalRFECA vence RMSE em 7/8 células; maiores ganhos vs MissForest em MAR 20–30% e MCAR 20–30% (`stats_final/effect_sizes_rfeca_vs_missforest.csv`).

### Particularmente favorecidos pelo MissForest?

**Não calculável ao nível do gene.** Ao nível de cenário: MissForest tem menor RMSE médio apenas em **MAR 5%** (Δ RFECA−MF ≈ +0.012). Em Macro-F1, MissForest lidera por margem mínima em MCAR/MAR 10%.

### Praticamente indiferentes ao método?

**Não calculável cross-method por gene.** Proxy sob OriginalRFECA: genes com menor CV do RMSE ao longo de mecanismo×taxa (dificuldade estável, não indiferença entre imputadores): SLC39A6, CXXC5, MAPT, CENPF, MYC, PGR, UBE2C, UBE2T, FOXC1, MYBL2.

## Artefacto tabular

`gene_method_heatmap.csv` — matriz gene×método com NaNs explícitos.
