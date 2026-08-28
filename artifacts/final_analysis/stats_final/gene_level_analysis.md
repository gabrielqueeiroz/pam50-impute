# Gene-level analysis (OriginalRFECA)

**Nota:** Mean/KNN/MissForest não têm RMSE por gene nos artefactos finais; não é possível calcular per-gene winners cross-method nem dizer quais genes apenas MissForest melhora sem recalcular imputações.

## Genes mais difíceis (RMSE médio, todos os slots)

| gene | rmse_mean | rmse_std | mae_mean | n | n_pred_mean | rmse_rank |
| --- | --- | --- | --- | --- | --- | --- |
| MMP11 | 1.2767 | 0.0816 | 0.9958 | 40 | 21.6750 | 1 |
| NAT1 | 1.2106 | 0.0964 | 0.9372 | 40 | 23.1000 | 2 |
| MAPT | 1.0643 | 0.0539 | 0.8106 | 40 | 23.2500 | 3 |
| SFRP1 | 0.9991 | 0.1095 | 0.6938 | 40 | 20.0750 | 4 |
| SLC39A6 | 0.9613 | 0.0419 | 0.7646 | 40 | 23.8750 | 5 |
| ESR1 | 0.9581 | 0.0597 | 0.7524 | 40 | 21.8000 | 6 |
| PHGDH | 0.9373 | 0.0517 | 0.7343 | 40 | 23.6750 | 7 |
| CDH3 | 0.8440 | 0.0850 | 0.6232 | 40 | 23.3750 | 8 |
| PGR | 0.8333 | 0.0579 | 0.6365 | 40 | 22.6000 | 9 |
| FGFR4 | 0.8310 | 0.0788 | 0.5649 | 40 | 23.7250 | 10 |
| MYC | 0.8282 | 0.0521 | 0.6585 | 40 | 23.7500 | 11 |
| KRT17 | 0.7894 | 0.0747 | 0.5282 | 40 | 5.7250 | 12 |
| FOXC1 | 0.7410 | 0.0568 | 0.5394 | 40 | 23.3250 | 13 |
| KRT14 | 0.6970 | 0.0896 | 0.4157 | 40 | 21.1250 | 14 |
| MLPH | 0.6629 | 0.0523 | 0.4997 | 40 | 22.6750 | 15 |

## Consistentemente difíceis (Q4 MCAR ∩ Q4 MAR)

CDH3, ESR1, FGFR4, MAPT, MMP11, NAT1, PGR, PHGDH, SFRP1, SLC39A6

## Mais fáceis (menor RMSE)

| gene | rmse_mean | rmse_std | mae_mean | n | n_pred_mean | rmse_rank |
| --- | --- | --- | --- | --- | --- | --- |
| ANLN | 0.3058 | 0.0214 | 0.2284 | 40 | 21.4500 | 41 |
| EXO1 | 0.3036 | 0.0175 | 0.2344 | 40 | 19.1750 | 42 |
| NUF2 | 0.2824 | 0.0334 | 0.1961 | 40 | 21.5750 | 43 |
| CEP55 | 0.2797 | 0.0197 | 0.2124 | 40 | 22.6750 | 44 |
| KIF2C | 0.2579 | 0.0218 | 0.1992 | 40 | 18.7250 | 45 |
| NDC80 | 0.2311 | 0.0194 | 0.1775 | 40 | 22.6000 | 46 |
| MDM2 | 0.2198 | 0.0309 | 0.1530 | 40 | 20.2750 | 47 |
| CDC6 | 0.2127 | 0.0369 | 0.1384 | 40 | 13.7250 | 48 |
| MKI67 | 0.2123 | 0.0182 | 0.1603 | 40 | 23.3500 | 49 |
| MYBL2 | 0.1655 | 0.0105 | 0.1289 | 40 | 13.9750 | 50 |

## Interpretabilidade — top preditores

| gene | count_as_predictor |
| --- | --- |
| MKI67 | 1427 |
| NDC80 | 1376 |
| UBE2T | 1341 |
| CEP55 | 1340 |
| PTTG1 | 1301 |
| KIF2C | 1289 |
| UBE2C | 1260 |
| EXO1 | 1254 |
| RRM2 | 1170 |
| CCNE1 | 1154 |
| MELK | 1144 |
| CDC20 | 1139 |
| ANLN | 1127 |
| NUF2 | 1060 |
| CCNB1 | 1052 |

Tamanho médio do subconjunto: **21.56** (mediana 23.0, intervalo [3, 24])

Jaccard médio entre réplicas: **0.677** (mediana 0.670)
