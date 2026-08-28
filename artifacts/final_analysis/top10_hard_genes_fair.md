# Top 10 genes mais difíceis (fair holdout)

Dificuldade = média do RMSE entre Mean, KNN, MissForest e OriginalRFECA (média sobre MCAR/MAR × 5/10/20/30% × 5 réplicas).

★ = melhor método (menor RMSE).

| Rank | Gene | Difficulty | Mean | KNN | MissForest | OriginalRFECA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | NAT1 | 1.441 | 1.972 | 1.342 | 1.238 | ★ 1.211 |
| 2 | MMP11 | 1.306 | 1.389 | 1.328 | ★ 1.229 | 1.277 |
| 3 | ESR1 | 1.265 | 2.173 | 1.023 | ★ 0.907 | 0.958 |
| 4 | MAPT | 1.220 | 1.565 | 1.171 | 1.080 | ★ 1.064 |
| 5 | SLC39A6 | 1.195 | 1.656 | 1.147 | 1.018 | ★ 0.961 |
| 6 | SFRP1 | 1.180 | 1.680 | 1.087 | ★ 0.955 | 0.999 |
| 7 | PHGDH | 1.077 | 1.356 | 1.051 | 0.963 | ★ 0.937 |
| 8 | KRT17 | 1.055 | 1.659 | 1.012 | ★ 0.762 | 0.789 |
| 9 | CDH3 | 0.960 | 1.294 | 0.893 | ★ 0.810 | 0.844 |
| 10 | MLPH | 0.960 | 1.722 | 0.769 | 0.686 | ★ 0.663 |

Fonte: `fair_gene_comparison_long.csv` · protocolo `repeated_mask_holdout` (máscaras do freeze).
