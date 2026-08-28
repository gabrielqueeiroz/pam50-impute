# One-page Results — METABRIC PAM50

Comparação: **Mean · KNN · MissForest · OriginalRFECA** (sem RFECA-k*). Fonte: artefactos congelados.

---

## Principais números

- **RMSE:** OriginalRFECA melhor em **7/8** células (MCAR/MAR × 5/10/20/30%); única perda: **MAR 5%** (MissForest 0.629 vs 0.641).
- **Macro-F1 (EnsembleSoft):** OriginalRFECA melhor em **6/8**; MissForest à frente em MCAR/MAR 10% (margem ≤0.002).
- **ΔRMSE vs MissForest (descritivo):** até **−0.066** (−9.3%) em MAR 30%; positivo só em MAR 5% (+0.012).
- **Estabilidade OriginalRFECA:** RMSE quase flat nas taxas; MAR ~+0.02–0.03 vs MCAR.
- **Seleção:** ~**21.6** preditores/gene; Jaccard réplicas ~**0.68**; top preditores: MKI67, NDC80, UBE2T, CEP55, PTTG1.
- **Operacional:** 40/40 slots class A; svr_coverage=1.0; fallback=0; wall ~**51.5 h** (16 gene-workers).

## Win counts (RMSE)

| method | wins | n_scenarios |
| --- | --- | --- |
| OriginalRFECA | 7 | 8 |
| MissForest | 1 | 8 |

## Headline RMSE / F1

| mechanism | rate_pct | Mean_RMSE | KNN_RMSE | MissForest_RMSE | OriginalRFECA_RMSE | Mean_F1 | KNN_F1 | MissForest_F1 | OriginalRFECA_F1 | rmse_best | f1_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCAR | 5 | 1.067 ± 0.016 | 0.680 ± 0.016 | 0.617 ± 0.017 | 0.613 ± 0.012 | 0.879 ± 0.005 | 0.881 ± 0.003 | 0.881 ± 0.004 | 0.882 ± 0.003 | OriginalRFECA | OriginalRFECA |
| MCAR | 10 | 1.063 ± 0.019 | 0.691 ± 0.010 | 0.624 ± 0.008 | 0.615 ± 0.008 | 0.875 ± 0.004 | 0.876 ± 0.003 | 0.879 ± 0.005 | 0.877 ± 0.003 | OriginalRFECA | MissForest |
| MCAR | 20 | 1.068 ± 0.009 | 0.710 ± 0.009 | 0.637 ± 0.010 | 0.614 ± 0.005 | 0.866 ± 0.004 | 0.869 ± 0.005 | 0.874 ± 0.005 | 0.876 ± 0.007 | OriginalRFECA | OriginalRFECA |
| MCAR | 30 | 1.069 ± 0.008 | 0.734 ± 0.007 | 0.653 ± 0.004 | 0.621 ± 0.004 | 0.855 ± 0.007 | 0.858 ± 0.006 | 0.866 ± 0.005 | 0.872 ± 0.006 | OriginalRFECA | OriginalRFECA |
| MAR | 5 | 1.169 ± 0.014 | 0.700 ± 0.009 | 0.629 ± 0.009 | 0.641 ± 0.015 | 0.881 ± 0.005 | 0.880 ± 0.003 | 0.881 ± 0.004 | 0.882 ± 0.005 | MissForest | OriginalRFECA |
| MAR | 10 | 1.160 ± 0.011 | 0.710 ± 0.007 | 0.643 ± 0.007 | 0.639 ± 0.011 | 0.874 ± 0.005 | 0.878 ± 0.002 | 0.879 ± 0.003 | 0.879 ± 0.005 | OriginalRFECA | MissForest |
| MAR | 20 | 1.149 ± 0.009 | 0.751 ± 0.013 | 0.672 ± 0.009 | 0.640 ± 0.005 | 0.864 ± 0.004 | 0.872 ± 0.005 | 0.871 ± 0.006 | 0.877 ± 0.007 | OriginalRFECA | OriginalRFECA |
| MAR | 30 | 1.143 ± 0.008 | 0.857 ± 0.022 | 0.705 ± 0.011 | 0.639 ± 0.002 | 0.848 ± 0.010 | 0.848 ± 0.006 | 0.861 ± 0.006 | 0.872 ± 0.004 | OriginalRFECA | OriginalRFECA |

## Effect sizes descritivos vs MissForest

| mechanism | rate_pct | ΔRMSE | ΔRMSE_pct | IC95_RMSE | ΔMAE | ΔMAE_pct | IC95_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MCAR | 5 | -0.0046 | -0.75% | [-0.0185, 0.0086] | -0.0095 | -2.29% | [-0.0170, -0.0017] |
| MCAR | 10 | -0.009 | -1.44% | [-0.0164, -0.0012] | -0.0099 | -2.37% | [-0.0136, -0.0059] |
| MCAR | 20 | -0.0229 | -3.59% | [-0.0299, -0.0160] | -0.0205 | -4.80% | [-0.0236, -0.0173] |
| MCAR | 30 | -0.0319 | -4.89% | [-0.0360, -0.0279] | -0.0272 | -6.23% | [-0.0297, -0.0248] |
| MAR | 5 | 0.0123 | 1.96% | [0.0001, 0.0258] | -0.0053 | -1.24% | [-0.0104, -0.0004] |
| MAR | 10 | -0.004 | -0.62% | [-0.0137, 0.0059] | -0.0122 | -2.83% | [-0.0171, -0.0064] |
| MAR | 20 | -0.0324 | -4.82% | [-0.0395, -0.0258] | -0.0260 | -5.84% | [-0.0292, -0.0230] |
| MAR | 30 | -0.0658 | -9.33% | [-0.0723, -0.0591] | -0.0465 | -9.99% | [-0.0498, -0.0429] |

*Sem p-values: protocolos distintos (TARGET-WISE holdout vs CV shared-mask).*

## Genes mais difíceis (OriginalRFECA)

| gene | rmse_mean | rmse_std | n_pred_mean |
| --- | --- | --- | --- |
| MMP11 | 1.2767 | 0.0816 | 21.6750 |
| NAT1 | 1.2106 | 0.0964 | 23.1000 |
| MAPT | 1.0643 | 0.0539 | 23.2500 |
| SFRP1 | 0.9991 | 0.1095 | 20.0750 |
| SLC39A6 | 0.9613 | 0.0419 | 23.8750 |
| ESR1 | 0.9581 | 0.0597 | 21.8000 |
| PHGDH | 0.9373 | 0.0517 | 23.6750 |
| CDH3 | 0.8440 | 0.0850 | 23.3750 |

## Top preditores selecionados

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

---

## Figuras correspondentes

| Figura | Path |
|---|---|
| RMSE por missingness | `figures/fig_rmse_by_missingness.png` |
| MAE | `figures/fig_mae_by_missingness.png` |
| Macro-F1 | `figures/fig_macrof1_by_missingness.png` |
| RMSE bars | `figures/copied_comparison_fig05_rmse_bars.png` |
| F1 bars | `figures/copied_comparison_fig06_f1_bars.png` |
| RV baselines | `figures/fig_rv_by_missingness_baselines.png` |

## Tabelas correspondentes

| Tabela | Path |
|---|---|
| Headline RMSE+F1 | `central/headline_rmse_f1.csv` |
| Comparison display | `central/comparison_display.csv` |
| Wins | `stats_final/win_*.csv` |
| Effect sizes | `stats_final/effect_sizes_rfeca_vs_missforest.csv` |
| Stats A (válidas) | `stats_final/A_*.csv` |
| Stats B (descritivas) | `stats_final/B_*.csv` |
| Gene difficulty | `stats_final/gene_difficulty_overall_original_rfeca.csv` |

---

## Frases objetivas para Results

1. Across eight METABRIC PAM50 missingness settings (MCAR/MAR × 5–30%), OriginalRFECA achieved the lowest mean RMSE in seven settings; MissForest was best only at MAR 5%.

2. Relative to MissForest, OriginalRFECA reduced RMSE by up to 9.3% (MAR 30%; Δ=−0.066, descriptive bootstrap IC95% excluding zero); at MAR 5% the descriptive Δ favored MissForest (Δ=+0.012).

3. OriginalRFECA RMSE remained nearly flat across missingness rates, whereas KNN and MissForest degraded more under MAR at higher rates.

4. Macro-F1 differences among imputers were small at 5–10% missingness and more favorable to OriginalRFECA at 20–30%, with the caveat that classification nesting differs between protocols.

5. Within the shared-mask CV baseline protocol, Friedman and Wilcoxon–Holm tests confirmed systematic RMSE differences among Mean, KNN, and MissForest (MissForest best among the three).

6. OriginalRFECA selected on average ~22 predictors per gene (median 23); pairwise Jaccard of selected subsets across replicates was ~0.68, indicating moderate stability.

7. The hardest genes to impute under OriginalRFECA included MMP11, NAT1, MAPT, SFRP1, SLC39A6 (highest mean RMSE across slots).

8. No confirmatory p-values are reported for OriginalRFECA versus baselines because evaluation protocols, replicate counts, and missingness seeds differ.
