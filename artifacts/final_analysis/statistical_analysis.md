# Análise estatística

## Pode-se fazer Wilcoxon pareado OriginalRFECA vs MissForest?
**NÃO** — formalmente inválido para inferência confirmatória.

Motivos:
1. Protocolos diferentes: TARGET-WISE mask-holdout (5 reps, seed v2) vs imputer-within-CV shared-mask (10 reps, seed legacy).
2. Unidades experimentais não emparelhadas (máscaras/seeds distintos).
3. As tabelas Wilcoxon/Holm existentes (`artifacts/stats_mcar_mar_20260727_160425/`) referem-se a **RFECA_SVR(k=*) legado**, não ao OriginalRFECA TARGET-WISE.

Existing Wilcoxon/Holm tables under stats/ compare legacy RFECA_SVR(k=*) within the six-imputer shared-mask CV campaign — NOT OriginalRFECA TARGET-WISE.

## O que foi calculado (descritivo)
Welch t não pareado + Cohen's d + IC bootstrap 95% da diferença de médias de RMSE
(`statistical_rfeca_vs_missforest_descriptive.csv`). Interpretar apenas como magnitude descritiva.

## Resumo das diferenças (RMSE: OriginalRFECA − MissForest)

| Mecanismo | Taxa | Δ média | Cohen's d | IC95 boot Δ | Welch p (descritivo) |
|---|---:|---:|---:|---|---:|
| MCAR | 5% | -0.0046 | -0.295 | [-0.0186, +0.0089] | 0.5552 |
| MCAR | 10% | -0.0090 | -1.124 | [-0.0163, -0.0012] | 0.07042 |
| MCAR | 20% | -0.0229 | -2.705 | [-0.0296, -0.0162] | 3.429e-05 |
| MCAR | 30% | -0.0319 | -7.671 | [-0.0360, -0.0279] | 5.66e-07 |
| MAR | 5% | +0.0123 | +1.109 | [-0.0003, +0.0256] | 0.144 |
| MAR | 10% | -0.0040 | -0.449 | [-0.0143, +0.0056] | 0.5073 |
| MAR | 20% | -0.0324 | -3.954 | [-0.0394, -0.0257] | 1.17e-06 |
| MAR | 30% | -0.0658 | -7.171 | [-0.0723, -0.0592] | 3.311e-09 |

**Leitura:** valores negativos de Δ favorecem OriginalRFECA (menor RMSE). Em MCAR o Δ é tipicamente pequeno/negativo; em MAR o OriginalRFECA ganha de forma mais consistente em magnitude.
