# A) Comparações formalmente válidas (baselines)
**Âmbito:** METABRIC · Mean / KNN / MissForest · mesmo protocolo CV shared-mask.
**Excluídos:** RFECA-k* e OriginalRFECA (protocolos distintos).
**Unidade:** média dos 5 folds por réplica (n=10).
**Testes:** Friedman; Wilcoxon signed-rank pareado; Holm por (mech×rate×metric); bootstrap IC95% do Δ pareado; rank-biserial.

## Métricas
- `rmse`, `mae`, `corr_rv`, `f1_macro`

## Artefactos
- `A_friedman_baselines_metabric.csv`
- `A_wilcoxon_holm_baselines_metabric.csv`
- `A_primary_contrasts_baselines.csv`

## Destaque RMSE — MissForest vs Mean (Holm)
| Mech | Rate | Δ (MF−Mean)* | IC95% | p_Holm |
|---|---:|---:|---|---:|
| MAR | 5% | -0.5398 | [-0.5499, -0.5298] | 0.005859 |
| MAR | 10% | -0.5174 | [-0.5223, -0.5122] | 0.005859 |
| MAR | 20% | -0.4770 | [-0.4823, -0.4725] | 0.005859 |
| MAR | 30% | -0.4377 | [-0.4414, -0.4336] | 0.005859 |
| MCAR | 5% | -0.4495 | [-0.4602, -0.4381] | 0.005859 |
| MCAR | 10% | -0.4397 | [-0.4506, -0.4300] | 0.005859 |
| MCAR | 20% | -0.4304 | [-0.4350, -0.4258] | 0.005859 |
| MCAR | 30% | -0.4156 | [-0.4190, -0.4118] | 0.005859 |

\*Δ negativo favorece MissForest.
