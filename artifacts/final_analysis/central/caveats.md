# Caveats — comparação OriginalRFECA vs Mean / KNN / MissForest

Esta pasta compara **apenas** estes quatro métodos. RFECA-k5/k10/k20 foram excluídos de propósito.

## Protocolos distintos

| | Baselines (Mean/KNN/MissForest) | OriginalRFECA |
|---|---|---|
| Avaliação imputação | Stratified 5-fold CV, shared-mask | `repeated_mask_holdout` TARGET-WISE |
| n réplicas | 10 | 5 |
| Seed missingness | legacy | v2 |
| Classificação | imputer **dentro** do CV | pós-imputação (identity no CV) |
| RV | sim | n/a no freeze |

Consequência: deltas e “vitórias” são **descritivos**. Não há Wilcoxon pareado válido entre OriginalRFECA e MissForest.

## O que é seguro afirmar

- Dentro do freeze OriginalRFECA: RMSE/MAE e F1 entre taxas/mecanismos.
- Dentro da campanha baselines: Mean vs KNN vs MissForest (mesmo protocolo).
- Ranking descritivo OriginalRFECA vs baselines, com o caveat acima.

## O que evitar

- p-values confirmatórios OriginalRFECA vs MissForest.
- Afirmar que F1 é diretamente comparável sem notar o nesting diferente.
- Reintroduzir k5/k10/k20 na narrativa principal.
