# Win counts (RMSE descritivo · 4 métodos)

## Por cenário

| mechanism | rate_pct | winner |
| --- | --- | --- |
| MCAR | 5 | OriginalRFECA |
| MCAR | 10 | OriginalRFECA |
| MCAR | 20 | OriginalRFECA |
| MCAR | 30 | OriginalRFECA |
| MAR | 5 | MissForest |
| MAR | 10 | OriginalRFECA |
| MAR | 20 | OriginalRFECA |
| MAR | 30 | OriginalRFECA |

## Por mecanismo

| mechanism | best | wins |
| --- | --- | --- |
| MAR | MissForest | 1 |
| MAR | OriginalRFECA | 3 |
| MCAR | OriginalRFECA | 4 |

## Por taxa

| rate_pct | best | wins |
| --- | --- | --- |
| 5 | MissForest | 1 |
| 5 | OriginalRFECA | 1 |
| 10 | OriginalRFECA | 2 |
| 20 | OriginalRFECA | 2 |
| 30 | OriginalRFECA | 2 |

## Totais RMSE

| method | wins | n_scenarios |
| --- | --- | --- |
| OriginalRFECA | 7 | 8 |
| MissForest | 1 | 8 |

## Totais Macro-F1 (EnsembleSoft)

| method | wins | metric | n_scenarios |
| --- | --- | --- | --- |
| OriginalRFECA | 6 | f1_macro | 8 |
| MissForest | 2 | f1_macro | 8 |
