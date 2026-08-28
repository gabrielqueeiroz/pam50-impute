# Análise interna do OriginalRFECA

## Features selecionadas
- Média de `n_predictors_selected`: **21.56**
- Mediana: **23.0**
- Desvio-padrão: **3.93**
- Intervalo: [3, 24]
- Prefixo vencedor (mean length): **43.62** (mediana 47.0)

## Estabilidade dos subconjuntos
- Jaccard médio pairwise entre réplicas (mesmo gene×mecanismo×taxa): **0.677**
- Mediana Jaccard: **0.670**
- Detalhe: `rfeca_subset_jaccard_by_gene.csv`

## RMSE por gene
- Média: **0.5637**
- Mediana: **0.5159**
- Std: **0.2762**

## Tempo
- Wall total 40 slots: **51.46 h**
- Wall médio/slot: **4631.0 s**
- Wall/slot/50 genes: **92.6 s** (aproximação)

## Top preditores (frequência como membro do subconjunto vencedor)
Ver `rfeca_top_predictors.csv`.

## Comparação de custo com MissForest
Ver `computational_cost.md` — MissForest é tipicamente mais barato por fold CV; OriginalRFECA concentra custo em RFE/SVR por gene.
