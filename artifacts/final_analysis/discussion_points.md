# Insights para a Discussão (tópicos científicos)

## Por que o OriginalRFECA pode superar / empatar MissForest em RMSE?
- Preditores TARGET-WISE usam a matriz completa original → evita erro em cascata de chaining.
- Seleção gene-específica (prefixos de correlação + RFE) adapta o subconjunto ao alvo.
- Em MAR, métodos globais (Mean/KNN) degradam; RFECA mantém RMSE quase plano entre taxas.
- MissForest modela dependências multivariadas bem em MCAR, mas sofre mais sob MAR estruturado.

## Por que pouca diferença de RMSE do OriginalRFECA entre 10%, 20% e 30%?
- Com preditores sempre completos, aumentar a fração mascarada no *alvo* reduz n de treino do SVR, mas a estrutura de preditores permanece.
- Span MCAR RMSE across rates: {'min': 0.6126294477322903, 'max': 0.6212641927187249, 'range': 0.008634744986434573, 'cv': 0.00618888341814785}.
- Span MAR: {'min': 0.6388336206778934, 'max': 0.6414245784651376, 'range': 0.0025909577872441636, 'cv': 0.0017247697053176855}.

## MCAR vs MAR
- OriginalRFECA: MAR sistematicamente ~0.02–0.03 acima de MCAR (pior), mas estável nas taxas.
- MissForest/KNN: degradação mais acentuada em MAR alto (especialmente KNN @ 30%).
- Mean: RMSE alto e relativamente plano (já saturado).

## Características do METABRIC que favorecem
- n=1608, 50 genes PAM50, classes razoavelmente representadas → CV/F1 estáveis.
- Correlações entre genes PAM50 informativas para seleção supervisionada por alvo.

## Limitações do protocolo TARGET-WISE
- Não é imputação multivariada simultânea; cada gene é um problema univariado condicional.
- RV / estrutura de correlação global não foi a métrica principal do freeze.
- Classificação PAM50 pós-imputação não aninha OriginalRFECA no CV (custo).
- Comparação com baselines usa protocolos de avaliação distintos.

## Vantagens do protocolo
- Leakage-safe por construção no desenho alvo/preditores.
- Sem SimpleImputer/chaining; fallback=0 nos 40 slots.
- Reprodutível (seeds v2 + mask hashes).
- Robustez sob MAR.

## Implicações para imputação de expressão gênica
- Em painéis correlacionados (PAM50), seleção supervisionada por gene + preditores completos pode rivalizar ensembles genéricos.
- Em n pequeno (CPTAC 2C), F1 é frágil — priorizar RMSE/estrutura.
- Custo RFE/SVR é o trade-off central vs MissForest.
