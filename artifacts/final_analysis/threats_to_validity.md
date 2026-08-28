# Ameaças à validade

## Interna
- Protocolos de avaliação distintos entre OriginalRFECA e baselines (mask-holdout 5 reps vs CV 10 reps).
- Classificação RFECA pós-imputação vs baselines com imputer no CV.
- Freeze usa seed scheme v2; baselines do paper usam legacy — máscaras não compartilhadas.

## Externa
- METABRIC PAM50 (50 genes) pode não generalizar a transcriptomas densos.
- CPTAC 2C (n=117) mostra que F1 é instável em dados limitados.
- Apenas mecanismos MCAR/MAR simulados; MNAR real não avaliado.

## Construct
- RMSE em células mascaradas ≠ erro preditivo clínico.
- Macro-F1 PAM50 depende do classificador (EnsembleSoft).
- RV ausente para OriginalRFECA no freeze.

## Conclusão estatística
- Não usar p-values Welch/descritivos como evidência confirmatória.
- Stats Wilcoxon/Holm do pacote `stats/` referem-se a RFECA-k* legado.
