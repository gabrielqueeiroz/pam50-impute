# Trabalhos futuros (oportunidades naturais)

1. Replicar Mean/KNN/MissForest sob o **mesmo** `repeated_mask_holdout` + seed v2 para contraste formal.
2. Aninhar imputação OriginalRFECA no CV de classificação (amostra de genes/taxas) para F1 comparável.
3. Calcular RV / erro de correlação para matrizes TARGET-WISE imputadas.
4. Avaliar MNAR e missingness real de plataformas.
5. Extender além de PAM50 (centenas/milhares de genes) com seleção candidata escalável.
6. Comparar RFACA vs RFECA no mesmo protocolo TARGET-WISE.
7. Multiclf completo no CPTAC 2C (se n permitir) ou métricas Bayesianas/small-n.
8. Distilar subconjuntos estáveis de preditores (alta Jaccard) como assinatura interpretável.
9. Otimizar custo: early-stopping RFE, approx. prefixes, caching de correlações.
10. Estudo de sensibilidade a `max_candidates` e `use_scaler` já parcialmente explorado — consolidar na dissertação.
