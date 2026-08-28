# Principais achados (respostas objetivas)

1. **Principal resultado:** OriginalRFECA TARGET-WISE alcança RMSE competitivo/superior a MissForest/KNN/Mean no METABRIC PAM50, com estabilidade excepcional sob variação de taxa e robustez relativa em MAR; 40/40 slots A, sem fallback.
2. **Superou o SOTA interno?** Em RMSE, sim em vários cenário (especialmente MAR e taxas altas); empatado/ligeiramente melhor em MCAR baixo vs MissForest. Em F1, vantagem mais clara a 20–30%.
3. **Cenários:** MAR (todas as taxas) e MCAR a taxas ≥20%; menor diferenciação a MCAR 5–10% vs MissForest.
4. **Magnitude:** ver `results_tables.csv` colunas `OriginalRFECA_minus_MissForest` e `OriginalRFECA_pct_vs_*`. Tipicamente poucos centésimos de RMSE vs MissForest; dezenas de centésimos vs Mean (~40%+ redução relativa vs Mean).
5. **Dependência da taxa:** baselines sim (pioram); OriginalRFECA pouco.
6. **Dependência do mecanismo:** sim — MAR ~0.02–0.03 RMSE pior que MCAR para OriginalRFECA; baselines degradam mais em MAR.
7. **Custo justificável?** Para dissertação/evidência metodológica sim; para produção em tempo real depende — 51.5 h no grid completo nesta máquina.
8. **Limitações:** protocolos não idênticos vs baselines; sem RV; F1 nesting diferente; n_reps=5; PAM50 only.
9. **Ameaças:** ver `threats_to_validity.md`.
10. **Futuro:** ver `future_work.md`.
