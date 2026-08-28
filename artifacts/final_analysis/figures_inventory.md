# Figures inventory

| figura | objetivo | mensagem principal | seção do artigo |
|---|---|---|---|
| `figures/fig_rmse_by_missingness.png` | Comparar RMSE vs taxa/mecanismo | OriginalRFECA estável e tipicamente mais baixo; MF único rival | Results — imputação |
| `figures/fig_mae_by_missingness.png` | Idem para MAE | Mesmo padrão do RMSE | Results — imputação |
| `figures/fig_rmse_bars_5_10_20_30.png` / `copied_comparison_fig05_*` | Barras RMSE por célula | Ranking visual Mean≫KNN>MF≳RFECA | Results |
| `figures/fig_macrof1_by_missingness.png` / `copied_comparison_fig03_*` | Macro-F1 vs missingness | Diferenças pequenas a 5–10%; RFECA melhor a 20–30% (caveat) | Results — classificação |
| `figures/copied_comparison_fig06_f1_bars.png` | Barras F1 | Comparação F1 lado a lado | Results — classificação |
| `figures/fig_rv_by_missingness_baselines.png` | RV só baselines | MF preserva melhor correlação entre baselines; RFECA sem RV | Results / Discussion |
| `figures/fig_stability_rmse_cv.png` | Estabilidade entre réplicas | Dispersão por método/taxa | Results / Supplement |
| `figures/gene_method_heatmap_rmse.png` | RMSE gene×método | Dificuldade por gene (RFECA); baselines NaN | Results — gene-level |
| `figures/gene_method_heatmap_rmse_zscore.png` | z-score por gene | Relativo entre métodos (indefinido se <2 métodos) | Supplement / Results |
| flowchart OriginalRFECA (paper_results_original_rfeca) | Protocolo TARGET-WISE | Leakage-safe pipeline | Methods |
| parallel speedup (computational_cost / parallel_benchmark) | Custo e paralelismo | Speedup com fingerprints idênticos | Methods / Supplement |

Canvas auxiliar: `central-results-comparison.canvas.tsx` (exploração, não figura do artigo).
