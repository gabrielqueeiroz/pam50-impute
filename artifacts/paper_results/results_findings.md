# Results findings (tied to generated outputs)

## Reconstruction accuracy
- **Statistically supported (METABRIC):** MissForest has the lowest RMSE/MAE at all rates under MCAR and MAR; Mean has the highest.
- **Descriptively consistent:** KNN and RFECA($k$) occupy an intermediate tier; larger RFECA $k$ often helps RMSE but remains above MissForest.
- Ranking of best/worst methods is stable as missingness increases; mid-tier swaps are secondary.

## Structural preservation
- **Statistically supported:** MissForest best, Mean worst on RV; Friedman significant at all rates.
- Structure largely agrees with RMSE for headline methods; mild mid-tier RMSE–RV disagreements are descriptively observed.

## Downstream classification
- Absolute Macro-F1 remains high for all methods.
- **Statistically supported:** near-ties at ~5% (Friedman often n.s.); modest MissForest/RFECA advantages vs Mean at ~20–30% after Holm in primary contrasts.
- **Unsupported:** RFECA superiority over MissForest on F1.
- Reconstruction improvements do **not** translate proportionally into F1 gains (Figure 4; descriptive Spearman overall $\rho=-0.55$).

## MCAR versus MAR
- **Directionally and statistically consistent:** same qualitative ordering and two-regime message under both mechanisms.
- MAR can make Mean look somewhat worse on reconstruction/structure; it does not reorder methods.

## Statistical evidence
- Source: `artifacts/stats_mcar_mar_20260727_160425`.
- Friedman (reduced set) + Wilcoxon–Holm primary family + rank-biserial + bootstrap CIs on paired deltas.
- Unit of analysis: fold-averaged replication means ($n=10$).

## CPTAC 2C exploratory validation
- Reconstruction direction matches METABRIC (MissForest best / Mean worst) **descriptively**.
- F1 rankings are high-variance / unstable — **exploratory only**.

## Inductive RFECA sensitivity
- Separate artifacts (`*_full_rfeca_*`, seed scheme v2); **not mixed** into main figures.
- METABRIC MCAR F1 changes are negligible relative to main conclusions; RMSE comparisons are confounded by mask changes.
- Label: **non-causal / not a pure ablation**.

## Main vs supplementary recommendation
**Main paper:** Figures 1–4; Table 1 (compact METABRIC); Table 2 (stats summary).  
**Supplementary:** Table S1–S2 (full METABRIC MCAR/MAR); Table S3 (CPTAC 2C); Table S4 (inductive sensitivity); Table S5 (classification metrics); pairwise CSVs if needed.

See `figure_table_captions.md` for locked provisional numbering and final captions.
