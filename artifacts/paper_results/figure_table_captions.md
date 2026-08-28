# Definitive figure/table captions (provisional numbering)

Use these numbers when writing Results. File names under `artifacts/paper_results/` are locked to this scheme.

**Protocol (all main items):** METABRIC / CPTAC 2C six-imputer shared-mask campaigns; 10 missingness replications × 5 stratified CV folds; fold means averaged within replication; EnsembleSoft for Macro-F1; seed scheme **legacy**. Inductive RFECA (v2) is Supplementary only.

---

## Main text

### Figure 1 — Reconstruction (RMSE)
**File:** `fig01_metabric_rmse_by_missingness.{pdf,png}`  
**LaTeX label:** `fig:metabric_rmse`

**Caption.** Reconstruction error (RMSE) on METABRIC under MCAR (**A**) and MAR (**B**) as a function of artificial missingness (5–30\%). Lines show means across ten missingness replications; shaded bands are 95\% Student $t$ intervals. Metrics are first averaged over five stratified CV folds within each replication.

**In-text cue.** MissForest lowest RMSE; Mean highest; KNN/RFECA intermediate; error rises with missingness.

### Figure 2 — Structure (RV)
**File:** `fig02_metabric_rv_by_missingness.{pdf,png}`  
**LaTeX label:** `fig:metabric_rv`

**Caption.** Gene–gene correlation preservation (RV coefficient) on METABRIC under MCAR (**A**) and MAR (**B**). Higher values indicate better structural preservation. Means and 95\% CIs across replications as in Figure~1.

**In-text cue.** Structure largely tracks RMSE: MissForest best, Mean worst.

### Figure 3 — Downstream PAM50 (Macro-F1)
**File:** `fig03_metabric_macrof1_by_missingness.{pdf,png}`  
**LaTeX label:** `fig:metabric_f1`

**Caption.** Downstream PAM50 Macro-F1 (EnsembleSoft) on METABRIC under MCAR (**A**) and MAR (**B**). The $y$-axis is restricted to highlight small differences; absolute performance remains high for all methods. Means and 95\% CIs across replications as in Figure~1.

**In-text cue.** Near-ties at low missingness; modest MissForest/RFECA separation mainly at $\ge$20\%; gaps much smaller than for RMSE. Do not read restricted axis as large absolute effects.

### Figure 4 — RMSE versus Macro-F1
**File:** `fig04_rmse_vs_macrof1.{pdf,png}`  
**LaTeX label:** `fig:rmse_vs_f1`

**Caption.** Mean RMSE versus mean Macro-F1 for each method $\times$ missingness rate on METABRIC (MCAR left; MAR right). Point labels indicate missingness percentage. Descriptive Spearman correlations on these aggregated points: overall $\rho=-0.55$; MCAR $\rho=-0.65$; MAR $\rho=-0.52$ (points are not independent).

**In-text cue.** Large reconstruction gains do not map proportionally onto PAM50 gains (descriptive only).

### Table 1 — Compact METABRIC performance
**File:** `table01_compact_metabric.tex`  
**LaTeX label:** `tab:metabric_compact`

**Caption.** Compact METABRIC summary at 5\%, 20\%, and 30\% missingness under MCAR and MAR. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\pm$ SD across 10 missingness replications (fold-averaged within replication).

**In-text cue.** Primary numeric table for the two-regime message (clear RMSE/RV separation; compressed F1).

### Table 2 — Statistical summary (METABRIC)
**File:** `table03_statistical_summary.tex`  
**LaTeX label:** `tab:stats_summary`

**Caption.** METABRIC statistical summary. Friedman omnibus on the reduced method set (Mean, KNN, RFECA-$k$20, MissForest). Pairwise column: Holm-significant primary contrasts versus Mean (matched-pairs rank-biserial $r$). Unit of analysis: replication means ($n=10$).

**In-text cue.** Reconstruction/structure differences significant at all rates; F1 omnibus significance mainly at higher missingness.

---

## Supplementary

### Table S1 — METABRIC MCAR (full rates)
**File:** `table01_metabric_mcar.tex`  
**LaTeX label:** `tab:metabric_mcar`

**Caption.** METABRIC performance under MCAR (mean $\pm$ SD across replications), including the 0\% missingness Macro-F1 baseline. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\pm$ SD across 10 missingness replications (fold-averaged within replication).

### Table S2 — METABRIC MAR (full rates)
**File:** `table02_metabric_mar.tex`  
**LaTeX label:** `tab:metabric_mar`

**Caption.** METABRIC performance under MAR (mean $\pm$ SD across replications), including the 0\% missingness Macro-F1 baseline. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\pm$ SD across 10 missingness replications (fold-averaged within replication).

### Table S3 — CPTAC 2C exploratory
**File:** `table04_cptac2c_summary.tex` (+ `table04_cptac2c_summary.csv`)  
**LaTeX label:** `tab:cptac2c`

**Caption.** CPTAC 2C exploratory external cohort ($n=117$; mean $\pm$ SD). Small $n$ yields high variance; rankings are not used as primary evidence of method superiority. Best value per row in **bold** (lowest RMSE; highest RV and Macro-F1). Entries are mean $\pm$ SD across 10 missingness replications (fold-averaged within replication).

### Table S4 — Inductive RFECA sensitivity
**File:** `table05_rfeca_inductive_sensitivity.tex` (+ full CSV for all cohorts)  
**LaTeX label:** `tab:rfeca_inductive`

**Caption.** Inductive RFECA sensitivity on METABRIC (MCAR and MAR). Legacy six-imputer campaign versus inductive RFECA-only re-run (seed scheme v2). Masks/seeds differ; differences are *not* a pure ablation of inductiveness. Full CPTAC 2C comparisons are in the accompanying CSV.

**In-text cue.** METABRIC Macro-F1 essentially unchanged; do not treat $\Delta$RMSE as causal effect of the inductive fix alone.

### Table S5 — Classification metrics (Macro-F1 / BalAcc / Precision / Recall)
**File:** `table06_metabric_classification.tex` (+ `table06_metabric_classification.csv`)  
**LaTeX label:** `tab:metabric_classification`

**Caption.** METABRIC downstream PAM50 classification metrics (EnsembleSoft) under MCAR and MAR, including the 0\% missingness baseline. Best value per row in **bold** (all metrics higher is better). Entries are mean $\pm$ SD across 10 missingness replications (fold-averaged within replication). Macro-F1 is the primary endpoint; balanced accuracy, macro-precision, and macro-recall are reported for completeness.

**In-text cue.** Companion metrics closely track Macro-F1; use for completeness, not as a second primary claim.

---

## Not numbered in the paper (data only)
- `aggregated_metrics.csv`, `replication_level_metrics.csv`
- `primary_contrasts_statistics.csv`, `full_pairwise_statistics.csv`
- `spearman_rmse_vs_f1.json`, `AUDIT.md`, `results_findings.md`
