# Manuscript numerics guide (final package)

Source: `artifacts/paper_results/` + `artifacts/stats_mcar_mar_20260727_160425`  
Grain: replication means (n=10); EnsembleSoft for classification.  
**No draft Results `.tex`/`.md` was found in the repo** — insertion lines below are carrier sentences keyed to your subsection outline (replace with your draft wording when available).

Full raw extract: `manuscript_numerics_extract.txt`

---

## 1. Reconstruction accuracy

### Exact values worth citing

| Item | Value |
|---|---|
| Lowest RMSE | **0.617** (MissForest, METABRIC MCAR, 5%) |
| Highest RMSE | **1.169** (Mean, METABRIC MAR, 5%) |
| MF vs Mean reduction MCAR | 5%: **42.1%**; 10%: **41.4%**; 20%: **40.3%**; 30%: **38.9%** |
| MF vs Mean reduction MAR | 5%: **46.2%**; 10%: **44.6%**; 20%: **41.5%**; 30%: **38.3%** |
| Largest consecutive RMSE jump | **+0.106** (KNN, MAR, 20%→30%: 0.751→0.857) |
| Ranking stability | **MissForest always best; Mean always worst.** Mid-tier swaps only (e.g. RFECA-k10 vs k20; KNN 2nd under MAR until 30%). |
| Effect size vs Mean | rank-biserial **r = −1.00** for MissForest and RFECA-k20 at **all** rates (MCAR & MAR) |
| ΔRMSE MF−Mean (bootstrap CI95) | MCAR 5%: **−0.450 [−0.460, −0.438]**; MAR 5%: **−0.540 [−0.550, −0.530]** |
| Friedman (reduced) | all rates **p ≈ 1.4×10⁻⁶ – 5.6×10⁻⁶** (χ² ≈ 27–30) |
| Holm pairwise vs Mean | **all** primary RMSE contrasts significant |

### Recommend in text (3–5 numbers)
1. RMSE span endpoints: **0.617 vs 1.169**
2. Reduction band: **~39–46%** (or cite 5% MCAR **42%** and 5% MAR **46%**)
3. One CI: MAR 5% ΔRMSE = **−0.54 [−0.55, −0.53]**
4. Effect size: **r = −1.00**
5. Optional: KNN MAR collapse **+0.11** from 20→30%

**Carrier sentences (outline §1):**
- “MissForest achieved the lowest RMSE (**0.617** under MCAR at 5%), whereas mean imputation reached the highest (**1.169** under MAR at 5%; Fig. 1, Table 1).”
- “Relative to mean imputation, MissForest reduced RMSE by **42%** (MCAR, 5%) to **39%** (MCAR, 30%), and by **46%** to **38%** under MAR.”
- “Friedman tests were significant at every rate (**p < 10⁻⁵**), with Holm-adjusted contrasts versus Mean yielding maximal paired effect sizes (**r = −1.00**).”

---

## 2. Transcriptomic structure (RV)

| Item | Value |
|---|---|
| Highest RV | **0.9995 ≈ 1.000** (MissForest, MCAR, 5%; table rounds to 1.000) |
| Lowest RV | **0.966** (Mean, MAR, 30%) |
| RV loss 5→30% MissForest | MCAR **0.0037**; MAR **0.0044** |
| RV loss 5→30% Mean | MCAR **0.019**; MAR **0.031** |
| MF−Mean gap | 5%: **+0.001–0.002**; 30%: MCAR **+0.016**, MAR **+0.029** |
| Ranking | MissForest always best; Mean always worst. Mid-tier: KNN often ≈ RFECA-k20 (mild RMSE↔RV inversions). |
| Friedman | all rates **p ≈ 2×10⁻⁶ – 6×10⁻⁶** |
| Holm vs Mean | MF & RFECA-k20 **r = +1.00**, all rates significant |

### Recommend in text
1. Near-ceiling MF RV at 5%: **≈1.000**
2. Mean collapse at MAR 30%: **0.966**
3. MF vs Mean gap at 30% MAR: **+0.029**
4. Friedman **p < 10⁻⁵**, **r = +1.00**

**Carrier (§2):**
- “Structural preservation tracked reconstruction: MissForest retained RV near **1.00** at 5% missingness, while Mean fell to **0.966** under MAR at 30% (Fig. 2).”
- “The MissForest–Mean RV gap widened from **~0.002** at 5% to **0.029** at 30% MAR.”

---

## 3. Downstream classification

### Macro-F1 (primary)

| Item | Value |
|---|---|
| Best Macro-F1 | **0.882** (RFECA-k20, MCAR, 5%) |
| Worst Macro-F1 | **0.848** (Mean, MAR, 30%) |
| Baseline 0% | **0.886** (identical all methods) |
| Avg MF−Mean (8 cells) | **+0.006** |
| Avg MF−best RFECA (8 cells) | **+0.0002** (range −0.001 to +0.002) |
| Loss 5→30% Mean | MCAR **0.024**; MAR **0.033** |
| Loss 5→30% MissForest | MCAR **0.015**; MAR **0.020** |
| Largest Holm-sig ΔF1 vs Mean | MissForest MAR 30%: **+0.0126** [+0.009, +0.017], **r = 1.00**, p_Holm = 0.016 |
| Largest non-sig \|ΔF1\| vs Mean | RFECA-k20 MCAR 30%: **+0.0076**, r = 0.82, **p_Holm = 0.068** |
| Friedman F1 | 5%: **n.s.** (MCAR p=0.11; MAR p=0.84); significant mainly **≥10–20%** |
| MF vs any RFECA pairwise | **0 / 24** Holm-significant |

### Companion metrics (MCAR; same story — do not over-quote)

| Metric | 5% Mean / MF | 30% Mean / MF |
|---|---|---|
| Balanced Acc | 0.869 / 0.872 | 0.845 / 0.857 |
| Precision | 0.892 / 0.895 | 0.869 / 0.879 |
| Recall | 0.869 / 0.872 | 0.845 / 0.857 |

### Recommend in text
1. Absolute band: **0.848–0.882** (or “~0.85–0.88”)
2. MF−Mean average **~0.006**; max sig **+0.013** at MAR 30%
3. MF ≈ RFECA (**Δ ≈ 0**; no Holm-sig pairwise)
4. Friedman n.s. at 5%; separation at **20–30%**
5. Optional: BalAcc/Prec/Recall only as “mirrored Macro-F1”

**Carrier (§3):**
- “Absolute Macro-F1 remained high for all methods (**0.848–0.882**), with a complete-data baseline of **0.886**.”
- “MissForest improved Macro-F1 over Mean by **0.006** on average, reaching **+0.013** under MAR at 30% (Holm-significant, r = 1.00), whereas differences versus RFECA were negligible (**≤0.002**) and never Holm-significant.”
- “Omnibus F1 differences were non-significant at 5% missingness and emerged mainly at **≥20%** (Table 2).”

---

## 4. Reconstruction vs classification (Fig. 4)

| Item | Value |
|---|---|
| Spearman ρ | overall **−0.55**; MCAR **−0.65**; MAR **−0.52** |
| Pearson r | **−0.37** (descriptive only; points not independent) |
| RMSE range / span | **0.617–1.169** / **0.552** |
| Macro-F1 range / span | **0.848–0.882** / **0.034** |
| Variation ratio | RMSE span is **~16×** Macro-F1 span |
| Strongest decoupling | **MAR 5%**: RMSE ↓ **0.54 (46%)**, F1 change **−0.0001** |
| Runner-up | **MCAR 5%**: RMSE ↓ **0.45 (42%)**, F1 ↑ **+0.0025** |

### Recommend in text
1. **ρ ≈ −0.55** (overall)
2. **~16×** larger RMSE variation than F1
3. MAR 5% example: **46%** RMSE reduction, **≈0** F1 change

**Carrier (§4):**
- “Across method×rate points, RMSE and Macro-F1 were only moderately associated (Spearman **ρ = −0.55**).”
- “RMSE varied over a span of **0.55**, about **16 times** the Macro-F1 span (**0.034**).”
- “Under MAR at 5%, MissForest cut RMSE by **46%** relative to Mean while Macro-F1 was essentially unchanged (**Δ ≈ 0**).”

---

## 5. MCAR vs MAR

| Item | Value |
|---|---|
| Avg RMSE increase MAR−MCAR | **+0.054** (Mean alone: **+0.07 to +0.10** by rate) |
| Avg RV decrease MAR−MCAR | **−0.0016** |
| Avg Macro-F1 decrease MAR−MCAR | **−0.0018** |
| Rankings | Best/worst **unchanged** (MF / Mean) under both mechanisms |

### Recommend in text
1. Mean RMSE inflation under MAR: **~+0.07–0.10**
2. Average F1 shift **≈ −0.002**
3. Qualitative ranking identity

**Carrier (§5):**
- “MAR inflated Mean RMSE by roughly **0.07–0.10** relative to MCAR, but average Macro-F1 differed by only **~0.002**, and method rankings were preserved.”

---

## 6. External validation (CPTAC 2C)

| Item | Value |
|---|---|
| Reconstruction trend | **Matches METABRIC**: MissForest always best RMSE |
| MF vs Mean reduction | MCAR 5%: **37%** (2.808→1.761); MAR 5%: **37%** |
| Classification variance | mean F1 SD **0.018** vs METABRIC **0.005** (**~3.9×**) |
| F1 ranking | **Unstable** (KNN wins 5/8 cells; MF wins 3/8) |
| Caution | RFECA-k20 can exceed Mean RMSE at 30% (MCAR 2.826 vs 2.786; MAR 3.067 vs 2.932) |

### Recommend in text
1. MF best RMSE; **~37%** reduction at 5%
2. F1 SD **~4×** METABRIC
3. Explicit “exploratory / not primary”

**Carrier (§6):**
- “On CPTAC 2C (n=117), MissForest again minimized RMSE (**~37%** below Mean at 5%), but Macro-F1 rankings were unstable and approximately **fourfold** more variable than on METABRIC (Table S3).”

---

## 7. Statistical analysis (concise)

**Strongest findings**
- RMSE & RV: Friedman **p ~ 10⁻⁶** every rate; primary vs Mean **r = ±1.00**, all Holm-significant.

**Largest effect size**
- Reconstruction/structure vs Mean: **\|r\| = 1.00**; largest \|ΔRMSE\| = **0.54** (MAR 5%).

**Smallest effect size (among primary F1)**
- MAR 5% RFECA-k20 vs Mean: **r ≈ −0.09**; ΔF1 **+0.0009** (n.s.).

**Remained non-significant**
- All primary F1 vs Mean at **5%** (both mechanisms).
- Most F1 at **10%** after Holm (except MCAR RFECA-k20).
- **All 24** MissForest vs RFECA F1 pairs.

**Significant mainly at high missingness**
- F1 vs Mean Holm-significant predominantly at **20–30%** (largest ΔF1 **+0.013**, MAR 30%).

---

## 8. Recommended numerical values for the manuscript

### § Reconstruction (cite 4)
| # | Value | Insert after / with |
|---|---|---|
| 1 | RMSE **0.617** vs **1.169** | opening claim of Fig. 1 |
| 2 | **42%** / **46%** MF reduction at 5% (MCAR/MAR) | relative improvement sentence |
| 3 | **r = −1.00**, **p < 10⁻⁵** | statistical support sentence |
| 4 | ΔRMSE MAR 5% **−0.54 [−0.55, −0.53]** | optional CI clause |

### § Structure (cite 3)
| # | Value | Insert with |
|---|---|---|
| 1 | RV **≈1.00** (MF, 5%) | Fig. 2 ceiling |
| 2 | Mean RV **0.966** (MAR 30%) | floor / degradation |
| 3 | Gap **+0.029** at MAR 30% | MF vs Mean |

### § Classification (cite 5)
| # | Value | Insert with |
|---|---|---|
| 1 | Macro-F1 **0.848–0.882** | absolute performance |
| 2 | Baseline **0.886** | 0% missing |
| 3 | Mean MF−Mean **+0.006**; max sig **+0.013** | effect magnitude |
| 4 | MF−RFECA **≈0** (no Holm-sig) | no RFECA superiority |
| 5 | Friedman n.s. at 5%; sig at **20–30%** | Table 2 |

*(BalAcc/Prec/Recall: one phrase only — “parallel Macro-F1” — unless a reviewer asks.)*

### § RMSE vs F1 (cite 3) — **priority finding**
| # | Value | Insert with |
|---|---|---|
| 1 | Spearman **ρ = −0.55** | Fig. 4 |
| 2 | RMSE variation **~16×** F1 | quantitative decoupling |
| 3 | MAR 5%: **46%** RMSE↓, F1 **Δ≈0** | concrete example |

### § MCAR vs MAR (cite 2)
| # | Value | Insert with |
|---|---|---|
| 1 | Mean RMSE **+0.07–0.10** under MAR | mechanism sensitivity |
| 2 | Avg F1 shift **≈ −0.002**; rankings unchanged | robustness |

### § CPTAC 2C (cite 3)
| # | Value | Insert with |
|---|---|---|
| 1 | MF RMSE reduction **~37%** at 5% | reconstruction concordance |
| 2 | F1 SD **~3.9×** METABRIC | variance caveat |
| 3 | “exploratory, n=117” | limit claim strength |

### § Stats wrap (cite 2)
| # | Value | Insert with |
|---|---|---|
| 1 | Reconstruction **r=±1.00** all rates | strongest evidence |
| 2 | F1 gains Holm-sig mainly **≥20%** | two-regime summary |
