# Experimental pipeline diagrams

Publication-oriented flowcharts matching `src/bcimpute/evaluation.py` and `pipelines.py`.  
Paste each Mermaid block into any Mermaid renderer (GitHub, Quarto, [mermaid.live](https://mermaid.live)).

Shared protocol (both experiments):

- Artificial missingness: MCAR or MAR; rates 0–30%; 10 replications
- Same mask shared across imputers within each rate times replication
- Stratified 5-fold CV; `random_state=42`
- No train/test leakage: fit on training fold only

---

## Figure A — Imputation evaluation pipeline

```mermaid
flowchart TD
  startNode([Complete expression matrix X]) --> maskNode{{MCAR or MAR masking}}
  maskNode --> xMiss[/Incomplete matrix X_missing/]
  maskNode --> maskOut[/Binary mask M/]

  xMiss --> loopRep[For each missingness rate and replication]
  maskOut --> loopRep
  loopRep --> loopImp[For each imputer]
  loopImp --> cvSplit{{Stratified 5-fold CV seed 42}}

  cvSplit --> trainPart[/Training fold X_train incomplete/]
  cvSplit --> testPart[/Test fold X_test incomplete/]

  trainPart --> fitImp[[fit imputer on X_train only]]
  fitImp --> transformTest[[transform X_test]]
  testPart --> transformTest
  transformTest --> xTestImp[/Imputed test fold/]

  xTestImp --> rmseNode[RMSE and MAE on masked test cells only]
  xTestImp --> structNode[RV and correlation metrics on full test fold]
  rmseNode --> aggNode[\Aggregate over folds then replications\]
  structNode --> aggNode
  aggNode --> outImp([Imputation tables and figures])
```

**Notes**

- Ground-truth for RMSE/MAE: original complete values at cells where `M = true` on the test fold.
- RV compares gene–gene Pearson correlation of the complete test fold versus the imputed test fold (all cells, not masked-only).
- Training fold is used to fit the imputer; reconstruction metrics are reported on the held-out fold.

---

## Figure B — Downstream classification pipeline

```mermaid
flowchart TD
  startCls([Same X_missing and mask as imputation]) --> loopRepCls[For each rate and replication]
  loopRepCls --> loopImpCls[For each imputer]
  loopImpCls --> pickClf[Select classifier pipeline]
  pickClf --> cvCls{{Stratified 5-fold CV seed 42}}

  cvCls --> xTr[/X_train y_train/]
  cvCls --> xTe[/X_test y_test/]

  xTr --> fitPipe[[fit pipeline on training fold]]
  fitPipe --> predTe[[predict on X_test]]
  xTe --> predTe
  predTe --> metricsCls[Macro-F1 BalAcc Precision Recall]
  metricsCls --> aggCls[\Aggregate over folds then replications\]
  aggCls --> outCls([Classification tables and figures])

  pickClf -.-> svcPipe[SVC: impute then StandardScaler then SVC]
  pickClf -.-> lrPipe[LogReg: impute then StandardScaler then LogReg]
  pickClf -.-> rfPipe[RF: impute then RandomForest]
  pickClf -.-> gbPipe[GB: impute then GradientBoosting]
  pickClf -.-> ensPipe[EnsembleSoft: impute then soft vote SVC LogReg RF GB]
```

**Notes**

- Paper primary endpoint uses **EnsembleSoft**; individual classifiers are for the interaction analysis.
- Imputation inside the sklearn `Pipeline` is **independent** of the imputation-experiment fit (clone per fold; same mask and CV splits).
- Scalers, when present, are fit on the training fold only.
- Main Tables 1 / S1 / S2 report EnsembleSoft Macro-F1; multi-clf interaction uses the dedicated METABRIC multi-classifier run.

---

## How the two pipelines relate

```mermaid
flowchart LR
  shared([Shared incomplete dataset and CV splits]) --> impExp[Imputation experiment]
  shared --> clsExp[Classification experiment]
  impExp --> impFit[[Separate imputer fit per fold]]
  clsExp --> clsFit[[Separate pipeline fit per fold]]
  impFit --> impMetrics[RMSE MAE RV]
  clsFit --> clsMetrics[Macro-F1 and companions]
```

Both use identical missingness masks and stratified CV splits, but each experiment **re-fits** imputation independently so there is no cross-experiment model reuse.
