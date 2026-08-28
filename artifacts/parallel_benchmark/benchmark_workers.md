# Gene-level parallel autotune — OriginalRFECA

- Hardware: Ryzen-class (8C/16T), RAM 16232 MB
- Genes (8): CXXC5, EGFR, ERBB2, ESR1, EXO1, FGFR4, FOXA1, FOXC1
- Mask: MCAR 20% rep 0 seed=200042
- Parallelism: genes only; BLAS threads=1; methodology unchanged

## Recommended workers: **8**

Lowest valid wall time among eligible configs (wall=962.0s, speedup=4.70×).

Justification notes:

- All configs (1/2/4/6/8) produced **identical fingerprints** vs serial (RMSE, prefix, predictors, seed).
- RAM peak ≤ 63% (< 80% threshold); no swap pressure; no throttling detected.
- 4→6 wall improvement ≈ 10.6% (> 5%), so the “prefer 4 over 6” stability rule did **not** apply.
- 6→8 still improved by **1.29×**; 8 remained fastest without oversubscription flags.
- Efficiency declines with more workers (load imbalance: longest gene dominates), but wall time keeps falling through 8.

## Comparative table

| workers | wall (s) | s/gene | speedup | efficiency | CPU mean % | RAM peak % | valid |
|---------|----------|--------|---------|------------|------------|------------|-------|
| 1 | 4520.6 | 565.1 | 1.00 | 1.00 | 13.9 | 56.5 | True |
| 2 | 2334.4 | 580.4 | 1.94 | 0.97 | 20.0 | 57.4 | True |
| 4 | 1383.6 | 665.8 | 3.27 | 0.82 | 31.7 | 59.6 | True |
| 6 | 1237.0 | 719.2 | 3.65 | 0.61 | 37.2 | 61.5 | True |
| 8 | 962.0 | 802.5 | 4.70 | 0.59 | 49.7 | 63.0 | True |

## Pairwise relative speedup

- 1->2: **1.94×**
- 2->4: **1.69×**
- 4->6: **1.12×**
- 6->8: **1.29×**

## Updated ETA (reduced benchmark)

- s/gene (serial): 565.1
- MCAR 20% × 5 reps @ 8 workers: **8.4 h**
- MAR 20% × 5 reps @ 8 workers: **8.4 h**
- Reduced full (MCAR+MAR): **16.7 h**

(Serial MCAR-only estimate was ~39.2 h.)

## Remaining bottleneck

RFE with linear SVR inside each gene (~97% of gene time). Gene-level parallelism raises CPU utilization; it does not reduce per-gene RFE cost.
