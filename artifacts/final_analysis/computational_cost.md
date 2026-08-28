# Custo computacional

## Configuração de produção (OriginalRFECA freeze)
- `gene_workers`: **16**
- BLAS threads: **1** (OMP/MKL/OPENBLAS=1)
- Plataforma: Windows-11-10.0.26200-SP0 / Python 3.13.5
- Tempo total wall (40 slots): **51.46 h**
- Wall médio por slot (réplica × taxa × mecanismo): ver `rfeca_wall_by_mech_rate.csv`
- RSS médio após slot: **183.0 MB** (processo principal; workers adicionais)

## Tempo por slot (média wall_seconds)

mechanism  missing_rate        mean        std          sum  count
      MAR          0.05 5728.448163 143.203154 28642.240816      5
      MAR          0.10 5082.245847  60.383724 25411.229236      5
      MAR          0.20 4213.128437  60.315596 21065.642184      5
      MAR          0.30 3386.881643  71.508231 16934.408214      5
     MCAR          0.05 6304.013200 393.203487 31520.065999      5
     MCAR          0.10 5031.290182 156.815983 25156.450912      5
     MCAR          0.20 3824.691384 699.397876 19123.456918      5
     MCAR          0.30 3477.407757  18.508587 17387.038783      5

## Paralelismo (benchmark autotune)
Fonte: `artifacts/parallel_benchmark/` (8 genes, MCAR 20%, fingerprint-identical).

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


### Nota sobre 16 workers
A produção usou **16 workers** (plano aprovado). O autotune em 8 genes recomendou **8** como ótimo local (speedup 4.70× vs serial). Com 50 genes, 16 workers aumenta ocupação; eficiência cai por imbalance (gene mais lento domina). Não há linha de benchmark formal a 16 no CSV principal — o ganho vs serial é qualitativamente >4× com base no perfil 1→8.

## Comparação com MissForest
MissForest wall times not stored as comparable gene-level slots; use six-imputer full run wall from report if available.

MissForest no CV aninhado (10 reps × 5 folds × taxas) é mais barato por slot que OriginalRFECA gene-a-gene com RFE, mas o OriginalRFECA só avalia a máscara holdout (sem refit por fold de classificação no freeze de imputação).

## Tempo médio por gene (aproximação)
- Wall/slot / 50 ≈ **92.6 s/gene** (com paralelismo; wall, não CPU-soma).
- Soma CPU de seleção / 50 (quando reportada): **1347.0 s/gene**.
