# Reduced OriginalRFECA benchmark (20% MCAR + MAR)

- Generated: 2026-08-01T06:11:53.076468+00:00
- MCAR valid replicas: 5/5
- MAR valid replicas: 5/5
- Methodological validity: A

## Aggregates

| Mechanism | RMSE mean | RMSE median | RMSE std | MAE mean | wall (h) |
|-----------|-----------|-------------|----------|----------|----------|
| MCAR | 0.5522 | 0.4999 | 0.2720 | 0.4062 | 5.31 |
| MAR | 0.5754 | 0.5239 | 0.2801 | 0.4193 | 5.85 |

## Scope expansion recommendation

**Decision: `nao_ampliar`**

Keep reduced 20%-only scope for OriginalRFECA; discuss cost/performance trade-off and asymmetry as a limitation.

Reasons:
- MCAR vs MAR nearly redundant (RMSE delta 4.2%)
- Degradation across missingness severity is scientifically useful if RFECA is retained in the dissertation comparison

## ETA (observed-based)

- s/gene effective: 80.4
- s/replica: 4018.9
- Reduced total central: 11.2 h (opt 9.5, cons 14.0)
- Extra for 10% grid: ~11.2 h
- Extra for 30% grid: ~11.2 h
- Full expanded 10/20/30: ~33.5 h

## Next step

Do **not** auto-start 10%/30%. Await explicit decision.
