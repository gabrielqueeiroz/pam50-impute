# PREFLIGHT_REPORT

## Verdict: A

- Genes: ['ACTR3B', 'ANLN']
- Mechanism: MCAR, rate=0.20, replicate=0
- predictor_nan_count: 0
- svr_coverage: 1.0
- fallback_count: 0
- mask_match_eval_positions: True
- wall_seconds: 1355.3
- resume_determinism_ok: True (resume wall 0.00s)
- Issues: none

## Protocol checks

- NaN only on target: enforced by target_wise matrix builder
- Predictors from original complete matrix: yes
- No SimpleImputer / chaining: yes
- Metrics only on masked target cells: yes
- Checkpoint per gene: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mcar\rate_0.20\rep_0\preflight\checkpoint\genes`

## Classification criteria

- A: all checks pass
- B: methodology ok, operational issues
- C: leakage / imputation / non-reproducible

Artifacts: `<repo-root>\artifacts\original_rfeca_reduced_metabric\mcar\rate_0.20\rep_0\preflight`
