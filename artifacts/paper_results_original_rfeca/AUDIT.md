# Audit — OriginalRFECA paper package

Generated: `2026-08-02T16:14:09.697371+00:00`
Checks: **19/19 PASS**

## Rounding
- Display decimals: RMSE=3, MAE=3 (half-up)
- Figure annotations use the same rounded means as `summary_by_mech_rate_display.csv`

## Table ↔ figure consistency
- MAR 10%: PASS (full=0.638834 → display/fig=0.639)
- MAR 20%: PASS (full=0.639723 → display/fig=0.640)
- MAR 30%: PASS (full=0.639494 → display/fig=0.639)
- MCAR 10%: PASS (full=0.614648 → display/fig=0.615)
- MCAR 20%: PASS (full=0.614263 → display/fig=0.614)
- MCAR 30%: PASS (full=0.621264 → display/fig=0.621)

## Automated checks
- [PASS] protocol_evaluation_protocol
- [PASS] protocol_input_protocol
- [PASS] protocol_predictor_values
- [PASS] protocol_selection_protocol
- [PASS] use_scaler_false
- [PASS] max_candidates_49
- [PASS] all_slots_class_A
- [PASS] svr_coverage_1
- [PASS] zero_fallbacks_and_pred_nans
- [PASS] report_match_mar_10 Δrmse=0.00e+00
- [PASS] report_match_mar_20 Δrmse=0.00e+00
- [PASS] report_match_mar_30 Δrmse=0.00e+00
- [PASS] report_match_mcar_10 Δrmse=0.00e+00
- [PASS] report_match_mcar_20 Δrmse=0.00e+00
- [PASS] report_match_mcar_30 Δrmse=0.00e+00
- [PASS] freeze_mask_hash_match
- [PASS] freeze_seed_match
- [PASS] per_gene_row_count
- [PASS] per_gene_all_ok

No issues.
