# CPTAC LinkedOmics preparation report

- Generated (UTC): `2026-07-19T21:48:40.852527+00:00`
- Cohort key: **`cptac_linkedomics`**
- Source: LinkedOmics CPTAC-BRCA prospective RNAseq + clinical

## Preparation rules (aligned with discovery)
- LinkedOmics clinical+RNAseq joined on native sample IDs.
- Retained exact 50 PAM50 genes in canonical study order (same as discovery).
- No expression transformation applied (values copied as-is from LinkedOmics).
- No scaling/centering applied at preparation time.
- Rows with labels outside LumA/LumB/Her2/Basal excluded.
- Rows with any NA among the 50 genes excluded (same rule as discovery).
- Native LinkedOmics sample IDs preserved.
- Cohort key: cptac_linkedomics (does not replace discovery).

## Results
- n after filters: **80**
- class distribution: `{'LumA': 38, 'Basal': 20, 'Her2': 12, 'LumB': 10}`
- excluded: **42**
- values+labels sha256: `86e5987f8299bbf53868db007dbe5e2aa3e53c66ff22e2519d1d71022385ec65`

## Note vs discovery
- Discovery keeps n=117 with no native missing values.
- CPTAC LinkedOmics RNAseq has sporadic NAs among PAM50 genes; the shared
  rule (drop rows with any NA) therefore yields fewer than 117 samples.
