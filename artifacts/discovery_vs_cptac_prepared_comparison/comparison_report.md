# Discovery vs CPTAC LinkedOmics — same preparation pipeline

- Generated (UTC): `2026-07-19T21:48:48.233044+00:00`

## Shared prep rules
- sample_id | 50 PAM50 genes (canonical order) | PAM50
- Keep LumA/LumB/Her2/Basal only
- Drop rows with any NA among the 50 genes
- No scaling/centering/transform at preparation time

## Dimensions & labels
- Discovery: n=**117** · `{'LumA': 57, 'Basal': 29, 'LumB': 17, 'Her2': 14}`
- CPTAC prepared: n=**80** · `{'LumA': 38, 'Basal': 20, 'Her2': 12, 'LumB': 10}`
- Dn (cptac - discovery): **-37**
- Class count deltas: `{'Basal': -9, 'Her2': -2, 'LumA': -19, 'LumB': -7}`

## Expression ranges (prepared)
- Discovery: `{'min': -13.8318, 'max': 13.203, 'mean': -0.07016080341880343, 'std': 2.8347244309934823, 'median': 0.029699999999999997}`
- CPTAC: `{'min': -10.7012, 'max': 10.8386, 'mean': 0.013566125, 'std': 2.3236773554239374, 'median': 0.06805}`

## Paired overlap (via putative ID map ∩ prepared CPTAC)
- Overlap n: **80** (68.4% of discovery; 100.0% of prepared CPTAC)
- Label agreement: **0.988**
- Pearson median / min: **1.000000** / 1.000000
- Z-cosine median: **1.000000**
- Raw max-abs median: **1.2839** (exact matches: 0)
- Affine a median / RMSE median: **0.8634** / 3.73e-05

## Unsupervised nearest neighbor (prepared matrices)
- Best z-cosine median/min: **1.000000** / 0.462099
- Frac ≥ 0.999: **0.684**

## Fingerprints
- Discovery: `9994c4ed68d2c3c299aeb2a4d609c8edc091768b46d2dca77ce3f5c669838bea`
- CPTAC: `86e5987f8299bbf53868db007dbe5e2aa3e53c66ff22e2519d1d71022385ec65`
- Identical prepared matrices: **False**

## Conclusion
Same prep rules applied. Prepared CPTAC is smaller than discovery because LinkedOmics RNAseq has native NAs among PAM50 genes and those rows are dropped. On the overlapping complete samples, profiles remain near-perfect affine matches to discovery (not byte-identical).
- Keys: discovery vs `cptac_linkedomics`
- Do not rename discovery to CPTAC.
