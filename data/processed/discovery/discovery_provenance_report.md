# Discovery cohort provenance report

- Generated (UTC): `2026-07-19T21:31:35.845155+00:00`
- Cohort key: **`discovery`** (not renamed to CPTAC)
- Provenance confidence: **High (Highly likely CPTAC; cohort key remains discovery)**

## Source
- Raw immutable path: `<repo-root>\data\raw\discovery\df_MI_no_missing.csv`
- SHA256: `21674b6bc19c0b44417729169cb16189e890129ee02cc87df16aa498f4b45ac6`
- Notebook historical path hint: `/content/drive/MyDrive/Database/brca_metabric/df_MI_no_missing.csv`

## Sample ID status
- Status: **synthetic_row_ids**
- Biological IDs available: **False**
- Scheme: `discovery_row_XXXX (1-based synthetic)`
- ID search conclusion: No biological sample IDs recovered from local notebooks/CSVs/archives for the discovery matrix. Available source uses a reset integer index (0..n-1). Other-cohort IDs (e.g. METABRIC MB-*) were ignored.

## Scientific naming policy
- Use **discovery cohort** / **laboratory cohort** in paper text until provenance is closed.
- Do **not** label outputs as CPTAC unless sample IDs / source documentation confirm it.

## Evidence summary
- Matrix has 117 samples and PAM50 4-class counts (LumA 57, Basal 29, LumB 17, Her2 14), which match CPTAC prospective after dropping Normal-like.
- External LinkedOmics audit: **1:1 molecular bijection** to CPTAC RNAseq under affine re-scaling → formal conclusion **Highly likely CPTAC** (not Confirmed: no native IDs / no download recipe / 2 label disagreements).
- Local files retain a reset integer index; no biological IDs recovered in the source export.

## External CPTAC audit (2026-07-19)

- Formal conclusion: **Highly likely CPTAC** (see `artifacts/discovery_cptac_provenance_audit/`).
- Exact 4-class counts match LinkedOmics CPTAC-BRCA prospective after excluding Normal-like.
- Discovery rows form a **1:1 molecular bijection** with CPTAC RNAseq profiles under per-sample affine scaling (Pearson ≈ 1); raw values are not byte-identical.
- Biological sample IDs remain unavailable in the source file; putative IDs are recovered only by matching.
- **Cohort key stays `discovery`** — not renamed to CPTAC.



## Preprocessing decisions
- Source CSV read with index_col=0 (matches original notebook).
- Retained exact 50 PAM50 genes in canonical study order.
- No expression transformation applied (values copied as-is).
- No scaling/centering applied at preparation time.
- Rows with labels outside LumA/LumB/Her2/Basal excluded if present.
- Rows with any NA among the 50 genes excluded if present.
- Synthetic IDs used only when biological IDs are unavailable.
- Cohort key remains 'discovery' (not renamed to CPTAC).

