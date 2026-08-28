# Discovery cohort (laboratory matrix from the original experiment)

## Files

| File | Role |
|---|---|
| `../raw/discovery/df_MI_no_missing.csv` | Immutable source copy (**not in this public snapshot**; prepare locally) |
| `discovery_pam50_4class.csv` | Analysis-ready matrix (`sample_id` + 50 genes + `PAM50`). **Gitignored** when IDs are recovered CPTAC Patient_IDs; generate locally with `prepare_discovery.py`. |
| `discovery_inventory.json` | Machine-readable inventory |
| `discovery_provenance_report.md` | Human provenance summary |
| `discovery_fingerprint.json` | Deterministic content hashes |
| `discovery_excluded_samples.csv` | Exclusion log |
| `discovery_preparation_log.txt` | Preparation log |

## Naming policy

Internal key: **`discovery`**.  
Do **not** label scientific outputs as CPTAC until provenance is confirmed.

## Prepare / validate

```powershell
python scripts\prepare_discovery.py
python scripts\test_prepare_discovery.py
```
