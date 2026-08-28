# Archive

Frozen materials from the original Colab / conference workflow.
**Not consumed by `src/bcimpute`.**

| Path | Contents |
|---|---|
| `colab_experiments/` | Past experiment folders (`experimento 1` … `4`) with raw/summary CSVs |
| `metricas_original/` | Paper tables, Wilcoxon CSVs, figures from the Colab analysis |
| `database/` | Duplicate discovery CSV from Drive-era folder |
| `legacy_root/` | Root-level CSVs (`correlation_table_pam50_full.csv`, merged raw results, discovery copy) |
| `failed_smokes/` | Incomplete smoke-test runs (pre-fix) |

Keep this tree for reproducibility of the submitted conference version.
New runs write only under `artifacts/`.

Classification: **historical / exploratory**. Not used by `src/bcimpute`.
`database/df_MI_no_missing.csv` and `legacy_root/df_MI_no_missing.csv` are
integer-index PAM50 copies (no recovered biological IDs).

