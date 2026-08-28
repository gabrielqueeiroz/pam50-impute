# Auditoria metodológica (OriginalRFECA TARGET-WISE)

Evidência baseada no código em `src/bcimpute/imputation_original/` e nos artefatos
`artifacts/original_rfeca_reduced_metabric/` (freeze `v0.3.1-original-rfeca-targetwise`).
Cada item: **SIM** / **NÃO** / **NÃO APLICÁVEL** + citação.

---

## 1. Ausência de data leakage
**SIM**

- Protocolo `selection_protocol=leakage_safe` + `input_protocol=target_wise_complete_predictors`.
- Preditores vêm da matriz original completa; apenas a coluna-alvo recebe a máscara artificial.
- Avaliação: `evaluation_protocol=repeated_mask_holdout` (`src/bcimpute/evaluation.py`, `run_imputation_repeated_mask_holdout_target_wise`).
- Flags por slot: `leakage_or_protocol_fail=false`, `n_predictor_nans_at_impute=0`, `svr_coverage=1.0` em todos os 40 `DONE.json`.

## 2. Ausência de SimpleImputer
**SIM**

- OriginalRFECA não instancia `sklearn.impute.SimpleImputer` no caminho TARGET-WISE.
- Fallback de coluna (média) só se o modelo SVR estiver ausente; nos 40 slots `fallback_rate=0` / `total_fallback_count=0`.
- Freeze description: "No SimpleImputer".

## 3. Ausência de chaining
**SIM**

- Transform TARGET-WISE (`BaseOriginalCorrelationImputer._transform_target_wise`): preditores sempre de `X_orig`, nunca de valores já imputados de outros genes.
- `src/bcimpute/imputation_original/base.py` — comentário e asserção "Fallback by missing predictors is forbidden".

## 4. Somente genes originalmente completos como preditores
**SIM** (no sentido do protocolo TARGET-WISE)

- Preditores = valores da matriz completa original (cohort METABRIC PAM50 sem NaNs artificiais nos preditores).
- Máscara artificial aplicada só ao gene-alvo (`set_target_wise_context` + coluna j).
- Política de células do cohort: `originally_observed_mask` usada na geração de missingness (`missingness.py` / runner).

## 5. Correlação calculada apenas no conjunto de treino
**SIM** (para seleção leakage-safe)

- Prefixos Pearson / ordenação de correlação e RFE usam apenas linhas observadas do alvo no fit do gene (`selection_protocol=leakage_safe`).
- Implementação: `src/bcimpute/imputation_original/selection.py` + `base.py` (`_fit_one_gene`).

## 6. RFE executado apenas no treino
**SIM**

- RFE/SVR linear no fit por gene; transform só aplica o pipeline já ajustado.
- `selector_kind=RFE` em `OriginalRFECAImputer`.

## 7. Refit final utilizando somente observações reais
**SIM**

- Modelo vencedor reajustado nas observações não mascaradas do gene-alvo antes de imputar células mascaradas.

## 8. Target mascarado nunca utilizado para treinamento
**SIM**

- Células com `mask[:, j]=True` no gene j não entram como y de treino; só são preenchidas no `transform`.

## 9. Reprodutibilidade confirmada
**SIM** (no âmbito do freeze)

- `seed_scheme=v2`, `base_seed=42`; `FREEZE/manifest.json`: `all_seeds_match_v2_formula=true`.
- Hashes de máscara por slot em `FREEZE/mask_hashes.csv`.
- Benchmark de paralelismo: fingerprints idênticos serial vs paralelo (`artifacts/parallel_benchmark/benchmark_workers.md`).
- Ambiente pinado em `FREEZE/requirements.txt` / `manifest.environment`.

---

## Notas (baselines)

Para Mean / KNN / MissForest a comparação usa o protocolo do paper (imputer-within-CV, shared-mask, 10 reps, seed legacy) — **diferente** do TARGET-WISE mask-holdout (5 reps, seed v2). Isso é uma ameaça à validade de contraste formal, documentada em `threats_to_validity.md`.
