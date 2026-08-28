# methods_freeze.md

**Single source of truth — methodology as executed**  
Freeze: `v0.3.1-original-rfeca-targetwise`  
Primary evidence: `artifacts/original_rfeca_reduced_metabric/FREEZE/`, `src/bcimpute/`,  
baseline configs `artifacts/metabric_full_*/config_snapshot.json`.  

This document describes **only** what was run. It is not a copy of Rodrigues’ notebook and not a copy of an older article draft.

---

# 1. Visão geral

## Objetivo do benchmark

Quantificar erro de imputação (e, secundariamente, utilidade PAM50) sob missingness artificial MCAR/MAR em matrizes PAM50, comparando:

- **Baselines:** Mean, KNN, MissForest (campanha six-imputer METABRIC, protocolo CV);
- **Método principal:** OriginalRFECA TARGET-WISE leakage-safe (freeze reduzido METABRIC).

## Datasets

- **Principal:** METABRIC PAM50 processado (`load_cohort("metabric")`).
- **Auxiliar / paper legado:** CPTAC 2C (`discovery`) na campanha six-imputer — **não** faz parte do freeze OriginalRFECA TARGET-WISE.

## Estratégia geral

Dois protocolos de avaliação **distintos** (não pareados):

| Braço | O quê | Como avalia imputação |
|---|---|---|
| A — Baselines | Mean / KNN / MissForest (+ legado RFECA-k* na campanha original) | Stratified 5-fold CV; imputer fit no train fold; RMSE/MAE/RV no test fold |
| B — OriginalRFECA | TARGET-WISE | `repeated_mask_holdout` na máscara persistida; preditores = matriz original completa |

Classificação PAM50:

- Baselines: imputer **dentro** do pipeline CV (EnsembleSoft principal).
- OriginalRFECA: classificação **após** imputação TARGET-WISE (imputer = identity no CV).

## Pipeline completo (texto)

```
[Matriz completa PAM50]
        │
        ▼
[Gerar máscara MCAR/MAR · exact cell count · seed]
        │
        ├──────────────────────────────┐
        ▼                              ▼
[Braço A: X_missing]            [Braço B: máscara + X original]
 StratifiedKFold(5)              Para cada gene g:
  fit imputer(train)               NaNs só em M[:,g]
  transform(test)                  preditores = X completo
  RMSE/MAE/RV                      seleção leakage_safe
  + classif. Pipeline              SVR → imputar M[:,g]
                                   RMSE/MAE em células M
        │                              │
        └──────────┬───────────────────┘
                   ▼
        [Tabelas / figuras / freeze]
```

Fluxograma Mermaid alinhado ao freeze: `artifacts/paper_results_original_rfeca/flowchart_original_rfeca.md` (atualizar freeze_id mentalmente para v0.3.1).

---

# 2. Datasets

## METABRIC (benchmark principal — freeze OriginalRFECA + baselines METABRIC)

| Campo | Valor |
|---|---|
| Loader | `bcimpute.data.load_cohort("metabric")` |
| Path típico | `data/processed/metabric/` (CSV `sample_id` + 50 genes + `PAM50`) |
| n amostras | **1608** |
| n genes | **50** (`PAM50_GENES` em `config.py`, ordem fixa) |
| Labels | LumA, LumB, Her2, Basal (`TARGET_LABELS`) |
| Completude na análise | Matriz **completa** exigida (`X.isna().any()` → erro) |
| Proveniência | Illumina HT-12 v3 microarray, log2 intensity (inventory); estudo METABRIC |
| Observation mask | `originally_observed_mask` se ficheiro existir; senão default all-observed |
| Filtro de labels | Apenas labels ∈ TARGET_LABELS |
| Reordenação | Colunas forçadas à ordem `PAM50_GENES` |
| Normalização **no loader** | **Nenhuma** adicional: `astype(float)` apenas |
| Transformações no benchmark | Missingness artificial em **cópia**; matriz fonte não mutada |

**Confirmação:** no braço de análise, os valores de expressão entram como na matriz processada (já log2 na preparação METABRIC). O benchmark **não** reaplica z-score/log no loader. Scaler opcional existe só no pipeline SVR do OriginalRFECA e está **desligado** no freeze (`use_scaler=false`).

## CPTAC 2C / discovery (auxiliar)

| Campo | Valor |
|---|---|
| n | **117** |
| genes | 50 PAM50 |
| Papel | Campanha six-imputer / paper legado; **fora** do freeze OriginalRFECA TARGET-WISE |

## Divisão principal vs validação externa

- **Principal (freeze):** METABRIC OriginalRFECA TARGET-WISE + comparação descritiva com Mean/KNN/MissForest METABRIC.
- **Externa/exploratória:** CPTAC 2C na campanha six-imputer (não TARGET-WISE freeze).

---

# 3. Simulação de missingness

Implementação: `src/bcimpute/missingness.py`.

## Mecanismos

- **MCAR** — `add_missing_values_mcar`
- **MAR** — `add_missing_values_mar`  
- **MNAR** — não implementado / não usado

## Política de células

- `target_cell_policy = "originally_observed_only"` (configs full + freeze)
- Máscara artificial só em células elegíveis; nunca em labels

## MCAR (exato)

1. Contar células elegíveis `n_eligible`.
2. `n_masked = round(missing_rate * n_eligible)`.
3. Amostrar **sem reposição** índices elegíveis com `numpy.random.default_rng(seed)`.
4. Aplicar máscara em cópia de X; persistir máscara booleana.

## MAR (exato — implementação efetiva)

Documentação no docstring + código:

1. Mesmo `n_masked` exact-count que MCAR.
2. Para cada célula elegível (i,j), preditor = **média leave-one-out da linha i** (média das outras genes, excluindo j) — **não** usa X[i,j] nem labels PAM50.
3. Standardizar essas médias por coluna (z-score).
4. Score = `|z| + Gumbel(0,1)` (ruído do RNG da seed).
5. Selecionar top-`n_masked` células elegíveis por score (`argpartition`).
6. Meta: `mar_predictor = "leave_one_out_row_mean_abs_z_plus_gumbel"`.

Isto é **MAR** no sentido P(R_ij=1 | X) = f(X_i, −j), não MNAR.

## Taxas e réplicas

| Braço | Taxas | Réplicas | Seed scheme | base_seed |
|---|---|---|---|---|
| Baselines METABRIC full | 0, 0.05, 0.10, 0.20, 0.30 | 10 (0–9) | **legacy** (artefactos pré-campo; AUDIT) | 42 |
| OriginalRFECA freeze | 0.05, 0.10, 0.20, 0.30 | 5 (0–4) | **v2** | 42 |

## Seeds

- Função: `missingness_seed(base, rate, rep, mechanism, scheme)`
- **v2:** `base + mech_offset + rep * 1_000_003 + round(rate * 1_000_000)`; `mech_offset=0` (MCAR) ou `17000` (MAR)
- **legacy:** `base + 1000 + rep + round(rate*100) + mech_offset`
- Freeze: `all_seeds_match_v2_formula=true` (`FREEZE/manifest.json`)

## Persistência de máscaras

- OriginalRFECA: `mask.npz` por slot (`mask`, `seed`, `mask_hash`) + hashes em `FREEZE/mask_hashes.csv`
- Baselines: máscaras geradas por seed e **partilhadas** entre imputers no mesmo rate×rep (`assert_shared_mask_across_imputers`)

---

# 4. Protocolos experimentais

## 4.1 Baselines (Mean / KNN / MissForest)

Fonte: `full_benchmark_config`, `evaluation.py`, `metabric_full_*/config_snapshot.json`, `pipelines.py`.

| Item | Valor executado |
|---|---|
| Shared-mask | Sim — mesma máscara para todos os imputers em rate×rep |
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Réplicas | 10 |
| Seed scheme missingness | legacy (artefactos principais) |
| Imputação | Fit imputer **apenas** no fold de treino; `transform` no teste |
| RMSE/MAE | Células mascaradas do **test fold** vs verdade completa |
| RV / correlação | Matriz completa do test fold imputada vs completa original |
| Classificação | Pipeline `imputer → (scaler) → clf` no mesmo CV; primary = **EnsembleSoft** |
| Classifiers extras | SVC, LogReg, RF, GB (multiclf a 20/30% METABRIC) |

**Mean:** `SimpleImputer(strategy="mean")`  
**KNN:** `KNNImputer(n_neighbors=5, weights="distance")`  
**MissForest-like:** `IterativeImputer(ExtraTreesRegressor(n_estimators=20), max_iter=5, random_state=42, sample_posterior=False, skip_complete=True)`; `missforest_n_jobs=4` no full config

## 4.2 OriginalRFECA (freeze)

Fonte: `FREEZE/config_snapshot.json`, `imputation_original/`, `evaluation.run_imputation_repeated_mask_holdout_target_wise`, `experiments/run_original_rfeca_targetwise.py`.

| Item | Valor executado |
|---|---|
| `evaluation_protocol` | `repeated_mask_holdout` |
| `input_protocol` | `target_wise_complete_predictors` |
| `predictor_values` | `original_complete_matrix` |
| `selection_protocol` | `leakage_safe` |
| SimpleImputer no caminho SVR | **Ausente** |
| Chaining | **Ausente** — preditores nunca vêm de genes já imputados |
| Máscara no alvo | Apenas coluna do gene-alvo recebe NaNs artificiais na matriz de trabalho |
| Preditores | Sempre valores da matriz original completa |
| Seleção | Pearson abs no train → prefixos → RFE+SVR linear → melhor prefix_len por OOF RMSE |
| Validação interna | `KFold(n_splits=5)` nas linhas **observadas** do alvo (`validation_strategy=kfold`) |
| Refit final | Pearson + RFE + SVR em **todas** as linhas observadas do alvo |
| Imputação | `predict` só nas células mascaradas do alvo |
| Classificação PAM50 | Pós-imputação; identity imputer + StratifiedKFold(5); EnsembleSoft (+ outros) |

### Por que difere dos baselines

1. Não há fold externo de amostras para o imputer: a holdout é **nas células mascaradas**, com preditores completos.
2. Não há chaining nem preenchimento prévio de preditores.
3. n_reps / seed scheme / avaliação estrutural (RV) diferem.
4. Classificação não re-imputa dentro do CV.

---

# 5. OriginalRFECA — fluxo por gene

Código: `BaseOriginalCorrelationImputer` (`base.py`), `select_features_rfe` (`selection.py`), `make_svr_pipeline` / `fit_final_svr` (`svr_model.py`), transform TARGET-WISE.

## Fluxo

1. **Contexto:** `set_target_wise_context(X_original, mask)`.
2. **Máscara do gene g:** células `mask[:, g]`; y de treino = valores observados de g (não mascarados).
3. **Candidatos:** outros genes; `max_candidates=49` (todos os restantes no PAM50).
4. **Por cada `prefix_len` candidato (1…|order|):**
   - Inner KFold nas observadas;
   - Em cada fold de **treino:** ranking Pearson abs(g, candidatos); prefixo; RFE(SVR linear); fit SVR (`use_scaler=false`); predizer validação;
   - Agregar OOF RMSE.
5. **Escolher** `prefix_len` de menor RMSE OOF.
6. **Refit final** em todas as observadas: Pearson → RFE → SVR.
7. **Imputar** células mascaradas de g com preditores de `X_original`.
8. **Avaliar** RMSE/MAE nas células mascaradas vs verdade.

Paralelismo: um processo por gene (`parallel_genes.py`), BLAS=1, 16 workers no freeze.

## Respostas explícitas (SIM / NÃO)

| Pergunta | Resposta | Referência |
|---|---|---|
| Existe data leakage (no desenho TARGET-WISE leakage_safe)? | **NÃO** | `selection_protocol=leakage_safe`; máscara excluída do y de treino; `methodology_audit.md`; flags slot |
| Existe SimpleImputer no caminho SVR TARGET-WISE? | **NÃO** | Sem `SimpleImputer` em `imputation_original/`; freeze “No SimpleImputer” |
| Existe chaining? | **NÃO** | `_transform_target_wise` usa só `X_orig` |
| Valores imputados usados como preditores? | **NÃO** | Idem |
| Genes parcialmente imputados na seleção? | **NÃO** | Preditores da matriz original completa |
| Target mascarado participa do treinamento? | **NÃO** | Só linhas observadas de g |
| RFE utiliza dados de teste (holdout mascarado)? | **NÃO** | RFE no train fold / observadas; células mascaradas só na imputação final |
| Refit utiliza dados imputados? | **NÃO** | Refit só em observadas reais |

*Nota:* fallback de média de coluna existe no código se não houver modelo SVR; no freeze, `fallback_rate=0` em todos os slots.

---

# 6. Hiperparâmetros

Ver também `hyperparameters_table.csv`.

### OriginalRFECA (freeze)

| Componente | Parâmetro | Valor |
|---|---|---|
| SVR (pipeline final / folds) | kernel | `linear` |
| | C | `1.0` |
| | epsilon | `0.1` |
| | use_scaler | `false` |
| | gamma / shrinking / tol / cache_size | defaults sklearn (gamma irrelevante para linear) |
| RFE estimator | | `SVR(kernel="linear")` (C/ε defaults sklearn no estimator do RFE) |
| RFE | step | default `1` |
| | n_features_to_select | default `None` → `n_features // 2` |
| Inner CV | strategy | `kfold` |
| | n_splits | `5` |
| | random_state | `42` |
| Selection | max_candidates | `49` |
| | candidate_rule | `full_matrix` (OriginalRFECA) |
| | min_train_samples | `10` |
| Parallel | gene_workers | `16` |
| | BLAS threads | `1` |
| Seeds | scheme | `v2` |
| | base_seed | `42` |

### Baselines

| Método | Parâmetros |
|---|---|
| Mean | `SimpleImputer(strategy="mean")` |
| KNN | `n_neighbors=5`, `weights="distance"` |
| MissForest-like | ExtraTrees `n_estimators=20`; IterativeImputer `max_iter=5`, `random_state=42`, `sample_posterior=False`, `skip_complete=True`; `n_jobs=4` |

### Classificadores (`pipelines.py`, `random_state=42`)

| Modelo | Parâmetros chave |
|---|---|
| SVC | `kernel=rbf`, `C=1.0`, `gamma=scale`; EnsembleSoft usa `probability=True` |
| LogReg | `max_iter=5000` |
| RF | `n_estimators=200` |
| GB | defaults `GradientBoostingClassifier` + `random_state=42` |
| EnsembleSoft | soft vote SVC+LogReg+RF+GB |
| CV classif. | StratifiedKFold 5, `random_state=42` |

---

# 7. Avaliação

## Métricas de imputação

| Métrica | Baselines | OriginalRFECA freeze |
|---|---|---|
| RMSE (células mascaradas) | Sim (test fold) | Sim (máscara holdout) |
| MAE | Sim | Sim |
| RV / corr structure | Sim | **Não** no freeze |

## Por que OriginalRFECA não tem RV

O protocolo `repeated_mask_holdout` TARGET-WISE avalia erro nas células mascaradas após imputação gene-a-gene; o pipeline de freeze **não** calcula RV/correlação da matriz imputada completa. RV existe só no braço CV dos baselines (`evaluation.py` metrics estruturais no test fold).

## Métricas de classificação

| | Baselines | OriginalRFECA |
|---|---|---|
| Macro-F1, bal_acc, prec/rec por classe | Sim (EnsembleSoft; multiclf parcial) | Sim (pós-imputação; 5 clf) |
| Nesting | imputer-in-CV | identity-in-CV |

---

# 8. Estatística

## Testes realizados (pacote paper / stats)

- Wilcoxon + Holm + Friedman sobre campanha **six-imputer** (inclui **RFECA_SVR(k=*) legado**), artefacto `stats_mcar_mar_20260727_160425`.
- **Não** são testes do OriginalRFECA TARGET-WISE vs MissForest.

## Comparações válidas (mesmo protocolo)

- Entre Mean/KNN/MissForest/RFECA-k* **dentro** da campanha shared-mask CV.
- Entre slots OriginalRFECA (mesmas seeds v2, mesmo holdout) — descritivo entre taxas/mecanismos.

## Comparações a interpretar com cautela

- OriginalRFECA TARGET-WISE vs MissForest/KNN/Mean: protocolos, n_reps e seeds **diferentes**.
- F1 OriginalRFECA vs F1 baselines (nesting diferente).
- Qualquer p-value Welch/boot descritivo em `final_analysis/statistical_*.csv` — **não confirmatório**.

## n efetivo (aprox.)

- Baselines: até 10 reps × 5 folds por célula método×taxa×mecanismo (agregação fold→rep no paper).
- OriginalRFECA: 5 reps (slot-level RMSE); classificação 5 folds × 5 reps.

---

# 9. Custo computacional

| Item | Valor |
|---|---|
| OS | Windows 11 (freeze manifest) |
| Python | 3.13.5 |
| Hardware (benchmark parallel) | Ryzen-class 8C/16T, ~16 GB RAM |
| Workers freeze | **16** gene-workers; BLAS=1 |
| Wall total OriginalRFECA 40 slots | **~51.5 h** |
| Wall médio / slot | ~4631 s (~1.3 h) |
| Wall/slot/50 genes (aprox.) | ~93 s |
| Microbenchmark speedup | até **4.70×** @ 8 workers vs serial (8 genes); fingerprints idênticos |
| RSS processo principal (médio after) | ~183 MB (workers adicionais à parte) |

Fonte: `computational_cost.md`, `parallel_benchmark/`, `FREEZE/manifest.json`.

---

# 10. Diferenças em relação ao algoritmo / notebook de Rodrigues

Fonte de comparação: `AUDIT_STOP_rfeca_vs_notebook_20260730.md` + freeze final (pós-correções).  
Sem juízo de valor — apenas diferenças.

| Tópico | Notebook / original (Rodrigues) | Implementação final (freeze) | Justificativa registada |
|---|---|---|---|
| Leakage no RFE | RFE no conjunto completo observado; LOOCV só no SVR | RFE **dentro** de cada fold de treino; leakage_safe | Eliminar leakage de seleção |
| Validação para escolher prefixo | LOOCV | KFold(5) | Custo / protocolo leakage_safe |
| StandardScaler | Ausente no fit SVR do notebook | `use_scaler=false` no freeze | Alinhar ao notebook **e** config leakage_safe |
| Ranking de correlação | Tabela/ranking fixo (contexto Basal completo) | Pearson **recalculado** por fold / refit | Consistência com leakage_safe + missingness artificial |
| Universo de dados | Genes incompletos Basal vs completos | PAM50 METABRIC + MCAR/MAR artificial | Benchmark controlado do projeto |
| Preditores | Genes estruturalmente completos | Matriz original completa TARGET-WISE | Congelar preditores; anti-chaining |
| SimpleImputer / chaining | Não no caminho SVR notebook | Ausentes no TARGET-WISE | Contrato metodológico |
| Avaliação externa | Contexto do notebook | `repeated_mask_holdout` | Métrica reprodutível no estudo |
| Paralelização | Não (notebook) | Gene-workers 16 | Factibilidade do grid |
| Auditoria / freeze | Não | DONE flags, mask hashes, manifest | Reprodutibilidade |
| Fallback média | Não no caminho SVR tipico | Código permite; **0** no freeze | Robustez operacional |

---

# 11. Checklist de reprodutibilidade

Ver `methods_checklist.md` (detalhe). Resumo:

| Item | Estado |
|---|---|
| Seeds | ✓ v2 + base 42; audit match |
| Máscaras | ✓ `mask.npz` + hashes |
| Checkpoints | ✓ `checkpoint/gene_models.joblib` / models por gene |
| Ambiente | ✓ Python/packages no manifest |
| Requirements | ✓ FREEZE requirements |
| Hashes | ✓ mask_hashes.csv |
| Paralelização determinística | ✓ fingerprint-identical no autotune |
| Versionamento | ✓ freeze_id v0.3.1 |
| Freeze | ✓ |
| Artefactos | ✓ REPORT_* / classification / final_analysis |

---

# 12. Auditoria final

**Esta documentação descreve exatamente o protocolo executado?**

## SIM

com o âmbito explícito:

- Braço OriginalRFECA = freeze `v0.3.1` METABRIC TARGET-WISE;
- Braço baselines = `metabric_full_20260724_185916` / `metabric_full_mar_20260725_062517` (+ multiclf onde aplicável);
- CPTAC e RFECA-k* legado descritos só como contexto da campanha paper, **não** como freeze TARGET-WISE.

**Inconsistências menores a ter presentes ao escrever Methods (não invalidam o SIM):**

1. `flowchart_original_rfeca.md` ainda cita freeze_id `v0.3.0` no cabeçalho — conteúdo do fluxo é o TARGET-WISE; o id canónico é **v0.3.1**.
2. Não existe ficheiro de artigo `.tex` Methods no repositório; a comparação artigo usa o **pacote paper congelado** (`pipeline_diagrams.md` / `AUDIT.md`) como proxy da Methods publicada no paper package.
3. MissForest é **MissForest-like** (IterativeImputer + ExtraTrees), não o pacote R original — já documentado no código.

---

# 13. Compatibilidade com o artigo / pacote Methods atual

Ver plano detalhado em **`methods_vs_article.md`**.
