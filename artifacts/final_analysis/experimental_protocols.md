# experimental_protocols.md

Descrição operacional dos dois protocolos executados no benchmark final.  
Complementa `methods_freeze.md` sem reinterpretar intenções.

---

## Protocolo A — Baselines (Mean / KNN / MissForest)

**Artefactos:**  
`artifacts/metabric_full_20260724_185916/` (MCAR),  
`artifacts/metabric_full_mar_20260725_062517/` (MAR).

### Entrada

- Matriz completa METABRIC PAM50 (n=1608 × 50).
- Para cada `missing_rate` ∈ {0, 0.05, 0.10, 0.20, 0.30} e `rep` ∈ {0…9}:
  - Gerar **uma** máscara (seed scheme **legacy**, base 42).
  - Aplicar a **todos** os imputers (shared-mask).

### Loop de avaliação (imputação)

```
Para cada fold StratifiedKFold(5, shuffle=True, random_state=42):
  X_train, X_test ← split por amostras
  M_train, M_test ← máscara restrita aos folds
  imputer.fit(X_train_with_NaNs)
  X_test_imp ← imputer.transform(X_test_with_NaNs)
  RMSE/MAE ← células mascaradas em X_test
  RV / corr ← estrutura do test fold imputado vs completo
```

### Classificação

- Mesmo CV: `Pipeline(imputer → scaler? → clf)`.
- Primary: EnsembleSoft.
- Multiclf (SVC, LogReg, RF, GB) em taxas altas (20/30%) na campanha METABRIC.

### O que este protocolo mede

Erro de imputação **generalizando para amostras não vistas no fit do imputer**, sob máscara partilhada.

---

## Protocolo B — OriginalRFECA TARGET-WISE

**Artefactos:**  
`artifacts/original_rfeca_reduced_metabric/` + `FREEZE/`  
(`v0.3.1-original-rfeca-targetwise`).

### Entrada

- Mesma matriz completa METABRIC PAM50.
- Grid: mecanismos {MCAR, MAR} × rates {0.05, 0.10, 0.20, 0.30} × reps {0…4}.
- Seed scheme **v2**, base 42.
- Máscara persistida por slot.

### Loop de avaliação (imputação)

```
X_orig ← matriz completa (nunca alterada como preditor)
M ← máscara do slot

Para cada gene g (paralelo, 16 workers):
  Trabalho: NaNs apenas em M[:, g]
  Preditores: colunas ≠ g de X_orig (completas)
  y_train: X_orig[i,g] onde ~M[i,g]
  Seleção leakage_safe:
    Para cada prefix_len:
      KFold(5) nas linhas observadas de g
      Em cada train fold: Pearson → prefix → RFE(SVR linear) → SVR(C=1, ε=0.1, no scaler)
      OOF RMSE
    Escolher melhor prefix_len
  Refit final em todas as observadas
  Imputar M[:, g]
  (fallback média de coluna se sem modelo — 0 ocorrências no freeze)

RMSE/MAE ← todas as células M vs X_orig
RV ← não calculado neste protocolo
```

### Classificação

- Matriz já imputada pelo TARGET-WISE.
- CV StratifiedKFold(5) com **identity** imputer (não re-imputa).
- EnsembleSoft + SVC/LogReg/RF/GB.

### O que este protocolo mede

Erro de imputação **nas células mascaradas**, com preditores conhecidos (matriz original), seleção e treino **sem** usar o alvo mascarado nem valores já imputados.

---

## Por que A e B não são o mesmo experimento

| Dimensão | A Baselines | B OriginalRFECA |
|---|---|---|
| Unidade de holdout | Amostras (folds) | Células mascaradas |
| Preditores no fit | Podem conter NaNs (outros genes) | Sempre valores originais completos |
| Shared mask entre métodos | Sim | N/A (método único) |
| n_reps | 10 | 5 |
| Seed scheme | legacy | v2 |
| RV | Sim | Não |
| Classif. nesting | Imputer-in-CV | Pós-imputação |

Comparações numéricas A vs B são **descritivas**, não pareadas.

---

## Protocolo legado (contexto paper package — não freeze TARGET-WISE)

Campanha six-imputer inclui **RFECA_SVR(k∈{5,10,20})** no **mesmo** Protocolo A (CV + shared-mask).  
Esse braço **não** é o OriginalRFECA TARGET-WISE e **não** deve ser descrito como o algoritmo do freeze.

---

## Fluxograma textual unificado

```
                    METABRIC PAM50 complete
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     Protocol A (baselines)          Protocol B (OriginalRFECA)
     mask legacy · 10 reps           mask v2 · 5 reps
     StratifiedKFold(5)              repeated_mask_holdout
     Mean / KNN / MissForest         TARGET-WISE gene loop
     (+ RFECA-k* legado)             leakage_safe · no chain
              │                               │
              ▼                               ▼
     RMSE/MAE/RV + F1                RMSE/MAE + F1 (post)
              │                               │
              └───────────────┬───────────────┘
                              ▼
              Descriptive comparison / dissertation tables
```
