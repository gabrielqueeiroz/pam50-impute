# methods_vs_article.md

Comparação entre `methods_freeze.md` (protocolo **executado**) e a seção Methods do **pacote de artigo congelado** no repositório.

**Proxy do artigo atual:** não há `Methods.tex` / manuscript Methods no tree; usou-se:

- `artifacts/paper_results/pipeline_diagrams.md`
- `artifacts/paper_results/AUDIT.md`
- `artifacts/paper_results/manuscript_numerics_guide.md` (onde aplicável)

Não se modificou o artigo. Isto é um **plano de atualização**.

---

## Resumo executivo

A Methods do paper package descreve a campanha **six-imputer** (Mean, KNN, MissForest, RFECA_SVR k=5/10/20) com **shared-mask + Stratified 5-fold CV**.  
O protocolo principal da dissertação / contribuição atual é **OriginalRFECA TARGET-WISE** (`repeated_mask_holdout`, leakage_safe, sem SimpleImputer/chaining).  

Essas duas narrativas **não são intercambiáveis**. A maior parte da Methods do paper package precisa de **reescrita estrutural**, não de cosméticos.

---

## Por subseção (proxy paper package)

### Dataset / cohort loading

| Status | **Parcialmente compatível** |
|---|---|
| Permanece válido | METABRIC n=1608, 50 PAM50 genes, labels TARGET_LABELS, matriz completa no load |
| Reescrever | Enfatizar que o freeze OriginalRFECA é **só METABRIC**; CPTAC 2C é campanha auxiliar/legado |
| Remover | Qualquer leitura implícita de que CPTAC valida o TARGET-WISE freeze |
| Acrescentar | Confirmação: sem normalização adicional no loader; `use_scaler=false` no SVR freeze |

### Missingness (MCAR / MAR)

| Status | **Compatível** (núcleo) / **Parcial** (seeds) |
|---|---|
| Permanece válido | Exact-count MCAR; MAR leave-one-out row mean + \|z\| + Gumbel; `originally_observed_only` |
| Reescrever | Seeds: baselines paper = **legacy**; OriginalRFECA freeze = **v2** — não misturar |
| Remover | Afirmação de que o mesmo seed formula cobre ambos os braços |
| Acrescentar | Persistência `mask.npz` + hashes no braço TARGET-WISE |

### Imputation methods — Mean / KNN / MissForest

| Status | **Compatível** |
|---|---|
| Permanece válido | Shared-mask, StratifiedKFold(5), imputer-in-CV, hiperparâmetros KNN/MissForest-like |
| Reescrever | Pouco — alinhar nomenclatura “MissForest-like” |
| Remover | Nada essencial |
| Acrescentar | Explicitar que estes são **baselines de protocolo distinto** do OriginalRFECA |

### Imputation methods — RFECA / RFECA_SVR(k=*)

| Status | **Incompatível** (como método principal) |
|---|---|
| Permanece válido | Pode permanecer como **campanha legado / paper package** se o artigo ainda reportar k∈{5,10,20} |
| Reescrever | Separar claramente: (A) RFECA-k* CV legado vs (B) OriginalRFECA TARGET-WISE freeze |
| Remover | Como descrição do método principal da contribuição atual; qualquer fluxo que misture SimpleImputer/chaining com “RFECA” |
| Acrescentar | Secção completa TARGET-WISE: leakage_safe, repeated_mask_holdout, preditores da matriz original, sem SimpleImputer, sem chaining, RFE dentro dos folds, KFold(5), use_scaler=false |

### Evaluation protocol (imputation metrics)

| Status | **Parcialmente compatível** |
|---|---|
| Permanece válido | RMSE/MAE em células mascaradas para baselines (test fold) |
| Reescrever | OriginalRFECA: holdout de máscara, não test fold de amostras |
| Remover | Implicação de que RV se aplica a todos os métodos |
| Acrescentar | “OriginalRFECA não reporta RV no freeze” + razão |

### Classification

| Status | **Parcialmente compatível** |
|---|---|
| Permanece válido | EnsembleSoft, StratifiedKFold(5), macro-F1; classifiers SVC/LogReg/RF/GB |
| Reescrever | Nesting: baselines = imputer-in-CV; OriginalRFECA = pós-imputação + identity |
| Remover | Afirmação de F1 diretamente comparável entre braços sem caveat |
| Acrescentar | Caveat de interpretabilidade F1 cross-protocol |

### Statistical analysis

| Status | **Incompatível** (se aplicado ao freeze OriginalRFECA) |
|---|---|
| Permanece válido | Wilcoxon/Holm/Friedman **dentro** da campanha six-imputer |
| Reescrever | Declarar que esses testes **não** comparam OriginalRFECA TARGET-WISE vs MissForest |
| Remover | p-values como evidência confirmatória RFECA TARGET-WISE vs baselines |
| Acrescentar | Limitação: protocols/n_reps/seeds distintos; comparação descritiva apenas |

### Computational / reproducibility

| Status | **Parcialmente compatível** |
|---|---|
| Permanece válido | Seeds, shared masks (baselines), config snapshots |
| Reescrever | Freeze id, gene_workers=16, ~51.5 h wall, parallel fingerprint |
| Remover | Números de runtime da campanha antiga se conflituarem |
| Acrescentar | Checklist freeze v0.3.1 + mask hashes |

### Differences vs Rodrigues / original algorithm

| Status | **Incompatível** / ausente no paper package |
|---|---|
| Permanece válido | N/A (pouco ou nada no paper package) |
| Reescrever | — |
| Remover | Equivalência implícita “RFECA = notebook Rodrigues” |
| Acrescentar | Tabela Secção 10 de `methods_freeze.md` (diferenças factuais) |

---

## Plano detalhado de atualização (ordem sugerida)

1. **Inserir secção “Two evaluation protocols”** no início dos Methods — diagrama braço A vs braço B.
2. **Substituir** a descrição do método principal por OriginalRFECA TARGET-WISE (Secções 4.2 + 5 do freeze).
3. **Mover** RFECA_SVR(k=*) para “legacy / paper campaign” ou appendix, se ainda reportado.
4. **Atualizar missingness seeds** (legacy vs v2).
5. **Corrigir métricas** — RV só baselines; sem RV no OriginalRFECA.
6. **Atualizar classificação** — nesting distinto + caveats F1.
7. **Reescrever estatística** — testes válidos vs comparações descritivas.
8. **Adicionar** tabela Rodrigues → implementação final (sem juízo).
9. **Atualizar** runtime / hardware / workers / freeze_id.
10. **Sincronizar** números com `artifacts/final_analysis/` e freeze REPORT_*.

---

## Matriz rápida

| Tópico Methods (artigo/proxy) | Status |
|---|---|
| Datasets METABRIC 1608×50 | Compatível |
| CPTAC como validação TARGET-WISE | Incompatível |
| MCAR/MAR exact-count + MAR formula | Compatível |
| Seeds únicos para tudo | Incompatível |
| Mean/KNN/MissForest CV | Compatível |
| RFECA-k* como método principal | Incompatível |
| OriginalRFECA TARGET-WISE | Ausente → Acrescentar |
| RV para todos | Incompatível |
| Stats Wilcoxon = RFECA vs MissForest freeze | Incompatível |
| F1 cross-protocol sem caveat | Incompatível |
| Reprodutibilidade freeze v0.3.1 | Ausente → Acrescentar |
