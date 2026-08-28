# OriginalRFECA TARGET-WISE — flowchart

Protocol frozen as `v0.3.0-original-rfeca-targetwise`.

Render the Mermaid block at [mermaid.live](https://mermaid.live) or in GitHub/Quarto.  
Figures: `flowchart_original_rfeca.png` / `.pdf`.

```mermaid
flowchart TD
  A([Complete METABRIC PAM50 matrix X<br/>all values finite]) --> B{{Persisted mask M<br/>MCAR or MAR · seed v2 · base_seed 42}}
  B --> C[For each target gene g]
  C --> D[Build TARGET-WISE matrix<br/>NaNs only on M positions of g<br/>Predictors = original complete columns]
  D --> E[Candidate pool: other genes<br/>full_matrix · max_candidates = 49]
  E --> F[Observed rows of g = train labels<br/>Masked cells of g = external holdout]
  F --> G{{Inner KFold k=5<br/>on observed rows only}}
  G --> H[Per training fold only:<br/>1. Absolute Pearson ranking<br/>2. Correlation prefixes<br/>3. sklearn RFE + linear SVR<br/>4. OOF predict on val fold]
  H --> I[Choose best prefix length<br/>by mean OOF RMSE]
  I --> J[Final refit on all observed rows<br/>Pearson → RFE → linear SVR<br/>use_scaler = false]
  J --> K[Impute masked cells of g only<br/>predictors from original X<br/>no SimpleImputer · no chaining]
  K --> L{More genes?}
  L -->|yes| C
  L -->|no| M([RMSE / MAE on masked cells<br/>aggregate over genes · reps · rates])
  M --> N([FREEZE · paper tables/figures])

  classDef data fill:#D9EAD3,stroke:#333,color:#111
  classDef mask fill:#FFF2CC,stroke:#333,color:#111
  classDef guard fill:#FCE4D6,stroke:#333,color:#111
  classDef cv fill:#EAD1DC,stroke:#333,color:#111
  classDef step fill:#E8F1F8,stroke:#333,color:#111
  class A,J,N data
  class B,M mask
  class F,K guard
  class G cv
  class C,D,E,H,I step
```

## Guarantees encoded in the flow
- Masked target values never enter correlation, RFE, or final SVR fit
- Predictors are never mean-filled or chained from other imputed genes
- RFE during selection runs only on each inner-train fold; final RFE/SVR uses all observed (non-masked) rows
