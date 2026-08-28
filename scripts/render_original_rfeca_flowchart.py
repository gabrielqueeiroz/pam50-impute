#!/usr/bin/env python3
"""
Render OriginalRFECA TARGET-WISE flowchart (PNG/PDF) + Mermaid source.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "paper_results_original_rfeca"


def _box(ax, xy, w, h, text, *, fc="#E8F1F8", ec="#1F4E79", fontsize=7.8, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#1A1A1A",
    )


def _arrow(ax, x, y1, y2):
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1),
    )


def render_flowchart() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 12.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.2)
    ax.axis("off")
    fig.suptitle(
        "OriginalRFECA TARGET-WISE — leakage-safe pipeline",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    cx = 5.0
    w = 7.4
    x0 = cx - w / 2

    items = [
        (14.2, 0.7, "Complete METABRIC PAM50 matrix X\n(all values finite)", "#D9EAD3", True),
        (
            13.2,
            0.7,
            "Persisted mask M (MCAR / MAR)\nseed_scheme = v2 · base_seed = 42",
            "#FFF2CC",
            False,
        ),
        (
            11.95,
            0.95,
            "For each target gene g ∈ PAM50:\nTARGET-WISE matrix — NaNs only on mask positions of g\nPredictors = original complete matrix (never imputed)",
            "#E8F1F8",
            False,
        ),
        (
            10.75,
            0.75,
            "Candidate pool = other genes (rule: full_matrix)\nmax_candidates = 49 · target excluded",
            "#E8F1F8",
            False,
        ),
        (
            9.6,
            0.75,
            "Training labels = rows with g observed\nMasked cells of g = external holdout (never enter fit)",
            "#FCE4D6",
            False,
        ),
        (
            8.45,
            0.7,
            "Inner validation: KFold(k=5) on observed rows\nselection_protocol = leakage_safe",
            "#EAD1DC",
            False,
        ),
        (
            7.0,
            1.15,
            "On each training fold only:\n1. Rank predictors by |Pearson r| with g\n2. Evaluate correlation prefixes\n3. sklearn RFE (estimator = linear SVR)\n4. Fit SVR · score OOF on validation fold",
            "#E8F1F8",
            False,
        ),
        (
            5.7,
            0.7,
            "Select best prefix length\nby mean OOF RMSE across folds",
            "#D0E2F3",
            False,
        ),
        (
            4.55,
            0.75,
            "Final refit on all observed rows:\nPearson → RFE → linear SVR  (use_scaler = False)",
            "#D9EAD3",
            False,
        ),
        (
            3.35,
            0.8,
            "Impute only masked cells of g\nPredictors from original X\nNo SimpleImputer · no chaining across genes",
            "#FCE4D6",
            False,
        ),
        (
            2.15,
            0.7,
            "Metrics on masked cells (RMSE, MAE)\nAggregate over genes × replications × rates",
            "#FFF2CC",
            False,
        ),
        (1.05, 0.65, "DONE.json · FREEZE · paper tables / figures", "#D9EAD3", True),
    ]

    mids = []
    for y, hh, text, fc, bold in items:
        _box(ax, (x0, y), w, hh, text, fc=fc, bold=bold)
        mids.append((y, hh))

    for i in range(len(mids) - 1):
        y1, _h1 = mids[i]
        y2, h2 = mids[i + 1]
        _arrow(ax, cx, y1, y2 + h2)

    ax.text(
        0.35,
        0.35,
        "Freeze v0.3.0-original-rfeca-targetwise  ·  evaluation: repeated_mask_holdout",
        fontsize=7,
        style="italic",
        color="#444444",
    )

    legend_items = [
        ("#D9EAD3", "Data / final model"),
        ("#FFF2CC", "Masking / evaluation"),
        ("#FCE4D6", "Leakage guards"),
        ("#EAD1DC", "Inner CV"),
        ("#E8F1F8", "Selection / fit"),
    ]
    lx, ly = 0.35, 14.55
    for i, (c, lab) in enumerate(legend_items):
        ax.add_patch(
            FancyBboxPatch(
                (lx + i * 1.9, ly),
                0.28,
                0.22,
                boxstyle="round,pad=0.01",
                facecolor=c,
                edgecolor="#666",
                linewidth=0.6,
            )
        )
        ax.text(lx + i * 1.9 + 0.35, ly + 0.1, lab, fontsize=6, va="center")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT / "flowchart_original_rfeca.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "flowchart_original_rfeca.pdf", bbox_inches="tight")
    plt.close(fig)


MERMAID = """# OriginalRFECA TARGET-WISE — flowchart

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
"""


def main() -> int:
    render_flowchart()
    (OUT / "flowchart_original_rfeca.md").write_text(MERMAID, encoding="utf-8")
    print(f"Wrote {OUT / 'flowchart_original_rfeca.png'}")
    print(f"Wrote {OUT / 'flowchart_original_rfeca.pdf'}")
    print(f"Wrote {OUT / 'flowchart_original_rfeca.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
