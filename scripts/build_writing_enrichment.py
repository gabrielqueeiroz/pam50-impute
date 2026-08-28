"""Derived writing analyses from frozen final artifacts only.

No new benchmarks, no imputation recalculation, no alteration of
existing numeric result tables. Baseline per-gene RMSE is absent from
artifacts → heatmap columns Mean/KNN/MissForest are explicit NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]
METHODS = ["Mean", "KNN", "MissForest", "OriginalRFECA"]


def load_rfeca_gene_means() -> pd.DataFrame:
    frames = []
    for p in sorted(
        (ROOT / "artifacts/original_rfeca_reduced_metabric").glob("per_gene_all_*.csv")
    ):
        frames.append(pd.read_csv(p))
    g = pd.concat(frames, ignore_index=True)
    agg = (
        g.groupby("gene")
        .agg(
            OriginalRFECA_rmse=("rmse", "mean"),
            OriginalRFECA_mae=("mae", "mean"),
            OriginalRFECA_rmse_std=("rmse", "std"),
            n_slots=("rmse", "size"),
            n_pred_mean=("n_predictors_selected", "mean"),
        )
        .reset_index()
    )
    return agg


def build_heatmap_matrix(rfeca: pd.DataFrame) -> pd.DataFrame:
    """Only OriginalRFECA has gene-level RMSE in final artifacts."""
    rows = []
    for gene in PAM50:
        r = rfeca[rfeca.gene == gene]
        if len(r) == 0:
            rmse_r, mae_r = np.nan, np.nan
        else:
            rmse_r = float(r.iloc[0]["OriginalRFECA_rmse"])
            mae_r = float(r.iloc[0]["OriginalRFECA_mae"])
        rows.append(
            {
                "gene": gene,
                "Mean_rmse": np.nan,
                "KNN_rmse": np.nan,
                "MissForest_rmse": np.nan,
                "OriginalRFECA_rmse": rmse_r,
                "Mean_mae": np.nan,
                "KNN_mae": np.nan,
                "MissForest_mae": np.nan,
                "OriginalRFECA_mae": mae_r,
                "data_availability": "OriginalRFECA_only",
                "note": (
                    "Mean/KNN/MissForest per-gene RMSE not stored in final "
                    "baseline artifacts (fold-level aggregates only)."
                ),
            }
        )
    df = pd.DataFrame(rows)
    # order hardest → easiest by available global mean (= OriginalRFECA)
    df["rmse_global_available"] = df["OriginalRFECA_rmse"]
    df = df.sort_values("rmse_global_available", ascending=False, na_position="last")
    return df


def plot_heatmap(df: pd.DataFrame, zscore: bool, out_path: Path) -> None:
    mat = df[[f"{m}_rmse" for m in METHODS]].to_numpy(dtype=float)
    genes = df["gene"].tolist()

    if zscore:
        # z-score across methods per gene; skip genes with <2 observed methods
        z = np.full_like(mat, np.nan)
        for i in range(mat.shape[0]):
            row = mat[i]
            obs = np.isfinite(row)
            if obs.sum() >= 2:
                mu = row[obs].mean()
                sd = row[obs].std(ddof=0)
                if sd > 0:
                    z[i, obs] = (row[obs] - mu) / sd
                else:
                    z[i, obs] = 0.0
            # with only 1 method, z-score undefined → leave NaN (explicit)
        data = z
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color="#d0d0d0")
        title = (
            "Per-gene RMSE z-score across methods\n"
            "(NaN = unavailable or undefined with <2 methods; not interpolated)"
        )
        cbar_label = "z-score (lower RMSE → cooler)"
        # with only OriginalRFECA, entire matrix is NaN — still plot with note
        vmin, vmax = -2, 2
        if np.isfinite(data).any():
            lim = np.nanmax(np.abs(data))
            lim = max(lim, 1.0)
            vmin, vmax = -lim, lim
    else:
        data = mat
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad(color="#d0d0d0")
        title = (
            "Mean RMSE by gene × method (METABRIC PAM50)\n"
            "Gray = not available in final artifacts (not interpolated)"
        )
        cbar_label = "RMSE"
        finite = data[np.isfinite(data)]
        vmin = float(finite.min()) if len(finite) else 0.0
        vmax = float(finite.max()) if len(finite) else 1.0

    fig_h = max(10, 0.22 * len(genes) + 2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=30, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_title(title, fontsize=11)
    if zscore and not np.isfinite(data).any():
        ax.text(
            0.5,
            0.5,
            "Z-score undefined:\nonly OriginalRFECA has\ngene-level RMSE in\nfinal artifacts",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5"),
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    ax.legend(
        handles=[Patch(facecolor="#d0d0d0", edgecolor="k", label="Unavailable (NaN)")],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def gene_difficulty_report(df: pd.DataFrame, rfeca: pd.DataFrame) -> None:
    # Expand with ranking where possible
    lines = []
    lines.append("# Gene difficulty report\n\n")
    lines.append("## Disponibilidade dos dados\n\n")
    lines.append(
        "Nos artefactos finais, **apenas OriginalRFECA** possui RMSE/MAE por gene "
        "(`per_gene_all_*.csv`). Mean, KNN e MissForest reportam apenas métricas "
        "agregadas por fold/réplica. Colunas desses métodos no heatmap são **NaN "
        "explícitos** (sem interpolação).\n\n"
    )
    lines.append(
        "Consequência: rankings cross-method, Δ melhor−pior e CV entre métodos "
        "**por gene** não são calculáveis a partir dos artefactos finais. "
        "A classificação abaixo usa a dificuldade sob OriginalRFECA e, quando "
        "aplicável, vitórias descritivas ao nível de cenário (não gene).\n\n"
    )

    hard = rfeca.sort_values("OriginalRFECA_rmse", ascending=False)
    q75 = hard["OriginalRFECA_rmse"].quantile(0.75)
    q25 = hard["OriginalRFECA_rmse"].quantile(0.25)
    hard_genes = hard[hard["OriginalRFECA_rmse"] >= q75]["gene"].tolist()
    easy_genes = hard[hard["OriginalRFECA_rmse"] <= q25]["gene"].tolist()

    # stability across rates for RFECA
    frames = []
    for p in sorted(
        (ROOT / "artifacts/original_rfeca_reduced_metabric").glob("per_gene_all_*.csv")
    ):
        frames.append(pd.read_csv(p))
    g = pd.concat(frames, ignore_index=True)
    by_cell = g.groupby(["gene", "mechanism", "missing_rate"])["rmse"].mean().reset_index()
    stab = by_cell.groupby("gene")["rmse"].agg(["mean", "std"]).reset_index()
    stab["cv"] = stab["std"] / stab["mean"]
    # low CV across mech×rate → relatively stable difficulty
    indifferent_proxy = stab.nsmallest(10, "cv")["gene"].tolist()

    lines.append("## RMSE / MAE por gene (OriginalRFECA)\n\n")
    show = hard.copy()
    show = show.rename(
        columns={
            "OriginalRFECA_rmse": "RMSE",
            "OriginalRFECA_mae": "MAE",
            "OriginalRFECA_rmse_std": "RMSE_std",
        }
    )
    # markdown table top/bottom
    lines.append("### Mais difíceis (top 15)\n\n")
    lines.append(_md(show.head(15)[["gene", "RMSE", "MAE", "RMSE_std", "n_pred_mean"]]))
    lines.append("\n\n### Mais fáceis (bottom 10)\n\n")
    lines.append(_md(show.tail(10)[["gene", "RMSE", "MAE", "RMSE_std", "n_pred_mean"]]))

    lines.append("\n\n## Respostas pedidas\n\n")
    lines.append("### Consistentemente difíceis para todos os métodos?\n\n")
    lines.append(
        "**Não verificável** nos artefactos finais (falta RMSE gene-level dos baselines). "
        "Sob OriginalRFECA, genes no quartil superior de dificuldade (Q4) incluem: "
        + ", ".join(hard_genes)
        + ". Estes são os melhores candidatos a 'difíceis', mas não se pode afirmar "
        "que Mean/KNN/MissForest falham nos mesmos genes sem os dados.\n\n"
    )

    lines.append("### Particularmente favorecidos pelo OriginalRFECA?\n\n")
    lines.append(
        "**Não calculável ao nível do gene** (sem RMSE gene-level de MissForest/KNN/Mean). "
        "Ao nível de **cenário** (descritivo): OriginalRFECA vence RMSE em 7/8 células; "
        "maiores ganhos vs MissForest em MAR 20–30% e MCAR 20–30% "
        "(`stats_final/effect_sizes_rfeca_vs_missforest.csv`).\n\n"
    )

    lines.append("### Particularmente favorecidos pelo MissForest?\n\n")
    lines.append(
        "**Não calculável ao nível do gene.** Ao nível de cenário: MissForest tem "
        "menor RMSE médio apenas em **MAR 5%** (Δ RFECA−MF ≈ +0.012). "
        "Em Macro-F1, MissForest lidera por margem mínima em MCAR/MAR 10%.\n\n"
    )

    lines.append("### Praticamente indiferentes ao método?\n\n")
    lines.append(
        "**Não calculável cross-method por gene.** Proxy sob OriginalRFECA: genes com "
        "menor CV do RMSE ao longo de mecanismo×taxa (dificuldade estável, não "
        "indiferença entre imputadores): "
        + ", ".join(indifferent_proxy)
        + ".\n\n"
    )

    # detailed CSV companion
    detail = df.copy()
    detail["best_method_rmse"] = np.nan
    detail["worst_method_rmse"] = np.nan
    detail["delta_best_worst"] = np.nan
    detail["cv_across_methods"] = np.nan
    detail["ranking_rmse"] = detail.apply(
        lambda r: "OriginalRFECA (only available)"
        if np.isfinite(r["OriginalRFECA_rmse"])
        else "all missing",
        axis=1,
    )
    detail.to_csv(OUT / "gene_method_heatmap.csv", index=False)

    lines.append("## Artefacto tabular\n\n")
    lines.append("`gene_method_heatmap.csv` — matriz gene×método com NaNs explícitos.\n")
    (OUT / "gene_difficulty_report.md").write_text("".join(lines), encoding="utf-8")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


BIO_NOTES = {
    "MKI67": (
        "Antígeno Ki-67; marcador clássico de proliferação celular. "
        "Associado a ciclo celular / fração de crescimento tumoral; "
        "não é receptor hormonal nem HER2. Papel: índice proliferativo em mama."
    ),
    "NDC80": (
        "Componente do complexo cinetocoro NDC80; essencial para segregação "
        "cromossómica na mitose. Associado a ciclo celular/proliferação. "
        "Não é receptor hormonal nem HER2."
    ),
    "UBE2T": (
        "Enzima conjugadora de ubiquitina; envolvida em reparo de DNA "
        "(via FANCL) e turnover proteico. Ligada a proliferação/manutenção "
        "genómica; não é receptor hormonal nem HER2."
    ),
    "CEP55": (
        "Proteína de centrossomo/citocinese; regula abscisão celular. "
        "Fortemente ligada a ciclo celular e proliferação. "
        "Não é receptor hormonal nem HER2."
    ),
    "PTTG1": (
        "Securina (pituitary tumor-transforming 1); regula separação de "
        "cromátides-irmãs. Associada a ciclo celular, proliferação e "
        "instabilidade genómica. Não é receptor hormonal nem HER2."
    ),
    "KIF2C": (
        "Cinesina (MCAK); despolimeriza microtúbulos no cinetocoro. "
        "Função mitótica / ciclo celular / proliferação. "
        "Não é receptor hormonal nem HER2."
    ),
    "UBE2C": (
        "Ubiquitina-conjugase do ciclo anáfase (APC/C); degradação de "
        "ciclina B e saída da mitose. Marcador de proliferação/ciclo. "
        "Não é receptor hormonal nem HER2."
    ),
    "EXO1": (
        "Exonucleases 1; reparo de DNA (mismatch/recombinação). "
        "Associada a manutenção genómica e frequentemente coexpressa com "
        "assinaturas proliferativas. Não é receptor hormonal nem HER2."
    ),
    "RRM2": (
        "Subunidade da ribonucleotídeo redutase; síntese de dNTPs para "
        "replicação do DNA. Ligada a ciclo S / proliferação. "
        "Não é receptor hormonal nem HER2."
    ),
    "CCNE1": (
        "Ciclina E1; transição G1/S. Driver clássico de ciclo celular e "
        "proliferação; amplificação relatada em subtipos agressivos. "
        "Não é receptor hormonal nem HER2 (embora relevante em Basal/HG)."
    ),
}


def biological_notes() -> None:
    pred = pd.read_csv(OUT / "rfeca_top_predictors.csv").head(10)
    lines = []
    lines.append("# Predictor gene biological notes (leve)\n\n")
    lines.append(
        "Notas qualitativas com base em conhecimento estabelecido da literatura "
        "PAM50 / biologia mamária. **Sem** enriquecimento funcional e **sem** "
        "consulta a bases externas nesta análise.\n\n"
    )
    for _, r in pred.iterrows():
        g = r["gene"]
        c = int(r["count_as_predictor"])
        note = BIO_NOTES.get(
            g,
            "Gene PAM50; função não detalhada aqui além do papel no painel.",
        )
        lines.append(f"## {g} (frequência como preditor: {c})\n\n")
        lines.append(note + "\n\n")

    lines.append("---\n\n## Existe padrão biológico evidente?\n\n")
    lines.append(
        "**Sim, com cautela:** os 10 preditores mais frequentes são "
        "predominantemente genes de **proliferação / ciclo celular / mitose / "
        "replicação e reparo de DNA** (MKI67, NDC80, CEP55, PTTG1, KIF2C, "
        "UBE2C, UBE2T, EXO1, RRM2, CCNE1). "
        "Não dominam nesta lista os receptores hormonais clássicos (ESR1, PGR) "
        "nem ERBB2/HER2 — esses aparecem mais como **alvos** difíceis de imputar "
        "do que como preditores top (ver `gene_difficulty_report.md`).\n\n"
    )
    lines.append(
        "**Interpretação cautelosa:** o padrão é compatível com a estrutura de "
        "correlação do PAM50 (módulo proliferativo coeso), não com uma descoberta "
        "biológica nova. A seleção Pearson+RFE tende a escolher genes "
        "altamente correlacionados com o alvo; no METABRIC isso frequentemente "
        "cai no bloco proliferativo. "
        "**Não** se deve inferir causalidade, especificidade terapêutica ou "
        "validação clínica a partir destas frequências.\n\n"
    )
    lines.append(
        "Se o critério for um padrão além de 'proliferação', **não há evidência "
        "clara** nos resultados de um segundo eixo dominante (p.ex. invasão ou "
        "diferenciação luminal) entre os top-10 preditores.\n"
    )
    (OUT / "predictor_gene_biological_notes.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def strengths_limitations() -> None:
    text = """# Strengths vs Limitations

## Strengths

| contribuição | evidência experimental | figura | tabela | importância para o artigo |
|---|---|---|---|---|
| Protocolo OriginalRFECA TARGET-WISE leakage-safe (sem SimpleImputer/chaining) | 40/40 slots: coverage=1.0, fallback=0, n_predictor_nans=0 | flowchart OriginalRFECA | `methodology_audit.md`, FREEZE manifest | Alta — validade metodológica |
| RMSE competitivo/superior vs Mean/KNN/MissForest (descritivo) | Melhor RMSE em 7/8 células; Δ vs MF até −9.3% (MAR 30%) | `fig_rmse_by_missingness.png`, bars | `central/headline_rmse_f1.csv`, effect sizes | Alta — resultado principal |
| Estabilidade do RMSE entre taxas (esp. MAR) | Span RMSE OriginalRFECA pequeno vs degradação de KNN/MF | `fig_rmse_by_missingness.png` | `results_tables_display.csv` | Alta — diferenciação |
| Seleção interpretável (preditores + Jaccard) | ~21.6 preditores/gene; Jaccard ~0.68; top proliferativos | (opcional heatmap z) | `rfeca_top_predictors.csv`, gene reports | Média — discussão |
| Reprodutibilidade / freeze | Seeds v2, mask hashes, config snapshot, requirements | — | FREEZE/, `methods_checklist.md` | Alta — Methods |
| Stats formais entre baselines (mesmo protocolo) | Friedman + Wilcoxon–Holm Mean/KNN/MF | — | `stats_final/A_*.csv` | Média — suporte baselines |
| Separação explícita válida vs descritiva | Sem p-values inválidos RFECA vs baselines | — | `stats_final/B_*.csv`, `one_page_results.md` | Alta — credibilidade |

## Limitations

| limitação | impacto potencial | mitigação adotada | possibilidade de trabalho futuro |
|---|---|---|---|
| Protocolos distintos (holdout TARGET-WISE vs CV shared-mask) | Impede inferência confirmatória RFECA vs baselines | Comparação descritiva; disclaimer; stats A só baselines | Redesign paired shared-mask / mesmo n_reps |
| Ausência de comparação estatística confirmatória para OriginalRFECA vs baselines | Revisores podem pedir p-values | Não reportar p inválidos; IC bootstrap descritivo | Experimento pareado dedicado |
| Custo computacional (~51.5 h, 16 workers) | Barreira de adoção / escala | Paralelismo gene-nível; fingerprint-identical | Early-stop RFE, cache correlações, approx. prefixes |
| Apenas PAM50 (50 genes) | Generalização a transcriptoma denso incerta | Escopo explícito; painel clinicamente usado | Extender a painéis maiores / RNA-seq |
| Apenas câncer de mama (METABRIC; CPTAC auxiliar) | Validade externa limitada | Declarar domínio | Outras histologias / multi-câncer |
| Ausência de validação clínica | Não prova utilidade terapêutica | Foco em erro de imputação + F1 PAM50 | Estudos clínicos / utilidade em pipelines reais |
| Ausência de OriginalRFACA no freeze principal | Comparação RFE vs RFA incompleta | RFACA só smoke/audit | Campanha RFACA TARGET-WISE completa |
| Protocolo target-wise (preditores da matriz completa) | Não modela missingness conjunta realista nos preditores | Contrato metodológico anti-leakage explícito | Variante com preditores parcialmente observados |
| Sem RMSE gene-level para baselines | Heatmap/análise gene×método incompleta | NaNs explícitos; sem interpolação | Exportar métricas por gene nos baselines |
| Sem RV para OriginalRFECA | Não compara preservação de correlação | Reportar RV só baselines | Calcular RV pós-imputação TARGET-WISE |
| Classificação: nesting diferente | F1 cross-protocol pouco comparável | Caveat; F1 secundário | Aninhar RFECA no CV de classificação |
| Missingness MCAR/MAR simulados (não MNAR clínico) | Mecanismos reais podem diferir | Dois mecanismos + exact-count | MNAR / missingness real de plataforma |
"""
    (OUT / "limitations_vs_strengths.md").write_text(text, encoding="utf-8")


def reviewer_questions() -> None:
    text = """# Possíveis perguntas de um revisor relacionadas aos resultados

Respostas baseadas **exclusivamente** nos resultados/artefactos obtidos. Sem evidências inventadas.

---

### L1 — Protocolos distintos
**Pergunta:** Como podem comparar OriginalRFECA com MissForest se os protocolos de avaliação diferem?

**Resposta:** Não reivindicamos equivalência formal. A comparação é **descritiva** (ΔRMSE/MAE, rankings, win counts, IC bootstrap). Testes confirmatórios (Friedman/Wilcoxon–Holm) aplicam-se apenas entre Mean/KNN/MissForest no protocolo CV shared-mask (`stats_final/A_*`). OriginalRFECA usa `repeated_mask_holdout` TARGET-WISE com 5 reps e seeds v2.

---

### L2 — Sem p-values RFECA vs MissForest
**Pergunta:** Por que não há Wilcoxon pareado OriginalRFECA vs MissForest?

**Resposta:** Unidades experimentais não são emparelháveis (máscaras/seeds/n_reps/protocolos diferentes). Reportar p-values seria metodologicamente inválido; por isso `p_value=NOT_REPORTED` em `stats_final/B_*`.

---

### L3 — Custo computacional
**Pergunta:** O método é praticável dado ~51.5 h de wall time?

**Resposta:** O grid completo (40 slots, 50 genes, RFE+SVR) custou ~51.5 h com 16 gene-workers nesta máquina. Microbenchmarks mostraram speedup até ~4.7× com fingerprints idênticos. É custo de evidência metodológica; otimizações (early-stop, cache) ficam como trabalho futuro — não foram aplicadas ao freeze.

---

### L4 — Apenas PAM50
**Pergunta:** Os resultados generalizam para transcriptomas densos?

**Resposta:** Não foi testado. Todos os resultados de imputação do freeze são em 50 genes PAM50 METABRIC. Não há evidência experimental no pacote final para p≫50.

---

### L5 — Apenas mama
**Pergunta:** Funciona fora de câncer de mama?

**Resposta:** O freeze OriginalRFECA é METABRIC. CPTAC 2C aparece na campanha six-imputer legado, não como freeze TARGET-WISE. Não há validação multi-câncer no pacote final.

---

### L6 — Validação clínica
**Pergunta:** A melhoria de RMSE traduz-se em benefício clínico?

**Resposta:** Não avaliámos outcomes clínicos. Métricas: RMSE/MAE (e F1 PAM50 secundário). Qualquer afirmação de utilidade clínica estaria além dos dados.

---

### L7 — Ausência de RFACA
**Pergunta:** Por que não comparar OriginalRFACA?

**Resposta:** O freeze principal é OriginalRFECA apenas. RFACA existe no código/smoke, mas não no grid final de 40 slots. Não há números RFACA TARGET-WISE comparáveis no pacote final.

---

### L8 — TARGET-WISE “não é imputação real”
**Pergunta:** Usar a matriz completa como preditores não é irrealista?

**Resposta:** É uma escolha metodológica explícita (`input_protocol=target_wise_complete_predictors`) para evitar leakage/chaining. Avalia erro nas células mascaradas do alvo com preditores observados. Não pretende simular missingness conjunta em todos os genes; isso é limitação declarada, não um bug oculto.

---

### L9 — Heatmap gene×método incompleto
**Pergunta:** Por que Mean/KNN/MissForest estão em cinza no heatmap?

**Resposta:** Os artefactos finais dos baselines não armazenam RMSE por gene — só agregados por fold. Marcámos NaN sem interpolar. A dificuldade gene-level reportada refere-se ao OriginalRFECA.

---

### L10 — MAR 5% MissForest melhor
**Pergunta:** O método falha sob MAR a baixa taxa?

**Resposta:** Em MAR 5%, MissForest tem RMSE médio 0.629 vs 0.641 do OriginalRFECA (Δ descritivo +0.012). É a única célula RMSE em que MissForest vence. Em MAR ≥10% e em todo MCAR do grid, OriginalRFECA tem menor RMSE médio.

---

### L11 — F1 quase empatado
**Pergunta:** Se F1 quase não muda, qual a relevância prática?

**Resposta:** Em 5–10% as diferenças de Macro-F1 são pequenas; a 20–30% o OriginalRFECA aparece relativamente melhor, com caveat de nesting de classificação diferente. O resultado primário do estudo é **erro de imputação (RMSE)**, não F1.

---

### L12 — Reprodutibilidade
**Pergunta:** Os resultados são reprodutíveis?

**Resposta:** Freeze `v0.3.1-original-rfeca-targetwise` com mask hashes, seeds v2 auditados, config snapshot, requirements e 40/40 slots completos. Paralelismo gene-nível foi validado com fingerprints idênticos ao serial no microbenchmark.
"""
    (OUT / "reviewer_results_questions.md").write_text(text, encoding="utf-8")


def figures_inventory() -> None:
    text = """# Figures inventory

| figura | objetivo | mensagem principal | seção do artigo |
|---|---|---|---|
| `figures/fig_rmse_by_missingness.png` | Comparar RMSE vs taxa/mecanismo | OriginalRFECA estável e tipicamente mais baixo; MF único rival | Results — imputação |
| `figures/fig_mae_by_missingness.png` | Idem para MAE | Mesmo padrão do RMSE | Results — imputação |
| `figures/fig_rmse_bars_5_10_20_30.png` / `copied_comparison_fig05_*` | Barras RMSE por célula | Ranking visual Mean≫KNN>MF≳RFECA | Results |
| `figures/fig_macrof1_by_missingness.png` / `copied_comparison_fig03_*` | Macro-F1 vs missingness | Diferenças pequenas a 5–10%; RFECA melhor a 20–30% (caveat) | Results — classificação |
| `figures/copied_comparison_fig06_f1_bars.png` | Barras F1 | Comparação F1 lado a lado | Results — classificação |
| `figures/fig_rv_by_missingness_baselines.png` | RV só baselines | MF preserva melhor correlação entre baselines; RFECA sem RV | Results / Discussion |
| `figures/fig_stability_rmse_cv.png` | Estabilidade entre réplicas | Dispersão por método/taxa | Results / Supplement |
| `figures/gene_method_heatmap_rmse.png` | RMSE gene×método | Dificuldade por gene (RFECA); baselines NaN | Results — gene-level |
| `figures/gene_method_heatmap_rmse_zscore.png` | z-score por gene | Relativo entre métodos (indefinido se <2 métodos) | Supplement / Results |
| flowchart OriginalRFECA (paper_results_original_rfeca) | Protocolo TARGET-WISE | Leakage-safe pipeline | Methods |
| parallel speedup (computational_cost / parallel_benchmark) | Custo e paralelismo | Speedup com fingerprints idênticos | Methods / Supplement |

Canvas auxiliar: `central-results-comparison.canvas.tsx` (exploração, não figura do artigo).
"""
    (OUT / "figures_inventory.md").write_text(text, encoding="utf-8")


def article_readiness() -> None:
    text = """# Article readiness checklist

| item | estado | notas |
|---|---|---|
| Methods | **parcialmente pronto** | `methods_freeze.md` é SSoT; precisa reescrever Methods do artigo (proxy paper package ainda descreve RFECA-k*) — ver `methods_vs_article.md` |
| Results | **parcialmente pronto** | Números e figuras prontos (`one_page_results.md`, `central/`, heatmaps); texto Results ainda a redigir |
| Discussion | **parcialmente pronto** | Outlines (`discussion_outline.md`, `discussion_points.md`, strengths/limitations) prontos; prosa pendente |
| Conclusion | **parcialmente pronto** | Pontos em `conclusion_points.md`; prosa pendente |
| Figuras | **pronto** | Inventário atualizado; heatmaps gene×método adicionados (com NaNs explícitos) |
| Tabelas | **pronto** | `central/headline_rmse_f1.csv`, effect sizes, wins, gene tables |
| Referências cruzadas | **parcialmente pronto** | Paths internos consistentes; cruzar labels LaTeX/Word no manuscrito ainda pendente |
| Limitações | **pronto** | `limitations_vs_strengths.md` + `threats_to_validity.md` |
| Reprodutibilidade | **pronto** | Freeze v0.3.1, hashes, checklist Methods |

## Bloqueadores para submissão
1. Reescrita da seção Methods alinhada a `methods_freeze.md`.
2. Redação Results/Discussion/Conclusion a partir de `one_page_results.md`.
3. Decisão editorial sobre como apresentar heatmap com colunas baseline NaN (ou omitir colunas vazias e declarar limitação).
"""
    (OUT / "article_readiness.md").write_text(text, encoding="utf-8")


def main() -> None:
    rfeca = load_rfeca_gene_means()
    df = build_heatmap_matrix(rfeca)
    # gene_difficulty_report writes the enriched CSV
    gene_difficulty_report(df, rfeca)

    # reload enriched csv for plotting order
    df = pd.read_csv(OUT / "gene_method_heatmap.csv")
    df = df.sort_values("rmse_global_available", ascending=False, na_position="last")

    plot_heatmap(df, zscore=False, out_path=FIG / "gene_method_heatmap_rmse.png")
    plot_heatmap(df, zscore=True, out_path=FIG / "gene_method_heatmap_rmse_zscore.png")

    biological_notes()
    strengths_limitations()
    reviewer_questions()
    figures_inventory()
    article_readiness()

    idx = OUT / "INDEX.md"
    if idx.exists():
        t = idx.read_text(encoding="utf-8")
        marker = "## Writing enrichment (derived)"
        block = (
            "## Writing enrichment (derived)\n"
            "| Ordem | Ficheiro | Uso |\n"
            "|---:|---|---|\n"
            "| ★ | `one_page_results.md` | Results one-pager |\n"
            "| ★ | `gene_difficulty_report.md` | Dificuldade por gene |\n"
            "| ★ | `predictor_gene_biological_notes.md` | Notas biológicas top preditores |\n"
            "| ★ | `limitations_vs_strengths.md` | Strengths / Limitations |\n"
            "| ★ | `reviewer_results_questions.md` | Q&A de revisor (resultados) |\n"
            "| ★ | `figures_inventory.md` | Inventário de figuras |\n"
            "| ★ | `article_readiness.md` | Checklist de prontidão |\n"
            "| ★ | `figures/gene_method_heatmap_rmse.png` | Heatmap gene×método |\n"
            "| ★ | `gene_method_heatmap.csv` | Dados do heatmap (NaNs explícitos) |\n\n"
        )
        if marker not in t:
            needle = "Excluídos: RFECA-k*."
            if needle in t:
                pos = t.find(needle)
                end = t.find("\n\n", pos)
                if end != -1:
                    t = t[: end + 2] + block + t[end + 2 :]
                else:
                    t = t + "\n\n" + block
            else:
                t = block + t
            idx.write_text(t, encoding="utf-8")

    meta = {
        "heatmap_methods_with_data": ["OriginalRFECA"],
        "heatmap_methods_nan": ["Mean", "KNN", "MissForest"],
        "reason": "Baseline final artifacts lack per-gene RMSE",
        "interpolation": False,
        "zscore_note": (
            "Per-gene z-score across methods is undefined when fewer than "
            "2 methods have values; cells remain NaN."
        ),
    }
    (OUT / "gene_method_heatmap_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Wrote derived analyses to", OUT)


if __name__ == "__main__":
    main()
