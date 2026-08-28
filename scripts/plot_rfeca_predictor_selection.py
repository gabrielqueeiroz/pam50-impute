#!/usr/bin/env python3
"""Generate OriginalRFECA predictor-selection overview figures (3 panels).

Writes:
  - fig_rfeca_predictor_selection.{png,pdf} — panel C = histogram
  - fig_rfeca_predictor_selection_dot.{png,pdf} — panel C = horizontal dot plot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "final_analysis"
FIG = FINAL / "figures"
TITLE = "Predictor selection analysis for OriginalRFECA on the METABRIC cohort"


def _load() -> tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame, int]:
    top = pd.read_csv(FINAL / "rfeca_top_predictors.csv").head(15)
    dist = pd.read_csv(FINAL / "rfeca_n_predictors_distribution.csv")
    n_col = "n_predictors_selected" if "n_predictors_selected" in dist.columns else "n_predictors"
    genes = pd.read_csv(FINAL / "stats_final" / "gene_difficulty_overall_original_rfeca.csv")
    n_events = int(dist["count"].sum())
    return top, dist, n_col, genes, n_events


def _panel_a(ax, top: pd.DataFrame, n_events: int) -> None:
    top15 = top.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top15))
    counts = top15["count_as_predictor"].to_numpy()
    pct = 100.0 * counts / n_events
    ax.barh(y, pct, color="#4C78A8", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(top15["gene"].tolist(), fontsize=8)
    ax.set_xlabel("Selection frequency across fitted models (%)")
    ax.set_title("A. Top-15 predictors")
    ax.set_xlim(0, max(pct) * 1.18)
    for yi, p in zip(y, pct):
        ax.text(p + 0.35, yi, f"{p:.1f}", va="center", fontsize=6.5, color="#333")


def _panel_b(ax, dist: pd.DataFrame, n_col: str) -> None:
    ax.bar(dist[n_col], dist["count"], color="#F58518", width=0.85, align="center")
    ax.set_xlabel("Subset size (n predictors)")
    ax.set_ylabel("Number of fitted models")
    ax.set_title("B. Selected subset size")
    ax.set_xticks(range(int(dist[n_col].min()), int(dist[n_col].max()) + 1, 3))
    mean_n = float(np.average(dist[n_col], weights=dist["count"]))
    ax.axvline(mean_n, color="#333", ls="--", lw=1.0, label=f"mean = {mean_n:.1f}")
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def _panel_c_hist(ax, genes: pd.DataFrame) -> None:
    vals = genes["n_pred_mean"].to_numpy(dtype=float)
    bins = np.arange(0.5, 25.5, 1.0)
    ax.hist(vals, bins=bins, color="#54A24B", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean n predictors per gene")
    ax.set_ylabel("Number of target genes")
    ax.set_title("C. Average subset size per target gene")
    ax.axvline(vals.mean(), color="#333", ls="--", lw=1.0, label=f"mean = {vals.mean():.1f}")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xticks(range(0, 25, 5))


def _panel_c_dot(ax, genes: pd.DataFrame) -> None:
    g = genes.sort_values("n_pred_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(g))
    x = g["n_pred_mean"].to_numpy(dtype=float)
    ax.hlines(y, 0, x, color="#c8c8c8", lw=0.7, zorder=1)
    ax.scatter(x, y, s=22, color="#54A24B", zorder=2, edgecolors="white", linewidths=0.3)
    mean_n = float(x.mean())
    ax.axvline(mean_n, color="#333", ls="--", lw=1.0, label=f"mean = {mean_n:.1f}", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(g["gene"].tolist(), fontsize=5.5)
    ax.set_xlabel("Mean n predictors per gene")
    ax.set_title("C. Average subset size per target gene")
    ax.set_xlim(0, max(25.0, x.max() + 1.0))
    ax.set_ylim(-1, len(g))
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.tick_params(axis="y", length=0)


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        out = FIG / f"{stem}{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def write_histogram_version(top, dist, n_col, genes, n_events) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    _panel_a(axes[0], top, n_events)
    _panel_b(axes[1], dist, n_col)
    _panel_c_hist(axes[2], genes)
    fig.suptitle(TITLE, fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_rfeca_predictor_selection")


def write_ab_only_version(top, dist, n_col, n_events) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    _panel_a(axes[0], top, n_events)
    _panel_b(axes[1], dist, n_col)
    fig.suptitle(TITLE, fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_rfeca_predictor_selection_ab")


def write_dot_version(top, dist, n_col, genes, n_events) -> None:
    # A|B stacked on the left; C full-height on the right for 50 gene labels.
    fig = plt.figure(figsize=(11.5, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1], wspace=0.35, hspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])
    _panel_a(ax_a, top, n_events)
    _panel_b(ax_b, dist, n_col)
    _panel_c_dot(ax_c, genes)
    fig.suptitle(TITLE, fontsize=10, y=0.995)
    _save(fig, "fig_rfeca_predictor_selection_dot")


def main() -> None:
    top, dist, n_col, genes, n_events = _load()
    write_histogram_version(top, dist, n_col, genes, n_events)
    write_ab_only_version(top, dist, n_col, n_events)
    write_dot_version(top, dist, n_col, genes, n_events)


if __name__ == "__main__":
    main()
