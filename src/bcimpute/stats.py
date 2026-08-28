"""Paired nonparametric stats for imputer comparisons."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def replicate_means(raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean metric per (imputer, missing_rate, replicate) for paired tests."""
    return (
        raw.groupby(["imputer", "missing_rate", "replicate"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )


def friedman_table(
    raw: pd.DataFrame,
    value_col: str,
    *,
    cohort: str,
    mechanism: str,
    imputers: list[str] | None = None,
    higher_is_better: bool = True,
    skip_rate_zero: bool = True,
) -> pd.DataFrame:
    """
    Friedman omnibus test across imputers at each missing rate.

    Blocks = replicates (after averaging folds); treatments = imputers.
    Also reports mean rank per imputer (rank 1 = best given higher_is_better).
    """
    means = replicate_means(raw, value_col)
    if imputers is None:
        imputers = sorted(means["imputer"].unique())
    # stable order for reporting
    imputers = [i for i in imputers if i in set(means["imputer"])]

    rows: list[dict] = []
    for rate, g_rate in means.groupby("missing_rate"):
        if skip_rate_zero and float(rate) == 0.0:
            continue

        # Wide matrix: rows=replicate, cols=imputer
        wide = (
            g_rate[g_rate["imputer"].isin(imputers)]
            .pivot(index="replicate", columns="imputer", values="value")
            .reindex(columns=imputers)
            .dropna(axis=0, how="any")
        )
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue

        samples = [wide[c].to_numpy(float) for c in imputers]
        note = ""
        try:
            stat, p = friedmanchisquare(*samples)
            stat_f = float(stat)
            p_f = float(p)
        except ValueError as exc:
            stat_f = float("nan")
            p_f = float("nan")
            note = str(exc)

        # Ranks within each block (replicate): best gets rank 1
        if higher_is_better:
            ranks = wide.rank(axis=1, ascending=False, method="average")
        else:
            ranks = wide.rank(axis=1, ascending=True, method="average")
        mean_ranks = ranks.mean(axis=0)

        row: dict = {
            "cohort": cohort,
            "mechanism": mechanism,
            "missing_rate": float(rate),
            "metric": value_col,
            "higher_is_better": higher_is_better,
            "n_blocks": int(wide.shape[0]),
            "n_imputers": int(wide.shape[1]),
            "friedman_chi2": stat_f,
            "p_value": p_f,
            "note": note,
        }
        for name in imputers:
            row[f"mean_rank__{name}"] = float(mean_ranks[name])
            row[f"mean_value__{name}"] = float(wide[name].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def matched_pairs_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation for paired Wilcoxon.

    r = 1 - 4 * W_minus / (n * (n + 1)), where W_minus is the Wilcoxon
    signed-rank statistic on negative differences (y - x > 0 favors y if
    we pass diffs = x - y with higher x better... we use diffs = x - y).

    Positive r means x tends to be larger than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[np.isfinite(d)]
    # drop exact zeros (Wilcoxon convention)
    d = d[d != 0]
    n = len(d)
    if n < 1:
        return float("nan")
    abs_d = np.abs(d)
    ranks = abs_d.argsort().argsort() + 1.0  # 1..n
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    # Prefer formula from W_minus; equivalent via w_plus + w_minus = n(n+1)/2
    denom = n * (n + 1)
    if denom == 0:
        return float("nan")
    return float(1.0 - (4.0 * w_minus) / denom)


def paired_delta_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean paired difference mean(x - y).

    Resamples replicate-level differences with replacement.
    Returns (mean_delta, ci_low, ci_high).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean_delta = float(np.mean(d))
    rng = np.random.default_rng(int(random_state))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = float(np.mean(d[rng.integers(0, n, size=n)]))
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean_delta, float(lo), float(hi)


def _stable_seed(*parts: object) -> int:
    """Deterministic 31-bit seed from string parts (process-independent)."""
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    out = np.full(m, np.nan)
    if m == 0:
        return []
    valid = np.isfinite(p)
    if not valid.any():
        return out.tolist()
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    raw_adj = np.empty(len(order), dtype=float)
    for i, oi in enumerate(order):
        raw_adj[i] = min(1.0, float(p[oi]) * (len(order) - i))
    for i in range(1, len(raw_adj)):
        raw_adj[i] = max(raw_adj[i], raw_adj[i - 1])
    for i, oi in enumerate(order):
        out[oi] = raw_adj[i]
    return out.tolist()


def pairwise_wilcoxon_table(
    raw: pd.DataFrame,
    value_col: str,
    *,
    cohort: str,
    mechanism: str,
    imputers: list[str] | None = None,
    higher_is_better: bool = True,
    skip_rate_zero: bool = True,
    n_boot: int = 5000,
) -> pd.DataFrame:
    """
    All pairwise imputer comparisons at each missing rate.

    Unit: mean over folds within replicate (n_pairs ≈ 10).
    Also reports percentile bootstrap CI95 for mean(delta = a - b).
    """
    means = replicate_means(raw, value_col)
    if imputers is None:
        imputers = sorted(means["imputer"].unique())

    rows: list[dict] = []
    for rate, g_rate in means.groupby("missing_rate"):
        if skip_rate_zero and float(rate) == 0.0:
            continue
        for a, b in combinations(imputers, 2):
            va = g_rate.loc[g_rate["imputer"] == a].set_index("replicate")["value"]
            vb = g_rate.loc[g_rate["imputer"] == b].set_index("replicate")["value"]
            common = va.index.intersection(vb.index).sort_values()
            if len(common) < 2:
                continue
            x = va.loc[common].to_numpy(float)
            y = vb.loc[common].to_numpy(float)
            note = ""
            if np.allclose(x, y):
                p = 1.0
                r_rb = 0.0
                note = "identical vectors"
                stat = float("nan")
            else:
                try:
                    res = wilcoxon(x, y, zero_method="wilcox")
                    p = float(res.pvalue)
                    stat = float(res.statistic)
                except ValueError as exc:
                    p = float("nan")
                    stat = float("nan")
                    note = str(exc)
                r_rb = matched_pairs_rank_biserial(x, y)

            boot_seed = _stable_seed(cohort, mechanism, rate, value_col, a, b)
            delta_mean, ci_lo, ci_hi = paired_delta_bootstrap_ci(
                x, y, n_boot=n_boot, random_state=boot_seed
            )

            rows.append(
                {
                    "cohort": cohort,
                    "mechanism": mechanism,
                    "missing_rate": float(rate),
                    "metric": value_col,
                    "higher_is_better": higher_is_better,
                    "method_a": a,
                    "method_b": b,
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "delta_a_minus_b": delta_mean,
                    "delta_ci95_low": ci_lo,
                    "delta_ci95_high": ci_hi,
                    "wilcoxon_stat": stat,
                    "p_value": p,
                    "rank_biserial_r": r_rb,
                    "n_pairs": int(len(common)),
                    "n_boot": int(n_boot),
                    "note": note,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Holm within each (cohort, mechanism, metric, missing_rate) family
    adj_parts = []
    for _, g in df.groupby(
        ["cohort", "mechanism", "metric", "missing_rate"], sort=False
    ):
        gg = g.copy()
        gg["p_holm"] = holm_adjust(gg["p_value"].tolist())
        adj_parts.append(gg)
    return pd.concat(adj_parts, ignore_index=True)


def primary_contrasts_table(
    pairwise: pd.DataFrame,
    *,
    focal_methods: list[str] | None = None,
    baselines: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep contrasts of scientific interest: RFECA/MissForest vs Mean/KNN.
    """
    if pairwise.empty:
        return pairwise
    if focal_methods is None:
        focal_methods = [
            "MissForest",
            "RFECA_SVR(k=5)",
            "RFECA_SVR(k=10)",
            "RFECA_SVR(k=20)",
        ]
    if baselines is None:
        baselines = ["SimpleMean", "KNN(k=5,dist)"]

    rows = []
    for _, r in pairwise.iterrows():
        a, b = r["method_a"], r["method_b"]
        if (a in focal_methods and b in baselines) or (
            b in focal_methods and a in baselines
        ):
            # Orient so method_focal is listed as method_a when possible
            if a in baselines and b in focal_methods:
                # swap orientation (also flip bootstrap CI bounds)
                ci_lo = r["delta_ci95_low"] if "delta_ci95_low" in r.index else np.nan
                ci_hi = r["delta_ci95_high"] if "delta_ci95_high" in r.index else np.nan
                if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                    new_lo, new_hi = -float(ci_hi), -float(ci_lo)
                else:
                    new_lo, new_hi = float("nan"), float("nan")
                rows.append(
                    {
                        **{
                            k: r[k]
                            for k in pairwise.columns
                            if k
                            not in {
                                "method_a",
                                "method_b",
                                "mean_a",
                                "mean_b",
                                "delta_a_minus_b",
                                "delta_ci95_low",
                                "delta_ci95_high",
                                "rank_biserial_r",
                            }
                        },
                        "method_a": b,
                        "method_b": a,
                        "mean_a": r["mean_b"],
                        "mean_b": r["mean_a"],
                        "delta_a_minus_b": -r["delta_a_minus_b"],
                        "delta_ci95_low": new_lo,
                        "delta_ci95_high": new_hi,
                        "rank_biserial_r": (
                            -r["rank_biserial_r"]
                            if np.isfinite(r["rank_biserial_r"])
                            else r["rank_biserial_r"]
                        ),
                        "p_value": r["p_value"],
                        "p_holm": r["p_holm"],
                        "wilcoxon_stat": r.get("wilcoxon_stat"),
                        "n_pairs": r["n_pairs"],
                        "note": r["note"],
                    }
                )
            else:
                rows.append(r.to_dict())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Re-Holm within primary family only (tighter / more relevant for paper)
    parts = []
    for _, g in out.groupby(
        ["cohort", "mechanism", "metric", "missing_rate"], sort=False
    ):
        gg = g.copy()
        gg["p_holm_primary_family"] = holm_adjust(gg["p_value"].tolist())
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)
