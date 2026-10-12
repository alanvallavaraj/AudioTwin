"""Statistics helpers: paired Wilcoxon, matched-pairs rank-biserial effect size,
percentile-bootstrap CIs, Holm correction, and a method-vs-baseline comparison table.

Unit of analysis = listener (profile). Average over seeds/utterances within a listener
BEFORE testing, so the n is the number of independent listeners, not of repeated runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def bootstrap_ci(x, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def rank_biserial_paired(diff) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1] (sign = direction of a - b)."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def paired_wilcoxon(a, b) -> tuple[float, float]:
    """(statistic, two-sided p). Returns p=1.0 if all differences are zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(d == 0):
        return 0.0, 1.0
    res = sps.wilcoxon(d, alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


def compare_to_baseline(df: pd.DataFrame, metric: str, method_col: str = "method",
                        unit_col: str = "listener", baseline: str = "prescription",
                        higher_is_better: bool = True, seed: int = 0) -> pd.DataFrame:
    """One row per non-baseline method: mean paired difference (method - baseline), bootstrap
    CI, Wilcoxon p, Holm-adjusted p, rank-biserial effect size. Averages within unit first."""
    wide = df.groupby([unit_col, method_col])[metric].mean().unstack(method_col)
    if baseline not in wide.columns:
        raise KeyError(f"baseline '{baseline}' not in methods {list(wide.columns)}")
    rows = []
    for m in wide.columns:
        if m == baseline:
            continue
        pair = wide[[m, baseline]].dropna()
        d = pair[m].to_numpy() - pair[baseline].to_numpy()
        if not higher_is_better:
            d = -d
        lo, hi = bootstrap_ci(d, seed=seed)
        _, p = paired_wilcoxon(d, np.zeros_like(d))
        rows.append({"method": m, "n_units": len(d), "mean_diff": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi, "p": p, "effect_rb": rank_biserial_paired(d)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    return out
