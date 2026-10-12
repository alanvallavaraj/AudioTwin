"""Frozen listener-level analysis for AudioTwin final protocol v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STATIC_METHODS = ("prescription", "prescription_linear", "prescription_slow")
PRE_REGISTERED_FAMILIES = ("intelligibility", "comfort", "balanced")
LEARNED_METHODS = ("audiotwin_qeubo", "audiotwin_bald", "audiotwin_gprandom")
BUDGETS = (0, 5, 10, 20, 30, 50)
CONTEXTS = ("noisy_conversation", "lecture", "music")
ALL_FAMILIES = PRE_REGISTERED_FAMILIES + ("noise_sensitive",)
ANALYSIS_SEED = 20261011
N_BOOTSTRAP = 20_000


def _listener_best_static(df: pd.DataFrame) -> pd.Series:
    per_cell = (
        df[df.method.isin(STATIC_METHODS)]
        .groupby(["listener_id", "context", "family"], as_index=False)
        .regret.min()
    )
    return per_cell.groupby("listener_id").regret.mean().sort_index()


def _listener_method(df: pd.DataFrame, method: str, budget: int) -> pd.Series:
    return (
        df[(df.method == method) & (df.n_queries == budget)]
        .groupby("listener_id").regret.mean().sort_index()
    )


def paired_delta_vs_best_static(
    df: pd.DataFrame,
    method: str,
    budget: int,
    families=PRE_REGISTERED_FAMILIES,
) -> pd.Series:
    """Listener-level method minus per-cell-best-static regret (negative is better)."""
    sub = df[df.family.isin(tuple(families))]
    learned = _listener_method(sub, method, budget).rename("learned")
    baseline = _listener_best_static(sub).rename("baseline")
    paired = pd.concat([learned, baseline], axis=1).dropna()
    return (paired.learned - paired.baseline).rename("delta")


def paired_method_delta(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    budget: int,
    families=PRE_REGISTERED_FAMILIES,
) -> pd.Series:
    """Listener-level method_a minus method_b regret (negative favours method_a)."""
    sub = df[df.family.isin(tuple(families))]
    a = _listener_method(sub, method_a, budget).rename("a")
    b = _listener_method(sub, method_b, budget).rename("b")
    paired = pd.concat([a, b], axis=1).dropna()
    return (paired.a - paired.b).rename("delta")


def bootstrap_mean_ci(values, n_resamples=N_BOOTSTRAP, seed=ANALYSIS_SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(int(n_resamples), len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def load_results(root: str | Path) -> pd.DataFrame:
    files = sorted(Path(root).glob("task_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no task_*.parquet files in {root}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def validate_final_results(df: pd.DataFrame, expected_n: int = 128) -> None:
    """Refuse inference unless the complete frozen final result grid is present."""
    task_ids = set(map(int, df.task_id.unique()))
    expected_tasks = set(range(int(expected_n)))
    if task_ids != expected_tasks:
        missing = sorted(expected_tasks - task_ids)
        extra = sorted(task_ids - expected_tasks)
        raise ValueError(f"final task IDs incomplete or unexpected; missing={missing} extra={extra}")

    listener_ids = set(map(int, df.listener_id.unique()))
    if listener_ids != expected_tasks:
        raise ValueError("final listener IDs do not match frozen task IDs")

    expected = set()
    for task_id in range(int(expected_n)):
        for context in CONTEXTS:
            for family in ALL_FAMILIES:
                for method in STATIC_METHODS:
                    expected.add((task_id, task_id, context, family, method, 0))
                for method in LEARNED_METHODS:
                    for budget in BUDGETS:
                        expected.add((task_id, task_id, context, family, method, budget))

    actual_rows = list(
        df[["task_id", "listener_id", "context", "family", "method", "n_queries"]]
        .itertuples(index=False, name=None)
    )
    actual = set((int(t), int(l), c, f, m, int(q)) for t, l, c, f, m, q in actual_rows)
    if actual != expected or len(actual_rows) != len(expected):
        raise ValueError(
            "final results do not match the frozen result grid "
            f"(expected {len(expected)} rows, found {len(actual_rows)} rows and {len(actual)} unique keys)"
        )

    if df.regret.isna().any():
        raise ValueError("final results contain missing regret values")


def _fmt_delta(delta: pd.Series) -> str:
    lo, hi = bootstrap_mean_ci(delta.to_numpy())
    return f"mean={delta.mean():+.4f}  95% bootstrap CI=[{lo:+.4f}, {hi:+.4f}]  n={len(delta)}"


def main(root: str | Path) -> None:
    df = load_results(root)
    validate_final_results(df)
    print(f"files/listeners: {df.task_id.nunique()}/{df.listener_id.nunique()}  rows={len(df)}")
    print(f"code: {', '.join(sorted(df.git_hash.str[:7].unique()))}")

    print("\nPRIMARY: qEUBO T=20 minus per-cell best static; pre-registered families")
    print(_fmt_delta(paired_delta_vs_best_static(df, "audiotwin_qeubo", 20)))

    print("\nSECONDARY: learned methods minus per-cell best static; pre-registered families")
    for method in LEARNED_METHODS:
        for budget in BUDGETS:
            d = paired_delta_vs_best_static(df, method, budget)
            print(f"{method:20s} T={budget:2d}  {_fmt_delta(d)}")

    print("\nACQUISITION EFFICIENCY: qEUBO minus GP-random; pre-registered families")
    for budget in (5, 10):
        d = paired_method_delta(df, "audiotwin_qeubo", "audiotwin_gprandom", budget)
        print(f"T={budget:2d}  {_fmt_delta(d)}")

    print("\nPOST-HOC noise_sensitive: qEUBO minus per-cell best static")
    for budget in BUDGETS:
        d = paired_delta_vs_best_static(df, "audiotwin_qeubo", budget, families=("noise_sensitive",))
        print(f"T={budget:2d}  {_fmt_delta(d)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/final_v1")
