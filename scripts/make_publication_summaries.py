"""Regenerate the compact publication summaries from the released final dataset.

Usage:
    python scripts/make_publication_summaries.py publication/data/final_v1.parquet [--out-dir publication/results]

Writes
* final_context_family_qeubo_t20.csv - qEUBO T=20 minus per-cell best static, overall and by context,
  family and context x family (listener-level bootstrap CIs; frozen settings);
* final_learned_curve_means.csv      - mean and SD of listener-level regret per learned method and budget;
* final_learned_curve_ci.csv         - the same means with listener-level 95% bootstrap CIs, plus the best
  static reference (used for the uncertainty bands in Fig. 2).
The first two files reproduce the versions used in the frozen analysis exactly.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("analyze_final", HERE / "analyze_final.py")
af = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(af)

PRIMARY = list(af.PRE_SPECIFIED_FAMILIES)


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return af.load_results(path)


def context_family_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context in ("ALL",) + af.CONTEXTS:
        for family in ("ALL",) + tuple(PRIMARY):
            sub = df if context == "ALL" else df[df.context == context]
            fams = PRIMARY if family == "ALL" else [family]
            d = af.paired_delta_vs_best_static(sub, "audiotwin_qeubo", 20, families=fams)
            lo, hi = af.bootstrap_mean_ci(d.to_numpy())
            rows.append(dict(context=context, family=family, n=len(d), mean_delta=d.mean(), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def curve_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df.family.isin(PRIMARY)]
    means, cis = [], []
    for method in af.LEARNED_METHODS:
        for budget in af.BUDGETS:
            s = af._listener_method(sub, method, budget)
            means.append(dict(method=method, budget=budget, n=len(s), mean_regret=s.mean(), sd_regret=s.std()))
            lo, hi = af.bootstrap_mean_ci(s.to_numpy())
            cis.append(dict(method=method, budget=budget, n=len(s), mean_regret=s.mean(), ci_low=lo, ci_high=hi))
    static = af._listener_best_static(sub)
    lo, hi = af.bootstrap_mean_ci(static.to_numpy())
    cis.append(dict(method="best_static", budget=0, n=len(static), mean_regret=static.mean(), ci_low=lo, ci_high=hi))
    return pd.DataFrame(means), pd.DataFrame(cis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="publication/data/final_v1.parquet")
    ap.add_argument("--out-dir", default="publication/results")
    args = ap.parse_args()
    df = load(args.data)
    af.validate_final_results(df, replacement_map=af.FINAL_REPLACEMENT_MAP)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context_family_table(df).to_csv(out / "final_context_family_qeubo_t20.csv", index=False)
    means, cis = curve_tables(df)
    means.to_csv(out / "final_learned_curve_means.csv", index=False)
    cis.to_csv(out / "final_learned_curve_ci.csv", index=False)
    print(f"wrote summaries to {out}")


if __name__ == "__main__":
    main()
