"""Exploratory (not pre-specified) descriptive analyses of the released final dataset.

Usage:
    python scripts/analyze_exploratory_final.py publication/data/final_v1.parquet > publication/results/final_v1_exploratory.txt

Governed by docs/protocol_v2_addendum.md (section v2.1). Runs no simulation and does not touch the
reserved final seed; it only re-reads the frozen final rows. Of the contrasts below, only qEUBO minus
GP-random at T = 5 and T = 10 were pre-specified secondary analyses; everything else is exploratory.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("analyze_final", HERE / "analyze_final.py")
af = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(af)

PRIMARY = list(af.PRE_SPECIFIED_FAMILIES)
Z = [f"z{i}" for i in range(1, 7)]


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else af.load_results(path)


def win_counts(df: pd.DataFrame) -> list[str]:
    out = ["LISTENER-LEVEL WIN COUNTS: qEUBO vs per-cell best static (pre-specified families)"]
    for b in af.BUDGETS:
        d = af.paired_delta_vs_best_static(df, "audiotwin_qeubo", b)
        out.append(f"T={b:2d}  qEUBO better for {(d < 0).sum():3d}/{len(d)}  median={d.median():+.4f}  "
                   f"IQR=[{d.quantile(.25):+.4f}, {d.quantile(.75):+.4f}]")
    return out


def cell_win_rates(df: pd.DataFrame) -> list[str]:
    sub = df[df.family.isin(PRIMARY)]
    stat = sub[sub.method.isin(af.STATIC_METHODS)]
    best = stat.loc[stat.groupby(["listener_id", "context", "family"]).utility.idxmax(),
                    ["listener_id", "context", "family", "utility", "method"]]
    best = best.rename(columns={"utility": "u_static", "method": "static_winner"})
    q = sub[(sub.method == "audiotwin_qeubo") & (sub.n_queries == 20)][["listener_id", "context", "family", "utility"]]
    m = q.merge(best, on=["listener_id", "context", "family"])
    m["win"] = m.utility > m.u_static + 1e-12
    out = [f"CELL-LEVEL: qEUBO T=20 utility above best static in {m.win.mean():.3f} of {len(m)} listener-cells"]
    out.append("cell win rate by context x family:")
    out += m.pivot_table(index="context", columns="family", values="win", aggfunc="mean").round(3).to_string().splitlines()
    # Which static configuration is the per-cell best (prescription_slow ties prescription in primary families).
    distinct = stat[stat.method.isin(["prescription", "prescription_linear"])]
    win = distinct.loc[distinct.groupby(["listener_id", "context", "family"]).utility.idxmax()]
    share = win.assign(linear=win.method == "prescription_linear").pivot_table(
        index="context", columns="family", values="linear", aggfunc="mean").round(3)
    out.append("share of cells where the linear (NR/SE-off) static setting is the better static choice:")
    out += share.to_string().splitlines()
    return out


def matched_acquisition(df: pd.DataFrame) -> list[str]:
    out = ["MATCHED ACQUISITION DIFFERENCES (listener-level bootstrap 95% CI; pre-specified families)",
           "(pre-specified: qEUBO - GP-random at T=5 and T=10; all other rows exploratory)"]
    for other in ("audiotwin_gprandom", "audiotwin_bald"):
        for b in af.BUDGETS[1:]:
            d = af.paired_method_delta(df, "audiotwin_qeubo", other, b)
            lo, hi = af.bootstrap_mean_ci(d.to_numpy())
            out.append(f"qEUBO - {other:18s} T={b:2d}  mean={d.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
                       f"qEUBO better for {(d < 0).sum()}/{len(d)}")
    return out


def regret_scale(df: pd.DataFrame) -> list[str]:
    ql = df[df.method == "audiotwin_qeubo"]
    return ["REGRET SCALE",
            f"rows with regret == 0 (setting at or above the numerical optimum): {int((df.regret == 0).sum())}",
            f"rows with regret > 1 (worse than the 5th-percentile reference): {int((df.regret > 1).sum())}; "
            f"max regret {df.regret.max():.3f}",
            f"qEUBO rows with utility above the numerical optimum: {int((ql.utility > ql.u_opt + 1e-12).sum())}"]


def bald_changes(df: pd.DataFrame) -> list[str]:
    key = ["listener_id", "context", "family"]
    b20 = df[(df.method == "audiotwin_bald") & (df.n_queries == 20) & df.family.isin(PRIMARY)].set_index(key)[Z]
    b30 = df[(df.method == "audiotwin_bald") & (df.n_queries == 30) & df.family.isin(PRIMARY)].set_index(key)[Z]
    same = (np.abs(b20.values - b30.loc[b20.index].values).max(axis=1) < 1e-9)
    return ["BALD T=20 vs T=30",
            f"recommendation unchanged between T=20 and T=30 in {same.mean():.3f} of {len(same)} primary cells"]


def nrse_geometry(df: pd.DataFrame) -> list[str]:
    out = ["NR/SE GEOMETRY: qEUBO recommendations vs oracle optimum (primary families)"]
    q = df[(df.method == "audiotwin_qeubo") & df.family.isin(PRIMARY)].copy()
    q["rec_corner"] = (q.z4 <= -0.8) & (q.z5 <= -0.8)
    q["opt_corner"] = (q.zopt4 <= -0.9) & (q.zopt5 <= -0.9)
    t20 = q[q.n_queries == 20].groupby(["context", "family"])[["z4", "z5", "zopt4", "zopt5", "rec_corner", "opt_corner"]].mean()
    out += t20.round(3).to_string().splitlines()
    mi = q[(q.context == "music") & (q.family == "intelligibility")]
    for b in af.BUDGETS[1:]:
        s = mi[mi.n_queries == b]
        out.append(f"music-intelligibility T={b:2d}: mean rec z4={s.z4.mean():+.3f} z5={s.z5.mean():+.3f}; "
                   f"recommendations at NR/SE-off corner: {s.rec_corner.mean():.3f}")
    return out


def main(path: str) -> None:
    df = load(path)
    af.validate_final_results(df, replacement_map=af.FINAL_REPLACEMENT_MAP)
    blocks = [
        [f"EXPLORATORY ANALYSIS OF RELEASED FINAL DATA  rows={len(df)}  listeners={df.listener_id.nunique()}  "
         f"code={','.join(sorted(df.git_hash.str[:7].unique()))}"],
        win_counts(df), cell_win_rates(df), matched_acquisition(df), regret_scale(df), bald_changes(df), nrse_geometry(df),
    ]
    print("\n\n".join("\n".join(b) for b in blocks))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "publication/data/final_v1.parquet")
