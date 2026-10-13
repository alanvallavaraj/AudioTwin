"""Development-only oracle/observation robustness checks for AudioTwin.

All scenarios use development seed 424242 and are governed by docs/protocol_v2_addendum.md
(sections v2 and v2.1). They do not replace the frozen final N=128 result.

Each listener-scenario runs qEUBO and GP-random with the same seed, so both acquisitions see the
same candidate-pool and oracle random-number streams (as in the frozen final runner). Listener-level
95% intervals use the frozen analysis bootstrap settings (20,000 resamples, seed 20261011).
"""
from __future__ import annotations
import argparse, dataclasses
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from experiments.runner import stable_seed
from oracle import CONTEXTS, find_optimum, observe_audiogram, sample_population, utility
from twin.active import run_laplace_twin
from twin.optimizer import feasible_anchor, project_from_anchor

DEV_SEED = 424242
CONTEXTS_FINAL = ('noisy_conversation', 'lecture', 'music')
FAMILIES = ('intelligibility', 'comfort', 'balanced')
SCENARIOS = ('base', 'no_ideal_term', 'double_ideal_offset', 'high_choice_noise', 'high_audiogram_error')
ANALYSIS_SEED = 20261011
N_BOOTSTRAP = 20_000


def bootstrap_mean_ci(values, n_resamples=N_BOOTSTRAP, seed=ANALYSIS_SEED):
    """Same listener-level percentile bootstrap as scripts/analyze_final.py."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(int(n_resamples), len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def modify_listener(L, scenario):
    if scenario == 'no_ideal_term':
        weights = {}
        for k, w in L.weights.items():
            x = np.array(w, float); x[2] = 0.0; x = x / x.sum(); weights[k] = x
        return dataclasses.replace(L, weights=weights)
    if scenario == 'double_ideal_offset':
        return dataclasses.replace(L, ideal_offset_db=2.0 * L.ideal_offset_db)
    if scenario == 'high_choice_noise':
        return dataclasses.replace(L, sigma_n=min(0.30, 2.0 * L.sigma_n), lapse=min(0.10, 2.0 * L.lapse))
    return L


def one(task_scenario):
    task_id, scenario = task_scenario
    base = DEV_SEED + 1009 * task_id
    rng = np.random.default_rng(base)
    L0 = sample_population(1, seed=base)[0]
    L = modify_listener(L0, scenario)
    sigma_a = 10.0 if scenario == 'high_audiogram_error' else 5.0
    a_hat = observe_audiogram(L, sigma_a, rng)
    try:
        anchor = feasible_anchor(a_hat)
    except ValueError:
        return {"task_id": task_id, "scenario": scenario, "feasible": False, "rows": []}
    z0 = project_from_anchor(np.zeros(6), anchor, a_hat)
    zlin = project_from_anchor(np.r_[0, 0, -1, -1, -1, 0], anchor, a_hat)
    rows = []
    for c in CONTEXTS_FINAL:
        ctx = CONTEXTS[c]
        for f in FAMILIES:
            opt = find_optimum(L, ctx, f, a_hat, seed=stable_seed(base, 'robust-opt', scenario, task_id, c, f),
                               n_random=400, maxiter=60, popsize=12)
            static = min(opt.regret(utility(z0, L, ctx, f, a_hat)), opt.regret(utility(zlin, L, ctx, f, a_hat)))
            seed = stable_seed(base, 'robust-gp', scenario, task_id, c, f)
            out = dict(task_id=task_id, scenario=scenario, context=c, family=f, static_regret=static)
            for acq, col in (('qeubo', 'qeubo_regret'), ('random', 'gprandom_regret')):
                run = run_laplace_twin(L, ctx, f, a_hat, n_queries=20, seed=seed, acquisition=acq, checkpoints=(20,))
                out[col] = opt.regret(utility(run.recs[20], L, ctx, f, a_hat))
            rows.append(out)
    return {"task_id": task_id, "scenario": scenario, "feasible": True, "rows": rows}


def summarise(df: pd.DataFrame, feasibility: pd.DataFrame) -> pd.DataFrame:
    pl = df.groupby(['scenario', 'task_id'])[['static_regret', 'qeubo_regret', 'gprandom_regret']].mean()
    pl['q_minus_static'] = pl.qeubo_regret - pl.static_regret
    pl['q_minus_random'] = pl.qeubo_regret - pl.gprandom_regret
    rows = []
    for scenario in SCENARIOS:
        if scenario not in pl.index.get_level_values(0):
            continue
        g = pl.loc[scenario]
        lo_s, hi_s = bootstrap_mean_ci(g.q_minus_static)
        lo_r, hi_r = bootstrap_mean_ci(g.q_minus_random)
        fz = feasibility[feasibility.scenario == scenario]
        rows.append(dict(
            scenario=scenario, feasible_listeners=int(fz.feasible.sum()), attempted_listeners=len(fz),
            mean_static_regret=g.static_regret.mean(), mean_qeubo_regret=g.qeubo_regret.mean(),
            mean_gprandom_regret=g.gprandom_regret.mean(),
            qeubo_minus_static=g.q_minus_static.mean(), qeubo_minus_static_ci_low=lo_s, qeubo_minus_static_ci_high=hi_s,
            listeners_qeubo_beats_static=int((g.q_minus_static < 0).sum()),
            qeubo_minus_gprandom=g.q_minus_random.mean(), qeubo_minus_gprandom_ci_low=lo_r,
            qeubo_minus_gprandom_ci_high=hi_r, listeners_qeubo_beats_gprandom=int((g.q_minus_random < 0).sum()),
        ))
    out = pd.DataFrame(rows)
    # Within-listener shift of the qEUBO-minus-static difference relative to the base scenario.
    shifts = []
    for scenario in SCENARIOS[1:]:
        if scenario not in pl.index.get_level_values(0):
            continue
        d = (pl.loc[scenario].q_minus_static - pl.loc['base'].q_minus_static).dropna()
        lo, hi = bootstrap_mean_ci(d)
        shifts.append(dict(scenario=scenario, shift_vs_base=d.mean(), shift_ci_low=lo, shift_ci_high=hi, n_paired=len(d)))
    return out.merge(pd.DataFrame(shifts), on='scenario', how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-dir', default='publication/postfinal_diagnostics')
    a = ap.parse_args()
    jobs = [(t, s) for s in SCENARIOS for t in range(a.tasks)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(one, jobs))
    df = pd.DataFrame([r for res in results for r in res['rows']])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'oracle_robustness_dev.csv', index=False)
    feasibility = pd.DataFrame([{k: res[k] for k in ('task_id', 'scenario', 'feasible')} for res in results])
    feasibility.to_csv(out / 'oracle_robustness_feasibility.csv', index=False)
    summary = summarise(df, feasibility)
    summary.to_csv(out / 'oracle_robustness_dev_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
