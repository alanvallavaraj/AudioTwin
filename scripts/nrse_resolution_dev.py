"""Development-only diagnostic: can the learner resolve the noise-reduction / speech-enhancement controls?

Governed by docs/protocol_v2_addendum.md (section v2.1). Development seed 424242 only; the reserved
final seed is not used. Intelligibility family, three final contexts, qEUBO at T=20.

For each listener and context it reports:
* NR/SE stake: [u(z_opt) - u(z_opt with z4, z5 set to 0)] / sigma_n, i.e. how much hidden utility the two
  enhancement controls carry relative to the listener's choice noise;
* qEUBO regret with the original listener and with response noise removed (sigma_n=1e-6, no lapses),
  using identical seeds - the control for scripts/music_intelligibility_diagnostic.py, applied to every
  intelligibility cell rather than music alone;
* where the recommendation and the optimum place z4 and z5;
* the per-cell best static regret (default or linear prescription);
* in music only, a candidate-pool ablation adding the projected NR/SE-off "linear" corner
  (0, 0, -1, -1, -1, 0) to every query and recommendation pool.
"""
from __future__ import annotations
import argparse, dataclasses, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from experiments.runner import stable_seed
from oracle import CONTEXTS, find_optimum, observe_audiogram, sample_population, utility
from twin.active import run_laplace_twin
from twin.optimizer import feasible_anchor, project_from_anchor

DEV_SEED = 424242
CONTEXTS_FINAL = ('noisy_conversation', 'lecture', 'music')
FAMILY = 'intelligibility'
LINEAR_CORNER = np.r_[0.0, 0.0, -1.0, -1.0, -1.0, 0.0]


def one(task_id: int):
    base = DEV_SEED + 1009 * task_id
    rng = np.random.default_rng(base)
    L = sample_population(1, seed=base)[0]
    a_hat = observe_audiogram(L, 5.0, rng)
    anchor = feasible_anchor(a_hat)
    statics = [project_from_anchor(z, anchor, a_hat) for z in (np.zeros(6), LINEAR_CORNER)]
    noiseless = dataclasses.replace(L, sigma_n=1e-6, lapse=0.0)
    rows = []
    for c in CONTEXTS_FINAL:
        ctx = CONTEXTS[c]
        opt = find_optimum(L, ctx, FAMILY, a_hat, seed=stable_seed(base, 'opt', task_id, c, FAMILY),
                           n_random=400, maxiter=60, popsize=12)
        z_no_nrse = opt.z_opt.copy(); z_no_nrse[3] = 0.0; z_no_nrse[4] = 0.0
        stake = (utility(opt.z_opt, L, ctx, FAMILY, a_hat) - utility(z_no_nrse, L, ctx, FAMILY, a_hat)) / L.sigma_n
        seed = stable_seed(base, 'gp', task_id, c, FAMILY)
        orig = run_laplace_twin(L, ctx, FAMILY, a_hat, n_queries=20, seed=seed, acquisition='qeubo', checkpoints=(20,))
        nl = run_laplace_twin(noiseless, ctx, FAMILY, a_hat, n_queries=20, seed=seed, acquisition='qeubo', checkpoints=(20,))
        reg = lambda z: opt.regret(utility(z, L, ctx, FAMILY, a_hat))
        row = dict(task_id=task_id, context=c, sigma_n=L.sigma_n, nrse_stake_over_sigma=stake,
                   best_static_regret=min(reg(z) for z in statics),
                   linear_is_best_static=bool(reg(statics[1]) < reg(statics[0])),
                   qeubo_regret=reg(orig.recs[20]), noiseless_regret=reg(nl.recs[20]),
                   rec_z4=float(orig.recs[20][3]), rec_z5=float(orig.recs[20][4]),
                   noiseless_rec_z4=float(nl.recs[20][3]), noiseless_rec_z5=float(nl.recs[20][4]),
                   opt_z4=float(opt.z_opt[3]), opt_z5=float(opt.z_opt[4]), corner_pool_regret=np.nan)
        if c == 'music':
            cp = run_laplace_twin(L, ctx, FAMILY, a_hat, n_queries=20, seed=seed, acquisition='qeubo',
                                  checkpoints=(20,), extra_base_points=LINEAR_CORNER)
            row['corner_pool_regret'] = reg(cp.recs[20])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-dir', default='publication/postfinal_diagnostics')
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        df = pd.DataFrame([r for rows in ex.map(one, range(a.tasks)) for r in rows])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'nrse_resolution_dev{a.tasks}.csv', index=False)
    by_ctx = {}
    for c, g in df.groupby('context'):
        by_ctx[c] = dict(
            median_nrse_stake_over_sigma=float(g.nrse_stake_over_sigma.median()),
            mean_best_static_regret=float(g.best_static_regret.mean()),
            mean_qeubo_regret=float(g.qeubo_regret.mean()),
            mean_noiseless_regret=float(g.noiseless_regret.mean()),
            noiseless_improvement=float((g.qeubo_regret - g.noiseless_regret).mean()),
            noiseless_better_fraction=float((g.noiseless_regret < g.qeubo_regret).mean()),
            mean_rec_z4=float(g.rec_z4.mean()), mean_rec_z5=float(g.rec_z5.mean()),
            mean_opt_z4=float(g.opt_z4.mean()), mean_opt_z5=float(g.opt_z5.mean()),
            linear_best_static_fraction=float(g.linear_is_best_static.mean()),
        )
    m = df[df.context == 'music']
    summary = dict(development_seed=DEV_SEED, n_listeners=int(df.task_id.nunique()), family=FAMILY, budget=20,
                   by_context=by_ctx,
                   music_corner_pool=dict(mean_qeubo_regret=float(m.qeubo_regret.mean()),
                                          mean_corner_pool_regret=float(m.corner_pool_regret.mean()),
                                          corner_pool_better_fraction=float((m.corner_pool_regret < m.qeubo_regret - 1e-12).mean())))
    (out / f'nrse_resolution_dev{a.tasks}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
