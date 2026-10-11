"""Dev-seed sanity check: does the Laplace-GP twin learn, and how do qEUBO/BALD/random compare?
Uses DEV seed 424242 (never the reserved final seed)."""
import sys, time
import numpy as np
import pandas as pd
from dsp import DSPConfig, Z_DIM
from oracle import CONTEXTS, find_optimum, observe_audiogram, sample_population, utility
from twin.active import ACQUISITIONS, run_laplace_twin
from twin.optimizer import feasible_anchor, project_from_anchor

DEV_SEED = 424242
N_LIST = int(sys.argv[1]) if len(sys.argv) > 1 else 6
T = int(sys.argv[2]) if len(sys.argv) > 2 else 30
CK = tuple(c for c in (0, 5, 10, 20, 30, 50) if c <= T)
CFG = DSPConfig()
pop = sample_population(N_LIST, seed=DEV_SEED)
rng = np.random.default_rng(DEV_SEED)
rows, t0 = [], time.time()
for L in pop:
    a_hat = observe_audiogram(L, 5.0, rng)
    anchor = feasible_anchor(a_hat, CFG)
    for ctx_name in ["noisy_conversation"]:
        ctx = CONTEXTS[ctx_name]
        for fam in ["balanced", "intelligibility"]:
            opt = find_optimum(L, ctx, fam, a_hat, seed=L.id)
            statics = {"prescription": project_from_anchor(np.zeros(Z_DIM), anchor, a_hat, CFG),
                       "prescription_linear": project_from_anchor(np.r_[0, 0, -1, -1, -1, 0.0], anchor, a_hat, CFG),
                       "prescription_slow": project_from_anchor(np.r_[0, 0, 0, 0, 0, 1.0], anchor, a_hat, CFG)}
            for name, z in statics.items():
                rows.append(dict(listener=L.id, ctx=ctx_name, fam=fam, method=name, T=0, regret=opt.regret(utility(z, L, ctx, fam, a_hat))))
            for acq in ACQUISITIONS:
                run = run_laplace_twin(L, ctx, fam, a_hat, n_queries=T, seed=1000 * L.id + 7, acquisition=acq, checkpoints=CK)
                for k, z in run.recs.items():
                    rows.append(dict(listener=L.id, ctx=ctx_name, fam=fam, method=acq, T=k, regret=opt.regret(utility(z, L, ctx, fam, a_hat))))
    print(f"listener {L.id} done ({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv("/tmp/diag_active.csv", index=False)
per = df.groupby(["listener", "method", "T"]).regret.mean().reset_index()
print("\nmean normalised regret (lower=better), n_listeners =", N_LIST)
print(per.pivot_table(index="method", columns="T", values="regret", aggfunc="mean").round(3).to_string())
