"""Sanity diagnostics for the oracle: is there headroom, are optima interior, do families differ?"""
import numpy as np
from dsp import Z_DIM
from oracle import CONTEXTS, FAMILIES, find_optimum, sample_population, utility

pop = sample_population(25, seed=123)
rows = []
for ctx_name in ["quiet", "noisy_conversation", "music"]:
    ctx = CONTEXTS[ctx_name]
    for fam in FAMILIES:
        reg0, bound, spread = [], [], []
        for L in pop:
            o = find_optimum(L, ctx, fam, L.audiogram, seed=L.id)
            u0 = utility(np.zeros(Z_DIM), L, ctx, fam, L.audiogram)
            reg0.append(o.regret(u0))
            bound.append(np.mean(np.abs(o.z_opt) > 0.98))
        rows.append((ctx_name, fam, np.mean(reg0), np.median(reg0), np.mean(bound)))
print(f"{'context':20s} {'family':16s} {'regret(z=0) mean':>17s} {'median':>8s} {'frac z at bound':>16s}")
for r in rows:
    print(f"{r[0]:20s} {r[1]:16s} {r[2]:17.3f} {r[3]:8.3f} {r[4]:16.3f}")
