# Post-final development-only diagnostics

These files are **post-final exploratory diagnostics** governed by `docs/protocol_v2_addendum.md` (sections v2 and v2.1). New simulations use development seed `424242` only and never reuse the reserved final seed `777000111`. They cannot alter the frozen primary result; all intervals are descriptive.

## Coordinate search (`coordinate_search_dev32*`)

A model-free pairwise coordinate-search baseline on the 32-listener development sweep at T=20. Its mean regret relative to static was +0.0275 (11/32 listeners better than static), against −0.0566 for qEUBO (21/32).

## Acquisition geometry (`acquisition_geometry_dev32*`)

Boundary criterion `max(abs(z)) >= 0.9`, query endpoints over the first 30 comparisons: 22.9% for qEUBO and 87.6% for BALD.

## Music–intelligibility (`music_intelligibility_dev32*`)

All 32 development optima put noise reduction (NR) and speech enhancement (SE) at the off corner, as the oracle construction implies for music. qEUBO T=20 regret: 0.294 as frozen, 0.382 with the GP prior coefficient set to zero, and 0.069 without response noise.

## NR/SE resolution (`nrse_resolution_dev32*`)

Intelligibility family, three contexts, qEUBO T=20, 32 listeners:

| Context | NR/SE stake (median, noise SD units) | Best static | qEUBO | qEUBO, noise-free | Mean rec. z4 | Mean opt. z4 |
|---|---:|---:|---:|---:|---:|---:|
| Noisy conversation | 0.46 | 0.379 | 0.347 | 0.082 | +0.06 | +0.53 |
| Lecture | 0.10 | 0.329 | 0.249 | 0.042 | −0.17 | −0.19 |
| Music | 0.18 | 0.256 | 0.294 | 0.069 | −0.05 | −1.00 |

Removing response noise helps every context, not only music. Adding the NR/SE-off corner to every candidate pool left all 32 music recommendations unchanged. The music rows reproduce `music_intelligibility_dev32.csv` exactly.

## Oracle robustness (`oracle_robustness_dev*`)

32 development listeners, qEUBO and GP-random at T=20 with shared seeds, listener-level bootstrap 95% CIs (20,000 resamples, seed 20261011):

| Scenario | Feasible | qEUBO − static | Better | qEUBO − GP-random | Better |
|---|---:|---:|---:|---:|---:|
| Base | 32/32 | −0.056 [−0.120, −0.001] | 19 | −0.017 [−0.040, +0.006] | 23 |
| Ideal-gain offsets doubled | 32/32 | −0.113 [−0.172, −0.055] | 23 | −0.026 [−0.053, +0.001] | 20 |
| Audiogram error doubled | 26/32 | −0.070 [−0.148, −0.005] | 16 | −0.025 [−0.059, +0.008] | 15 |
| Ideal-gain term removed | 32/32 | −0.181 [−0.244, −0.126] | 29 | −0.053 [−0.079, −0.027] | 25 |
| Choice noise and lapses doubled | 32/32 | +0.039 [−0.016, +0.088] | 9 | +0.014 [−0.014, +0.043] | 15 |

Within-listener shift of qEUBO − static from base to doubled noise: +0.095 [+0.071, +0.121]. Infeasible draws are recorded in `oracle_robustness_feasibility.csv` and are not repaired by relaxing constraints. The earlier 12-listener qEUBO and static results are reproduced exactly as a subset.
