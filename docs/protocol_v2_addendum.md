# Protocol v2 addendum — post-final development-only diagnostics

Date: 2026-10-12

This addendum is written **after** the frozen final N=128 evaluation and **before** any post-final new diagnostic or robustness run described below. It does not alter, rerun, or replace the frozen final cohort or its primary estimand.

## Provenance clarification

The final-evaluation decisions were recorded before inspection of final performance outcomes in a private development repository. The public repository was curated and created after the final run, so its commit/tag timestamps do **not** independently prove that chronology. The manuscript must therefore describe the final analysis as **pre-specified**, not pre-registered, and must not claim that the public Git history itself verifies the freeze.

## Post-final development-only analyses

All new runs below use development seed `424242` (or deterministic sub-seeds derived from it). The reserved final seed `777000111` is not to be used again.

1. **Coordinate-search comparator.** Add a simple interactive coordinate-search baseline with the same pairwise-query budgets. This is a secondary development-only comparator and cannot change the frozen primary result.
2. **Acquisition geometry diagnostic.** Quantify the fraction of qEUBO and BALD queried points near the boundary of the latent cube. Primary diagnostic threshold: `max(abs(z)) >= 0.9`; report the threshold explicitly.
3. **Music–intelligibility diagnostic.** Quantify whether the oracle optimum lies near a corner, with particular attention to `z4` (noise reduction) and `z5` (speech enhancement), and compare against the GP prior mean penalty. Do not tune the prior from this diagnostic in the current paper.
4. **Oracle robustness.** Evaluate the frozen algorithms on development listeners under perturbed oracle assumptions (preference-weight concentration, audiogram error, response noise/lapse, and ideal-gain offsets). These analyses are robustness checks only; they do not replace the frozen final evaluation.

## Locked interpretation

Post-final diagnostics may qualify or weaken conclusions. They must not be used to redesign the algorithm and then rerun the final seed. Any redesigned method would require a genuinely new held-out evaluation cohort in future work.

## Addendum v2.1 — additional exploratory analyses (12 October 2026)

This section was written **after** the v2 diagnostics above had been run and inspected. It is therefore not a pre-specification of anything; it records what was added and why, so that every number in the manuscript can be traced to a script.

1. **Robustness extension.** `scripts/oracle_robustness_dev.py` was extended from 12 to 32 development listeners and given a GP-random arm that shares each listener's seeds with qEUBO. Listener-level 95% intervals use the frozen bootstrap settings (20,000 resamples, seed 20261011). The 12-listener qEUBO results are reproduced exactly as a subset.
2. **NR/SE resolution diagnostic.** `scripts/nrse_resolution_dev.py` (32 development listeners, intelligibility family, three contexts) applies the noise-free control to every intelligibility cell, not only music. It measures how much hidden utility the noise-reduction and speech-enhancement controls carry relative to choice noise, and it tests adding the NR/SE-off corner to the candidate pool.
3. **Descriptive re-analysis of the released final rows.** `scripts/analyze_exploratory_final.py` reports win counts, the full matched acquisition table at every budget, regret-scale checks and NR/SE recommendation geometry from `publication/data/final_v1.parquet`. No simulation is run and the final seed is not used; only the qEUBO-versus-GP-random contrasts at T = 5 and 10 were pre-specified.

The locked interpretation above applies unchanged: these analyses may qualify conclusions but cannot redesign the method or alter the frozen primary estimand.

_Typographical note: on 12 October 2026 the phrase "not pre-registered" in the provenance clarification was restored after an automated find-and-replace had changed it to "not pre-specified"._

