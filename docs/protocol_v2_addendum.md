# Protocol v2 addendum — post-final development-only diagnostics

Date: 2026-10-12

This addendum is written **after** the frozen final N=128 evaluation and **before** any post-final new diagnostic or robustness run described below. It does not alter, rerun, or replace the frozen final cohort or its primary estimand.

## Provenance clarification

The final-evaluation decisions were recorded before inspection of final performance outcomes in a private development repository. The public repository was curated and created after the final run, so its commit/tag timestamps do **not** independently prove that chronology. The manuscript must therefore describe the final analysis as **pre-specified**, not pre-specified, and must not claim that the public Git history itself verifies the freeze.

## Post-final development-only analyses

All new runs below use development seed `424242` (or deterministic sub-seeds derived from it). The reserved final seed `777000111` is not to be used again.

1. **Coordinate-search comparator.** Add a simple interactive coordinate-search baseline with the same pairwise-query budgets. This is a secondary development-only comparator and cannot change the frozen primary result.
2. **Acquisition geometry diagnostic.** Quantify the fraction of qEUBO and BALD queried points near the boundary of the latent cube. Primary diagnostic threshold: `max(abs(z)) >= 0.9`; report the threshold explicitly.
3. **Music–intelligibility diagnostic.** Quantify whether the oracle optimum lies near a corner, with particular attention to `z4` (noise reduction) and `z5` (speech enhancement), and compare against the GP prior mean penalty. Do not tune the prior from this diagnostic in the current paper.
4. **Oracle robustness.** Evaluate the frozen algorithms on development listeners under perturbed oracle assumptions (preference-weight concentration, audiogram error, response noise/lapse, and ideal-gain offsets). These analyses are robustness checks only; they do not replace the frozen final evaluation.

## Locked interpretation

Post-final diagnostics may qualify or weaken conclusions. They must not be used to redesign the algorithm and then rerun the final seed. Any redesigned method would require a genuinely new held-out evaluation cohort in future work.
