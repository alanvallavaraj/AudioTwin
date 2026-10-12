# Evaluation protocol (private development record)

## Public provenance note

This text originated in the development repository. The original lineage is now public on the `protocol-history` branch with annotated tags `final-protocol-v1`, `v1.1`, `v1.2` and `v1.3`. The curated `main` branch was assembled later and must not be used to infer freeze timing. `final-protocol-v1.2` resolves to commit `fc040b3`, the code hash stored in every released final data row. The later v1.3 tag records only the deterministic infeasible-listener replacement rule after task 55 failed and before performance analysis. These records are version-control provenance rather than external protocol registration, so manuscript language uses **pre-specified**.


Purpose: keep method development and final evaluation separate so reported numbers are not tuned on.

- **Development seeds** (may be used freely for debugging and method design):
  `20261006` (all earlier pilots; produced under the leaked oracle, see findings.md) and `424242` (fresh dev set).
- **Final seed: `777000111` — RESERVED.** Do not run any experiment on it until (1) the method, oracle,
  baselines, hyperparameters and budgets are frozen and tagged in git (`git tag final-protocol-v1`), and
  (2) the decision is recorded here. The final run is executed once. If a bug forces a re-run, record why.
- Number of AudioTwin variants developed against dev seeds is itself reportable. Variants tried so far:
  random-pair GP, prior-seeded, local-prior, active (mean+uncertainty), exploit, and (this protocol)
  Laplace-GP with qEUBO / BALD / random pairs. Only the Laplace-GP family is a candidate headline method;
  the hand-seeded prior variants encode the HASPI/linear-compression finding and are post-hoc ablations only.
- Pre-registered oracle families: intelligibility, comfort, balanced. Post-hoc: noise_sensitive (reported separately).
- Static configurations recorded: prescription, prescription_linear, prescription_slow (per analysis cell). In the three primary families, prescription_slow is utility-identical to prescription because compression speed is absent from the primary oracle; thus there are two distinct primary static utility behaviours. The slow variant remains distinct in the post-hoc noise_sensitive family. prescription_linear also sets noise reduction and speech enhancement to zero.
- Primary outcome: normalised regret at each query budget T in {0, 5, 10, 20, 30, 50}; unit of analysis = listener.

## Frozen final-evaluation decision (recorded 2026-10-11, before exposing the final seed)

Development evidence used for this decision is restricted to seed `424242`. On the 32-listener development sweep, qEUBO showed its clearest acquisition advantage over the matched GP-random policy at low budgets (5 and 10 queries), and its paired mean regret was below the per-cell best static baseline from 20 queries onward on the three pre-specified families. No result from the reserved final seed has been inspected.

The final protocol is therefore frozen as follows:

- **Primary learned method:** Laplace preference GP + qEUBO. BALD and GP-random remain fixed secondary acquisition comparators; no further learned variant will be developed before final evaluation.
- **Primary population/outcome:** all three contexts crossed with the three pre-specified families (`intelligibility`, `comfort`, `balanced`). Average the nine cells within each listener first. The primary contrast is qEUBO at **20 queries** versus the per-cell minimum over the three recorded static rows (two distinct utility behaviours in the primary families), expressed as paired normalised-regret difference (qEUBO minus static; negative is better).
- **Primary uncertainty:** listener-level non-parametric bootstrap 95% confidence interval for the paired mean difference. Resample listeners, not rows. Use 20,000 bootstrap resamples with analysis RNG seed `20261011`.
- **Secondary analyses:** the complete regret curve at T in {0, 5, 10, 20, 30, 50}; qEUBO versus GP-random at 5 and 10 queries (acquisition efficiency); qEUBO versus static at 50 queries; BALD as a secondary acquisition comparator; family/context breakdowns.
- **Post-hoc family:** `noise_sensitive` is analysed and labelled separately and cannot support the primary claim.
- **Final sample size:** **128 independent listeners** (array tasks 0--127). The development qEUBO-vs-static paired effect at T=20 was about -0.0566 with SD about 0.158, implying about 82 listeners for 90% normal-approximation power at two-sided alpha 0.05. N=128 is fixed before final-seed exposure to provide margin for sampling variability and subgroup summaries.
- **Final seed:** `777000111`, run once with exactly the frozen oracle, safety projection, baselines, qEUBO/BALD/GP-random implementations, candidate settings and query budgets in tag `final-protocol-v1.1`.
- If a software/numerical failure prevents completion, preserve the failed outputs/logs and document the failure before any re-run. Scientific tuning after seeing final-seed outcomes is prohibited.

- **Operational launcher correction (2026-10-12, before final-seed execution):** tag `final-protocol-v1.1` differs from `final-protocol-v1` only by initialising QMUL environment modules via `/etc/profile` and using the installed module name `python/3.11.7-gcc-12.2.0`. Scientific settings, code, seed, methods, budgets, oracle, sample size and analysis are unchanged.
- **Operational launcher correction v1.2 (2026-10-12, before any final result was produced):** the failed v1.1 array exited before Python started because QMUL environment-module initialisation is incompatible with Bash `nounset`. Tag `final-protocol-v1.2` changes the launcher from `set -euo pipefail` to `set -eo pipefail` only. No scientific setting, seed, method, budget, oracle, sample size, analysis or Python experiment code changed. The failed v1.1 array produced zero final result files.

- **Deterministic infeasible-listener replacement rule (2026-10-12, before inspecting final performance outcomes):** if a reserved final listener cannot produce any feasible safe anchor under the frozen safety constraints, do not relax those constraints and do not rerun the same deterministic draw. Replace it with the next previously unused task ID starting at 128, in ascending order, and retain N=128 analysable listeners. The first such case is task 55, replaced by task 128. Report the infeasible-draw count and replacement rule transparently.
