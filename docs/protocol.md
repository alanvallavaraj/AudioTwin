# Evaluation protocol (written 2026-10-07, BEFORE any results on the final seed)

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
- Static baselines to beat: the BEST of prescription, prescription_linear, prescription_slow (per analysis cell).
- Primary outcome: normalised regret at each query budget T in {0, 5, 10, 20, 30, 50}; unit of analysis = listener.
