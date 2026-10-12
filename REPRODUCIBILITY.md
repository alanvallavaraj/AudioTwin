# Reproducibility

## 1. Environment

The final QMUL HPC runs used Python 3.11.7. The exact final-run environment is pinned in `requirements-final.txt`. Install those versions in an isolated Python 3.11.7 environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-final.txt
pip install -e . --no-deps
```

## 2. Verification

Run the non-slow test suite:

```bash
pytest -q -m "not slow"
```

The frozen publication state passed the complete non-slow suite before final analysis.

## 3. Final experimental design

The frozen protocol is in `docs/protocol.md`.

Core settings:

- final experiment seed: `777000111`
- listeners: 128 valid final listeners
- contexts: `noisy_conversation`, `lecture`, `music`
- pre-specified utility families: `intelligibility`, `comfort`, `balanced`
- post-hoc stress-test family: `noise_sensitive`
- query budgets: 0, 5, 10, 20, 30, 50
- learned methods: qEUBO, BALD and GP-random using the same Laplace preference GP
- bootstrap resamples: 20,000
- bootstrap seed: `20261011`

## 4. Final listener replacement

Original task 55 was infeasible under the frozen safety constraints because no safe low-gain anchor existed. No constraint was relaxed and no scientific parameter was changed. Before final performance outcomes were analysed, a deterministic replacement rule was frozen: use the next unused task ID. Task 128 replaced task 55, preserving N=128.

## 5. Run the final analysis

The released consolidated final dataset is `publication/data/final_v1.parquet`. Run:

```bash
python scripts/analyze_final.py publication/data/final_v1.parquet
```

The analyser also accepts the original 128-file `results/final_v1/` directory. The consolidated file has been verified to produce identical analysis output.

The script validates the complete expected result grid before inference and recognises the declared `{55: 128}` replacement.

## 6. Frozen publication summaries

The exact compact outputs used for manuscript preparation are under `publication/results/`.

Primary result at qEUBO T=20 versus the per-cell best static comparator across the three pre-specified utility families:

- best static mean regret: 0.3212173
- qEUBO mean regret: 0.2404755
- paired difference: -0.0807418
- 95% listener-level bootstrap CI: [-0.1027, -0.0593]

Lower regret is better.


## 7. Public-history limitation

The curated `main` branch was assembled after the final run, but the original development lineage is now published as `protocol-history` with the original annotated `final-protocol-v1` through `v1.3` tags. The released final rows record code hash `fc040b3`, which resolves to `final-protocol-v1.2`. These records establish the repository provenance but are not a external protocol registration; the manuscript therefore uses **pre-specified**.

## 8. Simulator assumptions

All fixed simulator values and distributions are listed in `docs/simulator_parameters.md`. They are modelling assumptions for a computational benchmark rather than clinical prescription constants.
