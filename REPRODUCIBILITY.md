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

The frozen publication state passed the complete non-slow suite before final analysis. `tests/test_frozen_equivalence.py` additionally checks that the current active loop returns the same recommendations as the frozen final-run commit `fc040b3`.

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

The curated `main` branch was assembled after the final run, but the original development lineage is published as `protocol-history` with the original annotated `final-protocol-v1` to `v1.3` tags (commits `52fcd8b`, `2017fe0`, `fc040b3`, `7b3ab57`). The released final rows record code hash `fc040b3`. These records establish repository provenance but are not an external protocol registration, so the manuscript uses **pre-specified**. See `PROTOCOL_PROVENANCE.md`.

## 8. Simulator assumptions

All fixed simulator values and distributions are listed in `docs/simulator_parameters.md`. They are modelling assumptions for a computational benchmark rather than clinical prescription constants.

## 9. Publication summaries, exploratory analyses and figures

```bash
python scripts/make_publication_summaries.py publication/data/final_v1.parquet
python scripts/analyze_exploratory_final.py publication/data/final_v1.parquet > publication/results/final_v1_exploratory.txt
python manuscript_bspc/make_manuscript_figures.py      # Figs 2-4; requires matplotlib
pdflatex manuscript_bspc/fig1_framework.tex          # Fig. 1 (TikZ)
```

`make_publication_summaries.py` reproduces `final_context_family_qeubo_t20.csv` and `final_learned_curve_means.csv` exactly and also writes `final_learned_curve_ci.csv`.

## 10. Post-final development diagnostics

These use development seed `424242` only (governed by `docs/protocol_v2_addendum.md`) and write to `publication/postfinal_diagnostics/`:

```bash
python scripts/coordinate_baseline_from_dev.py <dev_results_dir>   # needs the original dev sweep outputs
python scripts/acquisition_geometry_dev.py --tasks 32
python scripts/music_intelligibility_diagnostic.py --tasks 32
python scripts/nrse_resolution_dev.py --tasks 32
python scripts/oracle_robustness_dev.py --tasks 32
```

The robustness sweep is the most expensive (about 70 single-core seconds per listener and scenario; 160 units in total).

## 11. Known implementation note

`experiments/runner.py` seeds the audiogram-observation generator with the same base seed used to sample the listener, so measurement error and listener parameters are drawn from overlapping random streams. Over 4,000 seeds the largest absolute correlation between the first listener draw and any measurement error is below 0.04 (0.08 for the absolute error at 250 Hz). This is negligible for the reported comparisons and is kept unchanged to preserve the frozen experiment. New work should derive independent streams with `numpy.random.SeedSequence.spawn`.

