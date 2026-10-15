## Current target journal

The current AudioTwin submission package targets **Applied Soft Computing (Elsevier)** and is stored in `manuscript_asoc/`. The manuscript is anonymised for peer review; the separate editable title page contains author details.

# Current manuscript submission

The current submission package targets **Array (Elsevier)** and is stored in `manuscript_array/`. It was prepared after transfer from BSPC manuscript `BSPC-D-26-20436`. The abstract, introduction and conclusion have been retargeted to emphasise the engineering contribution and the potential real-world benefit of reducing unnecessary hearing-aid fine-tuning, while clearly stating that the present evidence is simulation-based and requires human-listener validation.

The four upload-ready files are in `manuscript_rie/submission_files/`: Declaration of Interests Statement, Highlights, Manuscript, and editable Title Page.

# AudioTwin

**AudioTwin** is a reproducible simulation framework for preference-guided, context-aware hearing calibration. It models listener-specific hearing profiles, safe DSP parameterisation, pairwise preference feedback, and active Bayesian optimisation for rapid personalisation.

This repository accompanies the AudioTwin publication and contains the reference implementation, released final synthetic dataset, final analysis code, tests, manuscript materials, and the original protocol history. `main` is the curated publication branch; the original pre-final development lineage is published separately as `protocol-history` with the original annotated protocol tags. See [`PROTOCOL_PROVENANCE.md`](PROTOCOL_PROVENANCE.md).

## Key result

On the frozen final evaluation of **128 simulated listeners**, the Laplace preference-GP with qEUBO acquisition reduced mean normalised regret at **20 preference queries** from **0.3212** for the best static comparator to **0.2405**.

- Paired mean difference: **-0.0807** (lower is better)
- 95% listener-level bootstrap CI: **[-0.1027, -0.0593]**
- Approximate reduction in mean regret: **25.1%**
- Listeners improved relative to the best static reference: **94 of 128**
- Final evaluation size: **N = 128**

qEUBO also outperformed the matched GP-random policy (same model, pools and seeds) at the pre-specified early budgets:

- 5 queries: **-0.0495**, 95% CI **[-0.0627, -0.0364]**
- 10 queries: **-0.0371**, 95% CI **[-0.0489, -0.0252]**

In exploratory analysis the gap narrows but persists at every budget up to 50 queries (`publication/results/final_v1_exploratory.txt`).

Limitations are reported alongside the gains. Gains are largest for comfort-oriented preferences and in the noisy-conversation and lecture contexts. In music with intelligibility-oriented preferences a static setting remains better. Development-seed robustness checks show that the advantages over static fitting and over random pair selection both disappear when simulated preference-response noise is doubled. The simulator's prescription is a synthetic half-gain rule, and all listeners are simulated.

## Repository structure

```text
baselines/       Static comparator implementations
configs/         Experiment configuration
 docs/           Methods and frozen protocol
 dsp/            DSP parameterisation and safety constraints
 eval/           Evaluation metrics and statistics
 experiments/    Experiment runner
 oracle/         Simulated listeners, contexts and utility families
 publication/    Released final synthetic data, summaries, and post-final diagnostics
 scripts/        Frozen final analysis, publication summaries, exploratory and diagnostic scripts
 slurm/          Reproducible HPC launchers for development/final runs
 tests/          Unit and regression tests
 twin/           Preference GP, acquisitions and active optimisation
```

## Manuscript

The current Elsevier **Biomedical Signal Processing and Control** manuscript source and compiled submission bundle are available in [`manuscript_bspc/`](manuscript_bspc/). The bundle contains the complete `elsarticle` LaTeX manuscript, BibTeX references, publication figures (with their generating scripts), highlights, cover letter and compiled PDF.

## Frozen final protocol

The final evaluation is documented in [`docs/protocol.md`](docs/protocol.md). Original pre-final history is available on the `protocol-history` branch. The publication analysis state is identified by:

- Scientific freeze: `final-protocol-v1` (commit `52fcd8b`)
- Final-run code: `final-protocol-v1.2` (commit `fc040b3`, stored in every result row)
- Replacement-rule record: `final-protocol-v1.3` (commit `7b3ab57`)
- Submitted manuscript snapshot: `manuscript-bspc-v1.0`
- Analysis/bootstrap seed: `20261011`
- Final experiment seed: `777000111`

One pre-specified final listener draw (`task_id=55`) had no feasible safe low-gain anchor under the frozen constraints. It was not forced through the optimiser. A deterministic replacement rule was frozen before inspecting final performance outcomes, and the next unused listener (`task_id=128`) was used to preserve **N=128**.

## Reproduce the analysis

For exact final-run package versions, use `requirements-final.txt` (Python 3.11.7). For development, create a Python environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the tests:

```bash
pytest -q -m "not slow"
```

The consolidated final synthetic dataset is released at `publication/data/final_v1.parquet`. Recompute the validated analysis directly with:

```bash
python scripts/analyze_final.py publication/data/final_v1.parquet
```

The same analyser also accepts the original directory of 128 `task_*.parquet` files. Regenerate the compact summaries, the exploratory final-cohort analyses and the figures with:

```bash
python scripts/make_publication_summaries.py publication/data/final_v1.parquet
python scripts/analyze_exploratory_final.py publication/data/final_v1.parquet > publication/results/final_v1_exploratory.txt
python manuscript_bspc/make_manuscript_figures.py      # needs matplotlib
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the complete workflow.

## Main methods

The final files record three static configurations and three preference-learning strategies. In the three primary utility families, `prescription_slow` is utility-identical to `prescription`, so the primary static reference contains only two distinct utility behaviours:

- `prescription`
- `prescription_linear`
- `prescription_slow` — recorded separately but identical to `prescription` for the primary families; distinct only in the post-hoc `noise_sensitive` family
- `audiotwin_qeubo`
- `audiotwin_bald`
- `audiotwin_gprandom`

The learned methods use the same Laplace-approximated preference GP and candidate framework; qEUBO, BALD and random pair selection differ only in acquisition strategy.

Query-budget checkpoints are **0, 5, 10, 20, 30 and 50**.

## Publication outputs

The `publication/data/` directory contains:

- `final_v1.parquet` — consolidated 32,256-row final synthetic dataset used to recompute the inferential analysis

The `publication/results/` directory contains:

- `final_v1_analysis.txt` — frozen primary and secondary statistical output
- `final_v1_exploratory.txt` — exploratory re-analysis of the released rows (win counts, full matched acquisition table, NR/SE geometry)
- `final_context_family_qeubo_t20.csv` — qEUBO T=20 context/family breakdown
- `final_learned_curve_means.csv` — learned-method regret curves
- `final_learned_curve_ci.csv` — the same curves with listener-level bootstrap intervals (figure input)

The `publication/postfinal_diagnostics/` directory contains development-seed-only exploratory diagnostics (see its README and `docs/protocol_v2_addendum.md`).

## Licence

Released under the MIT License; see [`LICENSE`](LICENSE).

## Citation

A machine-readable citation file is provided in [`CITATION.cff`](CITATION.cff). Please cite the associated paper once its bibliographic details are finalised.

## Author

**Anonymous Author**
Anonymous Institution

## Status

Submission snapshot `manuscript-bspc-v1.0` for *Biomedical Signal Processing and Control*. Scientific code is functionally identical to the frozen final-run commit `fc040b3`; `tests/test_frozen_equivalence.py` checks this. Bibliographic metadata will be updated on publication.
