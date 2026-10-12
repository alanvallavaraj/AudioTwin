# AudioTwin

**AudioTwin** is a reproducible simulation framework for preference-guided, context-aware hearing calibration. It models listener-specific hearing profiles, safe DSP parameterisation, pairwise preference feedback, and active Bayesian optimisation for rapid personalisation.

This repository accompanies the AudioTwin publication and contains the reference implementation, released final synthetic dataset, final analysis code, tests, manuscript materials, and the original protocol history. `main` is the curated publication branch; the original pre-final development lineage is published separately as `protocol-history` with the original annotated protocol tags. See [`PROTOCOL_PROVENANCE.md`](PROTOCOL_PROVENANCE.md).

## Key result

On the frozen final evaluation of **128 simulated listeners**, the Laplace preference-GP with qEUBO acquisition reduced mean normalised regret at **20 preference queries** from **0.3212** for the best static comparator to **0.2405**.

- Paired mean difference: **-0.0807** (lower is better)
- 95% listener-level bootstrap CI: **[-0.1027, -0.0593]**
- Approximate reduction in mean regret: **25.1%**
- Final evaluation size: **N = 128**

qEUBO also outperformed GP-random at early budgets, supporting improved query efficiency:

- 5 queries: **-0.0495**, 95% CI **[-0.0627, -0.0364]**
- 10 queries: **-0.0371**, 95% CI **[-0.0489, -0.0252]**

The context/family analysis is intentionally reported without hiding limitations: gains are strongest for comfort-oriented calibration and in noisy-conversation/lecture contexts, while music-intelligibility remains a challenging case.

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
 scripts/        Diagnostics and frozen final-analysis script
 slurm/          Reproducible HPC launchers for development/final runs
 tests/          Unit and regression tests
 twin/           Preference GP, acquisitions and active optimisation
```

## Manuscript

The current Elsevier **Machine Learning with Applications** manuscript source and compiled submission bundle are available in [`manuscript_mlawa/`](manuscript_mlawa/). The bundle contains the complete `elsarticle` LaTeX manuscript, BibTeX references, publication figures, highlights, cover letter draft, and compiled PDF.

## Frozen final protocol

The final evaluation is documented in [`docs/protocol.md`](docs/protocol.md). Original pre-final history is available on the `protocol-history` branch. The publication analysis state is identified by:

- Scientific freeze: `final-protocol-v1`
- Operational final-run tag / code hash: `final-protocol-v1.2` / `fc040b3`
- Replacement-rule record: `final-protocol-v1.3`
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

The same analyser also accepts the original directory of 128 `task_*.parquet` files. The compact publication outputs are provided in [`publication/results/`](publication/results/).

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
- `final_context_family_qeubo_t20.csv` — qEUBO T=20 context/family breakdown
- `final_learned_curve_means.csv` — learned-method regret curves

## Licence

Released under the MIT License; see [`LICENSE`](LICENSE).

## Citation

A machine-readable citation file is provided in [`CITATION.cff`](CITATION.cff). Please cite the associated paper once its bibliographic details are finalised.

## Author

**Dr Alan Immanuel Benjamin Vallavaraj**
School of Computer Science and Engineering, University of Westminster

## Status

Publication repository snapshot prepared from the frozen AudioTwin final-evaluation state. Manuscript bibliographic metadata will be updated when the paper is formally submitted/published.
