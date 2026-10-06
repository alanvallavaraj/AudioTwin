# AudioTwin

Purely computational study (NO human participants). Simulated listeners only.
Never claim clinical benefit or real-user benefit. Frame as hearing-support calibration, not diagnosis.

## Rules
- Spec is in docs/methods.md. Follow its notation (z in [-1,1]^5, safety set S, oracle utility u*).
- Oracle components must stay separate from reported evaluation metrics (HASPI/HASQI/PESQ/ESTOI).
- Everything seeded; log git hash, config and seed in every output file.
- Configs via Hydra; results as Parquet, one file per array task.
- Jobs must be idempotent (skip if output exists).
- Target cluster: QMUL Apocrita (Slurm). Do not hard-code partition names; take them from config.
- Build in stages; each stage needs unit tests before the next starts.
- Do not commit data, audio or results.

## Layout
twin/      listener twin + preference GP + acquisition
dsp/       multiband filter bank, compression, tilt, NR/SE hooks, safety clipping
oracle/    simulated listeners (hidden utility u*, choice noise, lapses)
baselines/ no-correction, fixed EQ, prescription, manual self-adjust, in-situ fit
eval/      metric wrappers (HASPI/HASQI via pyclarity, ESTOI, PESQ), stats
configs/   Hydra configs
slurm/     sbatch array scripts
tests/     pytest

## Stages
1 dsp  2 oracle  3 eval metrics  4 twin  5 baselines  6 runner + slurm
