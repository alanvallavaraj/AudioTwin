# AudioTwin methods specification — reported final implementation

This document describes the implementation used by the reported final simulation. Exact numeric assumptions are listed in `docs/simulator_parameters.md`. Historical development variants remain visible on the `protocol-history` branch.

## Listener and context

- True audiogram `a` in R^7 at 250, 500, 1000, 2000, 4000, 6000 and 8000 Hz.
- Twin estimate `a_hat = a + eps`, with independent Gaussian observation error and 5-dB rounding.
- Final contexts: noisy conversation, lecture and music.
- Final primary preference families: intelligibility, comfort and balanced.
- `noise_sensitive` is post-hoc and reported separately.

## Latent parameterisation `z in [-1,1]^6`

`z1` overall gain offset | `z2` spectral tilt | `z3` compression scale | `z4` noise reduction | `z5` speech enhancement | `z6` compression speed

- `g_k(z) = clip(g_k^Rx(a_hat) + b1*z1 + b2*z2*log2(f_k/1000), 0, g_k^max(a_hat))`
- **Reported prescription:** `g_k^Rx(a) = 0.5 * interpolated_threshold_k`. This is a synthetic half-gain rule; it is not NAL-NL2 or NAL-R.
- `CR_k(z) = clip(CR_k^Rx * 2^(2*z3), 1, CR_max)`; `z3=-1` reaches linear compression.
- `NR(z)=NR_max*(z4+1)/2`; `SE(z)=SE_max*(z5+1)/2`.
- Attack/release interpolate logarithmically from 5/50 ms at `z6=0` to 50/500 ms at `z6=1`; the same log-linear map extends to 0.5/5 ms at `z6=-1`. Timing has no utility effect in the primary families.
- Ten DSP bands: 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000 and 8000 Hz.

The hidden oracle's ideal-gain curve uses the same half-gain construction evaluated on the true audiogram plus listener-specific random offsets. This shared structure is explicit in the manuscript limitations.

## Feasible domain used by the final active loop

Final qEUBO/BALD/GP-random candidate generation enforces:

- latent box `z in [-1,1]^6`;
- gain bounds induced by `g_k^max`;
- fixed 100-dB-SPL ceiling on predicted band output for a 65-dB-SPL reference input (static compression curve, after the 105-dB-SPL limiter);
- maximum adjacent-band gain difference.

A conservative low-gain anchor is found first and candidate settings are projected towards it when necessary. The codebase contains an optional `z_prev` change-rate check in `dsp.safety.project_to_S`, but **the reported final active-learning path uses `twin.optimizer.project_from_anchor`, which checks feasibility without a previous setting, so no sequential change-rate constraint is applied**.

## Oracle (evaluation only)

`u*(z; listener, context)` is a weighted sum of:

- an audibility/intelligibility-like term;
- loudness-comfort penalty;
- distance to the hidden ideal-gain curve;
- processing/compression artefact penalty.

Pairwise responses follow a lapse-contaminated probit choice model. Compression-speed artefacts occur only in the post-hoc `noise_sensitive` family. Therefore `z6` has no utility effect in the three pre-specified primary families.

## Preference GP

The reported learned methods use a pairwise probit GP with:

- Matérn-5/2 kernel;
- prior mean `m(z)=-0.1 ||z||^2`;
- Laplace posterior around the MAP latent utilities;
- grid selection of lengthscale, signal variance and assumed comparison noise every five queries.

Predictive covariance propagates posterior uncertainty at queried settings to candidate points.

## Query selection

The final matched methods share the same preference GP and candidate-generation mechanism:

- **qEUBO:** expected utility of the better item in a two-setting query;
- **BALD:** approximate mutual-information acquisition;
- **GP-random:** random admissible pair selection.

The recommendation at each checkpoint maximises posterior mean over a feasible candidate pool. Checkpoints are 0, 5, 10, 20, 30 and 50 queries.

## Static configurations

Three rows were recorded in the final result grid:

- `prescription`: half-gain + prescribed compression, `z4=z5=0` (0.5 NR/SE), fast timing;
- `prescription_linear`: same gain rule, linear compression, **NR and SE both off** (`z3=z4=z5=-1`);
- `prescription_slow`: default prescription with `z6=1` (50/500-ms timing).

Because compression speed is absent from the three primary utility families, `prescription_slow` and `prescription` are utility-identical there. Thus the primary per-cell static minimum has only **two distinct utility behaviours**. Slow compression is distinct in the post-hoc `noise_sensitive` family.

## Final analysis

- Unit of analysis: listener.
- Primary endpoint: qEUBO T=20 minus the per-cell best static regret, averaged over 3 contexts × 3 pre-specified primary families within listener.
- 20,000 listener-level bootstrap resamples, seed 20261011.
- Final analysable N=128, including declared replacement task 128 for infeasible task 55.
- Released data: `publication/data/final_v1.parquet`.
