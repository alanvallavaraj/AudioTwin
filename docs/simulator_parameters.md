# AudioTwin simulator parameters used in the reported study

These values define a **synthetic computational testbed**. They are fixed modelling assumptions, not clinical prescription constants and not a claim that the simulator is a validated auditory model. The final experiment used the code hash stored in the released data (`fc040b3`).

## Hearing profile and observation model

| Quantity | Value / distribution |
|---|---|
| Audiogram frequencies | 250, 500, 1000, 2000, 4000, 6000, 8000 Hz |
| Parametric audiogram 1-kHz base | Uniform(15, 55) dB HL |
| High-frequency slope above 1 kHz | Uniform(0, 12) dB/octave |
| Low-frequency slope below 1 kHz | Uniform(0, 4) dB/octave |
| Per-frequency perturbation | Normal(0, 4 dB), then rounded to 5 dB |
| Accepted PTA4 range | 20–55 dB HL (500, 1000, 2000, 4000 Hz) |
| Estimated-audiogram noise | Normal(0, 5 dB) per measured frequency, rounded to 5 dB |
| Synthetic threshold-to-SPL offsets | [26, 12, 7, 9, 10, 15, 13] dB |
| UCL construction | max(threshold + 25 dB, 100 + Normal(0,4) dB), capped at 110 dB SPL |
| Ideal-gain offset | 3 random log-frequency knots Normal(0,4 dB), plus listener offset Normal(0,3 dB) |
| Compression preference | Uniform(0,1) in log2 mean-CR units |
| Choice-noise sigma | Uniform(0.05, 0.15) utility units |
| Lapse probability | Uniform(0, 0.05) |

The hidden ideal gain is **not independently prescribed**: it is the same synthetic half-gain curve used by the fitting system, evaluated on the true audiogram, plus the listener-specific ideal-gain offset above.

## DSP and feasible-domain assumptions

| Parameter | Value |
|---|---:|
| DSP bands | 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000 Hz |
| Latent dimension | 6 |
| Half-gain prescription factor | 0.5 × interpolated dB-HL threshold |
| Overall gain span (`z1`) | 10 dB per latent unit |
| Spectral tilt (`z2`) | 6 dB/octave per latent unit, 1-kHz pivot |
| Maximum-gain rule | clip(0.8 × threshold + 5 dB, 0, 50 dB) |
| Compression prescription | 1 + 1.5 × clip((threshold − 20)/40, 0, 1) |
| Compression scaling (`z3`) | prescribed CR × 2^(2 z3), clipped to [1,4] |
| Compression kneepoint | 45 dB SPL |
| Reference input for output check | 65 dB SPL |
| UCL output ceiling used by feasibility check | 100 dB SPL |
| MPO limiter | 105 dB SPL |
| Adjacent-band gain-difference limit | 12 dB |
| Noise-reduction range (`z4`) | 0–1 |
| Speech-enhancement range (`z5`) | 0–1 |
| Fast attack/release (`z6=0`) | 5/50 ms |
| Slow attack/release (`z6=1`) | 50/500 ms |

`DSPConfig.change_rate_delta=0.5` exists for the legacy `project_to_S(..., z_prev=...)` utility, but the **reported final active-learning loop did not pass a previous setting and therefore did not enforce a sequential change-rate constraint**. The final feasible domain used the latent box, output/UCL check and adjacent-band gain smoothness.

## Context assumptions

Long-term speech-band levels at 65 dB overall are `[55,55,54,52,49,47,44,41,36,31]` dB SPL. Normalised importance weights are proportional to `[0.06,0.09,0.09,0.12,0.12,0.14,0.14,0.12,0.08,0.04]`.

| Context | Speech dB SPL | SNR dB | Speech-like | NR benefit at 1 | SE benefit at 1 |
|---|---:|---:|---|---:|---:|
| Noisy conversation | 68 | 3 | yes | 3 dB | 4 dB |
| Lecture | 62 | 12 | yes | 2 dB | 3 dB |
| Music | 72 | 30 | no | 0 | 0 |

Other context definitions remain in the code for development, but the final study used only the three rows above.

## Oracle assumptions

The synthetic utility is a weighted sum of audibility, comfort, ideal-gain distance and artefact terms. Family weights are sampled independently per listener from Dirichlet distributions with concentration parameters:

| Family | Audibility | Comfort | Ideal-curve | Artefacts | Status |
|---|---:|---:|---:|---:|---|
| intelligibility | 6 | 1 | 1 | 1 | pre-specified primary |
| comfort | 1 | 5 | 2 | 1 | pre-specified primary |
| balanced | 3 | 3 | 2 | 2 | pre-specified primary |
| noise_sensitive | 4 | 2 | 1 | 3 | post-hoc robustness only |

Other utility constants: speech peak factor 12 dB; comfort distance scale 10 dB; over-UCL scale 5 dB; ideal-gain distance scale 10 dB. The noise-sensitive family adds compression-in-noise penalties of 3.0 (compression magnitude) and 1.5 (fast-speed interaction). Compression speed is absent from all three primary families.

## Preference GP and acquisition

| Quantity | Value |
|---|---|
| Kernel | Matérn 5/2, isotropic |
| Prior mean | `-0.1 ||z||²` |
| Default GP lengthscale | 1.0 |
| Default signal variance | 0.3 |
| Default assumed comparison noise | 0.10 |
| Jitter | 1e-6 |
| Hyperparameter grid | lengthscale {0.6,1.0,1.6}; variance {0.1,0.3,1.0}; noise {0.05,0.10,0.20} |
| Hyperparameter refit interval | every 5 queries |
| Uniform query candidates | 192 |
| Local query candidates | 64 |
| Recommendation candidates | 512 |
| Local sampling SD | 0.25 latent units |
| Preferred minimum pair separation | 0.4 latent Euclidean distance (falls back to 0.2 then 0 if needed) |
| Query budgets | 0, 5, 10, 20, 30, 50 |

## Oracle optimum and regret normalisation

The approximate oracle optimum uses SciPy differential evolution with 400 random feasible samples for the normalisation floor, 60 maximum iterations and population size 12. The lower reference `u_lo` is the fifth percentile of those random feasible utilities. Normalised simple regret is `(u_opt-u)/(u_opt-u_lo+1e-9)`, clipped below at zero.
