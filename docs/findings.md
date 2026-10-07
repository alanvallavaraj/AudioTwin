# Working findings (to resolve before/while writing the paper)

Sanity run: 6 s LibriSpeech clip (SLR12 via librosa example, CC-BY-4.0) at 65 dB SPL,
synthetic mild->moderate sloping audiogram [25,30,40,50,60,65,70] dB HL, speech-shaped noise.
These are SANITY numbers only (one clip, one audiogram), not results.

1. **ESTOI is audibility-blind.** Unaided clean speech scores ESTOI 1.000, and STOI is
   gain-invariant (test). Report HASPI/HASQI as primary; ESTOI/PESQ only after a
   hearing-loss simulation (e.g. MSBG in pyclarity), and say so.
2. **HASPI saturates in quiet at 65 dB SPL** (1.000 for any amplified setting). Quiet cannot
   discriminate fittings; use soft speech (e.g. 50 dB SPL) and/or noise conditions.
3. **Fast compression lowers HASPI in noise (5 dB SNR).** Same gains: CR=1 HASPI 0.726
   (> unaided 0.655) vs as-built CR 1.2-2.5 at 5/50 ms HASPI 0.494; slower 50/500 ms gives 0.704.
   NR/SE hooks as built change HASPI by ~0.003 (SE is identity; NR is a weak expander).
   Implications:
   - The 'prescription' baseline is handicapped by fast-compression defaults. Baselines MUST
     get fair compression settings (e.g. add a linear-prescription baseline and/or a
     slow-compression variant, report both), or AudioTwin 'wins' just by lowering CR.
   - The oracle utility does not penalise compression-in-noise, while reported HASPI does.
     AudioTwin (optimising the oracle) may therefore not improve HASPI in noise. Decide:
     add a context-dependent compression-distortion term to the oracle (less disjoint from
     the metrics, but defensible) or accept and report the oracle/metric gap honestly.
4. **Cost:** ~5 s per condition (HASPI+HASQI+ESTOI, 6 s clip) on this laptop. Evaluate only the
   final recommended settings, not every query step. Example grid
   60 listeners x 8 methods x 7 contexts x 4 SNRs x 5 utterances = 67,200 evals ~ 93 CPU-hours.

## Decision record (2026-10-06): how finding 3 was handled
- **Main oracle unchanged (option b).** The 3 pre-registered families (intelligibility, comfort,
  balanced) are NOT modified in response to the HASPI/compression result. This keeps the oracle
  independent of the reported metrics. A gap between oracle regret and HASPI in noise is reported
  as a finding, not engineered away.
- **Post-hoc family added:** `noise_sensitive` (extra compression-in-noise penalty). Added AFTER
  seeing the metric behaviour; must be labelled "post-hoc robustness check" and reported separately
  (`oracle.POST_HOC_FAMILIES`).
- **Baselines strengthened:** `prescription_linear` (CR=1, no NR/SE) and `prescription_slow`
  (50/500 ms) added next to `prescription`. AudioTwin must beat the BEST of the three.
- **Search space fix:** compression mapping now CR = CR_rx * 2^(2*z3), so z3=-1 reaches CR=1 for
  every prescribed CR (was only reaching ~1.25 for the most severe listeners). Compression
  time constants are still fixed, not a latent control (optional 6th control, not implemented).
- Non-vacuous tests added for each change (tests/test_baselines_and_changes.py).

## Correction (2026-10-07): oracle leak reverted; pilots are not comparable
- Commit 466e0b6 added a HASPI-motivated compression-speed penalty to the SHARED quality term, so it applied to the
  3 pre-registered families too. This contradicted the decision record above. Reverted: the penalty now exists only
  in `noise_sensitive` (post-hoc). Tests guard this (tests/test_baselines_and_changes.py).
- All seven pilot sets in results_hpc/ (code 4bdf2a9 ... 360def2) were produced under the LEAKED oracle (and the
  early ones under an older parameterisation). They are development data only, are not comparable with new runs,
  and must not be reported as results. Pilot readout (32 listeners, 12 queries, regret lower=better): the static
  prescription_linear (~0.35 pre-registered, ~0.22 noise_sensitive) was as good as or better than every
  AudioTwin variant; active/exploit acquisition scored worse than random queries.
- The hand-seeded 'prior' / 'local_prior' variants encode the linear/slow finding into the search and are NOT
  headline methods; keep only as clearly-labelled post-hoc ablations.
