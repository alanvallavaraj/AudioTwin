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
