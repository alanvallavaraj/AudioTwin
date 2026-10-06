# AudioTwin: methods spec (v0.3)

## Listener and context
- True audiogram a in R^7 (dB HL: 250, 500, 1k, 2k, 4k, 6k, 8k Hz); comfort limits u per band.
- Twin estimate a_hat = a + eps, eps ~ N(0, sigma_a^2 I)  (smartphone audiometry error).
- Contexts C = {quiet speech, noisy conversation, lecture, call, outdoor, music, live/worship}.

## Latent parameterisation  z in [-1,1]^6
z1 overall gain offset | z2 spectral tilt | z3 compression scale | z4 noise-reduction | z5 speech-enhancement | z6 compression speed
- g_k(z)  = clip( g_k^Rx(a_hat) + b1*z1 + b2*z2*log2(f_k/1000), 0, g_k^max(a_hat) )
- CR_k(z) = clip( CR_k^Rx * 2^(2*z3), 1, CR_max )   # z3=-1 reaches linear (CR=1)
- NR(z)   = NR_max * (z4+1)/2 ;  SE(z) = SE_max * (z5+1)/2
- Attack/release use log interpolation from 5/50 ms at z6=0 to 50/500 ms at z6=1.
- Prescription g^Rx: public rule (NAL-R or half-gain). z=0 is the original fast-compression prescription.
- Bands (10): 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000 Hz.

## Safety set S(a_hat, c)
g_k <= g_k^max ; predicted output <= UCL_k ; |g_k - g_{k+1}| <= delta_s ; ||z - z_prev||_inf <= delta_r.
Applied to every candidate in acquisition and recommendation. If z=0 is unsafe for an unusually severe/noisy measured audiogram, sampling shrinks candidates toward a conservative low-gain anchor.

## Oracle (evaluation only)
Hidden utility u*(z; listener, context). P(A>B) = lam/2 + (1-lam)*Phi((u*(zA)-u*(zB))/(sqrt2*sigma_n)).
u* = weighted sum: intelligibility term + quality term + loudness-comfort penalty + distance to ideal gain curve.
The quality term now includes a compression-speed artefact in noise, motivated by the HASPI sanity check where fast compression reduced predicted intelligibility.
Weights sampled per listener; 3 pre-registered families (intelligibility, comfort, balanced) plus 1 POST-HOC
family (noise_sensitive: extra compression-in-noise penalty, added after the HASPI/compression finding). Keep oracle components disjoint from reported metrics.

## Twin preference model
Stage-4 implementation: pairwise probit GP over queried z values with an RBF kernel, MAP latent utility fitting,
and GP-prior interpolation to feasible candidate settings. This is deterministic, CPU-only and covered by unit tests.
Planned full model: f: Z x C -> R, GP prior; mean m = -gamma*||z||^2; kernel = Matern52(ARD) on z x context-similarity kernel.
Probit pairwise likelihood (Chu & Ghahramani); Laplace posterior (BoTorch PairwiseGP).

## Query selection
Implemented baselines: random safe pair queries, mean-plus-uncertainty active queries, and exploitative active queries.
Planned: qEUBO (default) | BALD (ablation) | random pairs (ablation). Acquisition uses GP only; audio rendered only to query oracle.
Recommendation z*_T = argmax_{z in S} E[f(z,c)].

## Evaluation
simple regret r_T (normalised) | N_eps = min{T : r_T <= eps} | HASPI, HASQI, ESTOI, PESQ on held-out speech/noise |
safety-violation rate | robustness vs sigma_a, sigma_n, lapse, context-classifier accuracy.
Stats: paired Wilcoxon, bootstrap CIs, Holm correction.

## Baselines (same parameterisation and stimulus budget)
no correction | fixed EQ | prescription (z=0) | prescription-linear (CR=1) | prescription-slow (z6=1 / 50/500 ms) |
simulated manual self-adjust | in-situ audiometry fit |
AudioTwin-random | AudioTwin-active | AudioTwin-exploit | AudioTwin-BALD | AudioTwin-qEUBO.
