# AudioTwin: methods spec (v0.1)

## Listener and context
- True audiogram a in R^7 (dB HL: 250, 500, 1k, 2k, 4k, 6k, 8k Hz); comfort limits u per band.
- Twin estimate a_hat = a + eps, eps ~ N(0, sigma_a^2 I)  (smartphone audiometry error).
- Contexts C = {quiet speech, noisy conversation, lecture, call, outdoor, music, live/worship}.

## Latent parameterisation  z in [-1,1]^5
z1 overall gain offset | z2 spectral tilt | z3 compression scale | z4 noise-reduction | z5 speech-enhancement
- g_k(z)  = clip( g_k^Rx(a_hat) + b1*z1 + b2*z2*log2(f_k/1000), 0, g_k^max(a_hat) )
- CR_k(z) = clip( CR_k^Rx * 2^(z3), 1, CR_max )
- NR(z)   = NR_max * (z4+1)/2 ;  SE(z) = SE_max * (z5+1)/2
- Prescription g^Rx: public rule (NAL-R or half-gain). z=0 is the prescription.
- Bands (10): 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000 Hz.

## Safety set S(a_hat, c)
g_k <= g_k^max ; predicted output <= UCL_k ; |g_k - g_{k+1}| <= delta_s ; ||z - z_prev||_inf <= delta_r.
Applied to every candidate in acquisition and recommendation.

## Oracle (evaluation only)
Hidden utility u*(z; listener, context). P(A>B) = lam/2 + (1-lam)*Phi((u*(zA)-u*(zB))/(sqrt2*sigma_n)).
u* = weighted sum: intelligibility term + quality term + loudness-comfort penalty + distance to ideal gain curve.
Weights sampled per listener; >=3 oracle families. Keep oracle components disjoint from reported metrics.

## Twin preference model
f: Z x C -> R, GP prior; mean m = -gamma*||z||^2; kernel = Matern52(ARD) on z  x  context-similarity kernel.
Probit pairwise likelihood (Chu & Ghahramani); Laplace posterior (BoTorch PairwiseGP).

## Query selection
qEUBO (default) | BALD (ablation) | random pairs (ablation). Acquisition uses GP only; audio rendered only to query oracle.
Recommendation z*_T = argmax_{z in S} E[f(z,c)].

## Evaluation
simple regret r_T (normalised) | N_eps = min{T : r_T <= eps} | HASPI, HASQI, ESTOI, PESQ on held-out speech/noise |
safety-violation rate | robustness vs sigma_a, sigma_n, lapse, context-classifier accuracy.
Stats: paired Wilcoxon, bootstrap CIs, Holm correction.

## Baselines (same parameterisation and stimulus budget)
no correction | fixed EQ | prescription (z=0) | simulated manual self-adjust | in-situ audiometry fit |
AudioTwin-random | AudioTwin-BALD | AudioTwin-qEUBO.
