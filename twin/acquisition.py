"""Closed-form pairwise acquisition scores on a candidate pool, from a Gaussian posterior (mu, C).

* EUBO (Astudillo et al., 2023): E[max(f_i, f_j)] for the pair (i, j). Equals qEUBO with q = 2.
* BALD (Houlsby et al., 2011): approximate mutual information between the answer and f, for a probit
  comparison with scale s = sqrt(2)*sigma_n.
Both return an (N, N) matrix of scores (diagonal = -inf). Candidates are always drawn from the
FEASIBLE pool, so safety constraints hold by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_C2 = np.pi * np.log(2.0) / 2.0   # C^2 in the BALD approximation


def _pair_stats(mu: np.ndarray, C: np.ndarray):
    d = np.diag(C)
    theta2 = np.maximum(d[:, None] + d[None, :] - 2.0 * C, 1e-12)
    return mu[:, None] - mu[None, :], theta2


def eubo_matrix(mu: np.ndarray, C: np.ndarray) -> np.ndarray:
    dmu, theta2 = _pair_stats(mu, C)
    theta = np.sqrt(theta2)
    alpha = dmu / theta
    pdf = np.exp(-0.5 * alpha**2) / np.sqrt(2.0 * np.pi)
    out = mu[:, None] * ndtr(alpha) + mu[None, :] * ndtr(-alpha) + theta * pdf
    np.fill_diagonal(out, -np.inf)
    return out


def _h(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def bald_matrix(mu: np.ndarray, C: np.ndarray, noise_scale: float) -> np.ndarray:
    s = np.sqrt(2.0) * noise_scale
    dmu, theta2 = _pair_stats(mu, C)
    m, v = dmu / s, theta2 / s**2
    # Houlsby's approximation is stated in bits: E[h(Phi(f))] ~ C/sqrt(v+C^2) exp(-m^2/(2(v+C^2))) with h in bits.
    # We work in nats (entropy via natural log), so that term is scaled by ln 2.
    out = _h(ndtr(m / np.sqrt(v + 1.0))) - np.log(2.0) * np.sqrt(_C2) / np.sqrt(v + _C2) * np.exp(-0.5 * m**2 / (v + _C2))
    np.fill_diagonal(out, -np.inf)
    return out


def best_pair(score: np.ndarray, Z: np.ndarray, min_sep: float) -> tuple[int, int]:
    """Argmax of `score` over pairs whose settings are at least `min_sep` apart (z-space)."""
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    masked = np.where(dist >= min_sep, score, -np.inf)
    iu = np.triu_indices(len(Z), k=1)
    k = int(np.argmax(masked[iu]))
    if not np.isfinite(masked[iu][k]):
        raise ValueError("no admissible pair in candidate pool")
    return int(iu[0][k]), int(iu[1][k])
