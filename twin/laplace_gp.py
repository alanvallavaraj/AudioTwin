"""Pairwise-preference GP with a Laplace posterior (Chu & Ghahramani, 2005).

Latent utility f ~ GP(m, k) with m(z) = -gamma*||z||^2 (soft pull to the prescription, z=0) and a
Matern-5/2 kernel. A comparison "w beats l" has likelihood Phi((f(w)-f(l)) / (sqrt(2)*noise_scale)).
MAP by damped Newton (the probit log-likelihood is concave in f, so this is well behaved); posterior
covariance (K^-1 + W)^-1 at the training points, propagated to any candidate set. Unlike the earlier
`PairwisePreferenceGP`, the predictive covariance includes the uncertainty about f at the queried
points, which acquisition functions need. Pure NumPy/SciPy: CPU-only, deterministic, HPC-friendly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy.special import log_ndtr

SQRT5 = np.sqrt(5.0)
HYPER_GRID = {
    "lengthscale": (0.6, 1.0, 1.6),
    "signal_var": (0.1, 0.3, 1.0),
    "noise_scale": (0.05, 0.10, 0.20),
}


def matern52(xa, xb, lengthscale: float, variance: float) -> np.ndarray:
    xa = np.atleast_2d(np.asarray(xa, dtype=float))
    xb = np.atleast_2d(np.asarray(xb, dtype=float))
    d = np.sqrt(np.maximum(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1), 0.0)) / lengthscale
    return variance * (1.0 + SQRT5 * d + 5.0 * d**2 / 3.0) * np.exp(-SQRT5 * d)


def _lam(m: np.ndarray) -> np.ndarray:
    """phi(m)/Phi(m), computed stably."""
    return np.exp(-0.5 * m * m - 0.5 * np.log(2.0 * np.pi) - log_ndtr(m))


@dataclass
class LaplacePreferenceGP:
    lengthscale: float = 1.0
    signal_var: float = 0.3
    noise_scale: float = 0.10     # sigma_n assumed by the twin (the real listener's is unknown)
    gamma: float = 0.1            # prior mean m(z) = -gamma * ||z||^2
    jitter: float = 1e-6
    X: np.ndarray | None = None
    comps: np.ndarray | None = None
    mu_X: np.ndarray | None = None
    f_map: np.ndarray | None = None
    K: np.ndarray | None = None
    K_inv: np.ndarray | None = None
    Sigma: np.ndarray | None = None
    H: np.ndarray | None = None

    # ---- model pieces -------------------------------------------------------------------
    def kernel(self, xa, xb) -> np.ndarray:
        return matern52(xa, xb, self.lengthscale, self.signal_var)

    def prior_mean(self, Z) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        return -self.gamma * np.sum(Z**2, axis=1)

    @property
    def _s(self) -> float:
        return float(np.sqrt(2.0) * self.noise_scale)

    # ---- fitting ------------------------------------------------------------------------
    def fit(self, X, comparisons) -> "LaplacePreferenceGP":
        """X: (n, d) distinct settings; comparisons: (m, 2) index pairs (winner, loser)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        comps = np.asarray(comparisons, dtype=int).reshape(-1, 2)
        n = len(X)
        K = self.kernel(X, X) + self.jitter * np.eye(n)
        K_inv = np.linalg.inv(K)
        mu = self.prior_mean(X)
        w, l = comps[:, 0], comps[:, 1]
        s = self._s

        def psi(f):
            m = (f[w] - f[l]) / s
            return 0.5 * float((f - mu) @ K_inv @ (f - mu)) - float(log_ndtr(m).sum())

        def derivs(f):
            m = (f[w] - f[l]) / s
            lam = _lam(m)
            g_ll = np.zeros(n)
            np.add.at(g_ll, w, lam / s)
            np.add.at(g_ll, l, -lam / s)
            c = lam * (m + lam) / s**2
            W = np.zeros((n, n))
            np.add.at(W, (w, w), c)
            np.add.at(W, (l, l), c)
            np.add.at(W, (w, l), -c)
            np.add.at(W, (l, w), -c)
            return g_ll, W

        f = mu.copy()
        cur = psi(f)
        for _ in range(100):
            g_ll, W = derivs(f)
            grad = K_inv @ (f - mu) - g_ll
            step = np.linalg.solve(K_inv + W, grad)
            t = 1.0
            while t > 1e-10:
                f_new = f - t * step
                new = psi(f_new)
                if new <= cur + 1e-12:
                    break
                t *= 0.5
            else:
                break
            converged = abs(cur - new) < 1e-10
            f, cur = f_new, new
            if converged:
                break
        _, W = derivs(f)
        H = K_inv + W
        self.X, self.comps, self.mu_X, self.f_map = X, comps, mu, f
        self.K, self.K_inv, self.H = K, K_inv, H
        self.Sigma = np.linalg.inv(H)
        return self

    def log_marginal(self) -> float:
        """Laplace approximation to log p(comparisons | hyperparameters)."""
        if self.f_map is None:
            raise RuntimeError("fit first")
        w, l = self.comps[:, 0], self.comps[:, 1]
        ll = float(log_ndtr((self.f_map[w] - self.f_map[l]) / self._s).sum())
        d = self.f_map - self.mu_X
        quad = 0.5 * float(d @ self.K_inv @ d)
        _, ld_K = np.linalg.slogdet(self.K)
        _, ld_H = np.linalg.slogdet(self.H)
        return ll - quad - 0.5 * (ld_K + ld_H)

    # ---- prediction ---------------------------------------------------------------------
    def posterior(self, Zs) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean (m,) and covariance (m, m) of the latent utility at Zs."""
        Zs = np.atleast_2d(np.asarray(Zs, dtype=float))
        Kss = self.kernel(Zs, Zs)
        if self.X is None or len(self.comps) == 0:
            return self.prior_mean(Zs), Kss
        Ks = self.kernel(Zs, self.X)
        A = Ks @ self.K_inv
        mean = self.prior_mean(Zs) + A @ (self.f_map - self.mu_X)
        cov = Kss - A @ Ks.T + A @ self.Sigma @ A.T
        cov = 0.5 * (cov + cov.T)
        return mean, cov

    def predict_mean(self, Zs) -> np.ndarray:
        return self.posterior(Zs)[0]


def fit_with_selection(X, comparisons, base: LaplacePreferenceGP | None = None,
                       grid: dict | None = None, min_comparisons: int = 4) -> LaplacePreferenceGP:
    """Fit by maximising the Laplace marginal likelihood over a small hyperparameter grid.
    With fewer than `min_comparisons` comparisons the data cannot identify hyperparameters,
    so the defaults are kept."""
    base = base or LaplacePreferenceGP()
    comps = np.asarray(comparisons, dtype=int).reshape(-1, 2)
    if len(comps) < min_comparisons:
        return replace(base).fit(X, comps)
    g = grid or HYPER_GRID
    best, best_lm = None, -np.inf
    for ls, sv, ns in product(g["lengthscale"], g["signal_var"], g["noise_scale"]):
        m = replace(base, lengthscale=ls, signal_var=sv, noise_scale=ns).fit(X, comps)
        lm = m.log_marginal()
        if lm > best_lm:
            best, best_lm = m, lm
    return best
