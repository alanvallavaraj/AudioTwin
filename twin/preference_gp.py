"""Small pairwise-preference GP used by the AudioTwin optimiser.

This is a deterministic NumPy/SciPy implementation for local tests and CPU Slurm jobs.
It models latent utilities only at observed settings, then interpolates the MAP latent
function to candidate settings with the GP prior. qEUBO/BALD can replace acquisition later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from dsp.params import Z_DIM


@dataclass
class PairwisePreferenceGP:
    lengthscale: float = 0.75
    prior_var: float = 1.0
    choice_scale: float = 0.12
    jitter: float = 1e-6
    X_train: np.ndarray | None = None
    f_map: np.ndarray | None = None
    alpha: np.ndarray | None = None

    def _kernel(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        xa = np.atleast_2d(np.asarray(xa, dtype=float))
        xb = np.atleast_2d(np.asarray(xb, dtype=float))
        d2 = np.sum((xa[:, None, :] - xb[None, :, :]) ** 2, axis=-1)
        return self.prior_var * np.exp(-0.5 * d2 / self.lengthscale**2)

    def fit(self, X: np.ndarray, comparisons: np.ndarray) -> "PairwisePreferenceGP":
        X = np.asarray(X, dtype=float)
        comparisons = np.asarray(comparisons, dtype=int)
        if X.ndim != 2 or X.shape[1] != Z_DIM:
            raise ValueError(f"X must have shape (n, {Z_DIM})")
        if comparisons.ndim != 2 or comparisons.shape[1] != 2:
            raise ValueError("comparisons must contain winner, loser index pairs")
        if len(X) == 0:
            raise ValueError("at least one point is required")

        K = self._kernel(X, X) + self.jitter * np.eye(len(X))
        K_inv = np.linalg.inv(K)
        scale = np.sqrt(2.0) * self.choice_scale

        def objective(f: np.ndarray) -> float:
            prior = 0.5 * float(f @ K_inv @ f)
            margin = (f[comparisons[:, 0]] - f[comparisons[:, 1]]) / scale
            ll = np.sum(np.log(np.clip(ndtr(margin), 1e-12, 1.0)))
            return prior - float(ll)

        res = minimize(
            objective,
            np.zeros(len(X)),
            method="L-BFGS-B",
            options={"maxiter": 1000, "maxfun": 20000},
        )
        if not res.success and not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"preference GP fit failed: {res.message}")
        self.X_train = X
        self.f_map = np.asarray(res.x, dtype=float)
        self.alpha = K_inv @ self.f_map
        return self

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.alpha is None:
            raise RuntimeError("fit the model before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return self._kernel(X, self.X_train) @ self.alpha
