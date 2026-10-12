"""Noisy pairwise oracle responses and cached optimum / normalisation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from dsp.params import DSPConfig, Z_DIM
from dsp.safety import violations
from .contexts import Context
from .listener import Listener
from .utility import utility


def p_prefer_a(u_a: float, u_b: float, sigma_n: float, lapse: float) -> float:
    """P(A>B) = lam/2 + (1-lam) * Phi((uA-uB)/(sqrt2*sigma_n))."""
    x = (u_a - u_b) / (math.sqrt(2.0) * sigma_n)
    phi = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return lapse / 2.0 + (1.0 - lapse) * phi


def query(z_a, z_b, listener: Listener, ctx: Context, family: str, a_dsp, rng: np.random.Generator,
          cfg: DSPConfig = DSPConfig()) -> int:
    """Return 1 if the simulated listener chooses A, else 0."""
    ua = utility(z_a, listener, ctx, family, a_dsp, cfg)
    ub = utility(z_b, listener, ctx, family, a_dsp, cfg)
    return int(rng.random() < p_prefer_a(ua, ub, listener.sigma_n, listener.lapse))


@dataclass
class OracleOptimum:
    z_opt: np.ndarray
    u_opt: float
    u_lo: float       # 5th percentile of utility over random feasible z (normalisation floor)

    def regret(self, u: float) -> float:
        """Normalised simple regret: 0 = optimal, 1 ~ a poor random feasible setting."""
        return float(max(self.u_opt - u, 0.0) / (self.u_opt - self.u_lo + 1e-9))


def _penalised(z, listener, ctx, family, a_dsp, cfg):
    v = sum(violations(z, a_dsp, None, cfg).values())
    return -utility(z, listener, ctx, family, a_dsp, cfg) + 1e3 * v


def find_optimum(listener: Listener, ctx: Context, family: str, a_dsp, seed: int = 0,
                 cfg: DSPConfig = DSPConfig(), n_random: int = 400,
                 maxiter: int = 60, popsize: int = 12) -> OracleOptimum:
    """Optimum of u* within the safety set S(a_dsp); expensive, so cache the result."""
    res = differential_evolution(
        _penalised, bounds=[(-1.0, 1.0)] * Z_DIM, args=(listener, ctx, family, a_dsp, cfg),
        seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
    )
    z_opt = np.clip(res.x, -1.0, 1.0)
    rng = np.random.default_rng(seed + 1)
    us = []
    while len(us) < n_random:
        z = rng.uniform(-1, 1, size=Z_DIM)
        if all(v <= 1e-9 for v in violations(z, a_dsp, None, cfg).values()):
            us.append(utility(z, listener, ctx, family, a_dsp, cfg))
    u_opt = max(utility(z_opt, listener, ctx, family, a_dsp, cfg), max(us))
    return OracleOptimum(z_opt=z_opt, u_opt=float(u_opt), u_lo=float(np.percentile(us, 5)))
