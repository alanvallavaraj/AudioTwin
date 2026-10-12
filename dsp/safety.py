"""Safety set S(a_hat, c): feasibility checks and projection (docs/methods.md)."""
from __future__ import annotations

import numpy as np

from .params import DSPConfig, DSPSettings, Z_DIM, z_to_settings


def predicted_output_db(settings: DSPSettings, cfg: DSPConfig = DSPConfig(), input_db: float | None = None) -> np.ndarray:
    """Static-curve predicted band output (dB SPL) at a reference input level, after limiter."""
    L = cfg.ref_input_db_spl if input_db is None else input_db
    comp = -(1.0 - 1.0 / settings.cr) * max(L - cfg.comp_threshold_db_spl, 0.0)
    out = L + settings.gains_db + comp
    return np.minimum(out, settings.mpo_db_spl)


def violations(
    z: np.ndarray,
    audiogram_db_hl: np.ndarray,
    z_prev: np.ndarray | None = None,
    cfg: DSPConfig = DSPConfig(),
) -> dict:
    """Return a dict of constraint violation magnitudes (0 = satisfied)."""
    s = z_to_settings(z, audiogram_db_hl, cfg)
    out = predicted_output_db(s, cfg)
    smooth = np.abs(np.diff(s.gains_db)).max() - cfg.smooth_delta_db
    ucl = (out - cfg.ucl_db_spl).max()
    rate = 0.0
    if z_prev is not None:
        rate = np.abs(np.asarray(z) - np.asarray(z_prev)).max() - cfg.change_rate_delta
    return {"smoothness": max(smooth, 0.0), "ucl": max(ucl, 0.0), "change_rate": max(rate, 0.0)}


def is_feasible(z, audiogram_db_hl, z_prev=None, cfg: DSPConfig = DSPConfig()) -> bool:
    z = np.asarray(z, dtype=float)
    if z.shape != (Z_DIM,) or np.any(np.abs(z) > 1.0 + 1e-12):
        return False
    return all(v <= 1e-9 for v in violations(z, audiogram_db_hl, z_prev, cfg).values())


def project_to_S(z, audiogram_db_hl, z_prev=None, cfg: DSPConfig = DSPConfig(), iters: int = 40) -> np.ndarray:
    """Shrink z toward the anchor (z_prev, else the prescription z=0) until feasible.

    Assumes the anchor itself is feasible; raises if not.
    """
    z = np.clip(np.asarray(z, dtype=float), -1.0, 1.0)
    anchor = np.zeros(Z_DIM) if z_prev is None else np.asarray(z_prev, dtype=float)
    if not is_feasible(anchor, audiogram_db_hl, None, cfg):
        raise ValueError("anchor point is not feasible; check DSPConfig / audiogram")
    if is_feasible(z, audiogram_db_hl, z_prev, cfg):
        return z
    lo, hi = 0.0, 1.0  # fraction of the way from anchor to z
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = anchor + mid * (z - anchor)
        if is_feasible(cand, audiogram_db_hl, z_prev, cfg):
            lo = mid
        else:
            hi = mid
    return anchor + lo * (z - anchor)
