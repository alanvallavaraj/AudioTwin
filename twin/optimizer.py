"""Preference-based AudioTwin optimiser loop."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsp.params import DSPConfig, Z_DIM
from dsp.safety import is_feasible
from oracle.contexts import Context
from oracle.listener import Listener
from oracle.preference import query
from oracle.utility import utility
from .preference_gp import PairwisePreferenceGP


@dataclass
class PreferenceRun:
    X: np.ndarray
    comparisons: np.ndarray
    z_rec: np.ndarray
    u_rec: float
    model: PairwisePreferenceGP | None


def feasible_anchor(a_dsp: np.ndarray, cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    """Return a safe anchor even when the prescription z=0 violates safety limits."""
    anchors = [
        np.zeros(Z_DIM),
        np.array([-1.0, 0.0, -1.0, -1.0, -1.0]),
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0, -1.0, -1.0]),
    ]
    for anchor in anchors:
        if is_feasible(anchor, a_dsp, None, cfg):
            return anchor.copy()
    raise ValueError("no feasible low-gain anchor found; check DSPConfig / audiogram")


def project_from_anchor(z: np.ndarray, anchor: np.ndarray, a_dsp: np.ndarray,
                        cfg: DSPConfig = DSPConfig(), iters: int = 40) -> np.ndarray:
    """Shrink z toward a known-safe anchor without imposing a change-rate constraint."""
    z = np.clip(np.asarray(z, dtype=float), -1.0, 1.0)
    anchor = np.asarray(anchor, dtype=float)
    if not is_feasible(anchor, a_dsp, None, cfg):
        raise ValueError("anchor point is not feasible")
    if is_feasible(z, a_dsp, None, cfg):
        return z
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = anchor + mid * (z - anchor)
        if is_feasible(cand, a_dsp, None, cfg):
            lo = mid
        else:
            hi = mid
    return anchor + lo * (z - anchor)


def sample_feasible(a_dsp: np.ndarray, rng: np.random.Generator, n: int,
                    cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    """Sample feasible settings by projecting random draws into S(a_dsp)."""
    anchor = feasible_anchor(a_dsp, cfg)
    zs = []
    seen = set()
    attempts = 0
    while len(zs) < n and attempts < 50 * n:
        attempts += 1
        z = project_from_anchor(rng.uniform(-1.0, 1.0, Z_DIM), anchor, a_dsp, cfg)
        key = tuple(np.round(z, 8))
        if key not in seen and is_feasible(z, a_dsp, None, cfg):
            seen.add(key)
            zs.append(z)
    if len(zs) < n:
        raise RuntimeError(f"only sampled {len(zs)} feasible settings out of {n}")
    return np.asarray(zs)


def recommend_from_model(model: PairwisePreferenceGP | None, a_dsp: np.ndarray,
                         rng: np.random.Generator, n_candidates: int = 512,
                         cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    candidates = sample_feasible(a_dsp, rng, n_candidates, cfg)
    if model is None:
        return feasible_anchor(a_dsp, cfg)
    return candidates[int(np.argmax(model.predict_mean(candidates)))]


def run_random_preference_twin(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                               n_queries: int, seed: int, n_candidates: int = 512,
                               cfg: DSPConfig = DSPConfig()) -> PreferenceRun:
    """Run a first AudioTwin variant with random safe pair queries.

    This is the Stage-4 working baseline. Later stages can replace pair selection with
    qEUBO/BALD while keeping the same fitted model and result contract.
    """
    rng = np.random.default_rng(seed)
    X = sample_feasible(a_dsp, rng, max(2 * n_queries, 4), cfg)
    comparisons = []
    for i in range(n_queries):
        ia, ib = 2 * i, 2 * i + 1
        chose_a = query(X[ia], X[ib], listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))

    comp = np.asarray(comparisons, dtype=int)
    model = PairwisePreferenceGP().fit(X, comp)
    z_rec = recommend_from_model(model, a_dsp, rng, n_candidates=n_candidates, cfg=cfg)
    u_rec = utility(z_rec, listener, ctx, family, a_dsp, cfg)
    return PreferenceRun(X=X, comparisons=comp, z_rec=z_rec, u_rec=float(u_rec), model=model)
