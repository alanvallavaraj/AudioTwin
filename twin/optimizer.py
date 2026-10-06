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


def _add_point(z: np.ndarray, X: list[np.ndarray], index: dict[tuple[float, ...], int]) -> int:
    key = tuple(np.round(z, 8))
    if key not in index:
        index[key] = len(X)
        X.append(np.asarray(z, dtype=float))
    return index[key]


def _fit_model(X: list[np.ndarray], comparisons: list[tuple[int, int]]) -> PairwisePreferenceGP:
    return PairwisePreferenceGP().fit(np.asarray(X), np.asarray(comparisons, dtype=int))


def run_random_preference_twin(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                               n_queries: int, seed: int, n_candidates: int = 512,
                               cfg: DSPConfig = DSPConfig()) -> PreferenceRun:
    """Run a baseline AudioTwin variant with random safe pair queries."""
    rng = np.random.default_rng(seed)
    X_arr = sample_feasible(a_dsp, rng, max(2 * n_queries, 4), cfg)
    comparisons = []
    for i in range(n_queries):
        ia, ib = 2 * i, 2 * i + 1
        chose_a = query(X_arr[ia], X_arr[ib], listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))

    comp = np.asarray(comparisons, dtype=int)
    model = PairwisePreferenceGP().fit(X_arr, comp)
    z_rec = recommend_from_model(model, a_dsp, rng, n_candidates=n_candidates, cfg=cfg)
    u_rec = utility(z_rec, listener, ctx, family, a_dsp, cfg)
    return PreferenceRun(X=X_arr, comparisons=comp, z_rec=z_rec, u_rec=float(u_rec), model=model)


def run_active_preference_twin(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                               n_queries: int, seed: int, n_candidates: int = 512,
                               n_initial: int = 4, kappa: float = 0.75,
                               cfg: DSPConfig = DSPConfig()) -> PreferenceRun:
    """Run a model-guided preference loop.

    After a few random warm-up comparisons, query the current predicted best setting
    against a challenger with high predicted utility plus uncertainty. This is a cheap
    CPU approximation to utility-of-the-best-option acquisition for the pilot stage.
    """
    rng = np.random.default_rng(seed)
    pool = sample_feasible(a_dsp, rng, max(n_candidates, 2 * n_queries + 8), cfg)
    X: list[np.ndarray] = []
    index: dict[tuple[float, ...], int] = {}
    comparisons: list[tuple[int, int]] = []
    used_pairs: set[tuple[int, int]] = set()

    warmup = min(n_queries, max(1, n_initial))
    order = rng.permutation(len(pool))
    for i in range(warmup):
        za = pool[order[2 * i]]
        zb = pool[order[2 * i + 1]]
        ia = _add_point(za, X, index)
        ib = _add_point(zb, X, index)
        chose_a = query(za, zb, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))
        used_pairs.add(tuple(sorted((ia, ib))))

    model = _fit_model(X, comparisons)
    for _ in range(warmup, n_queries):
        mean = model.predict_mean(pool)
        std = np.sqrt(model.predict_variance(pool))
        best_pool_idx = int(np.argmax(mean))
        score = mean + kappa * std
        score[best_pool_idx] = -np.inf
        for cand_idx in np.argsort(score)[::-1]:
            za = pool[best_pool_idx]
            zb = pool[int(cand_idx)]
            ia = _add_point(za, X, index)
            ib = _add_point(zb, X, index)
            pair = tuple(sorted((ia, ib)))
            if ia != ib and pair not in used_pairs:
                break
        chose_a = query(za, zb, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))
        used_pairs.add(pair)
        model = _fit_model(X, comparisons)

    rec_pool = sample_feasible(a_dsp, rng, n_candidates, cfg)
    z_rec = rec_pool[int(np.argmax(model.predict_mean(rec_pool)))]
    u_rec = utility(z_rec, listener, ctx, family, a_dsp, cfg)
    return PreferenceRun(X=np.asarray(X), comparisons=np.asarray(comparisons, dtype=int),
                         z_rec=z_rec, u_rec=float(u_rec), model=model)
