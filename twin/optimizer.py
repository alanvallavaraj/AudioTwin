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
        np.array([-1.0, 0.0, -1.0, -1.0, -1.0, 0.0]),
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
        np.array([-1.0, 1.0, -1.0, -1.0, -1.0, 0.0]),
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


def prior_candidates(a_dsp: np.ndarray, cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    """Return feasible clinically meaningful anchor settings for warm starts."""
    anchors = [
        np.zeros(Z_DIM),
        np.r_[0.0, 0.0, -1.0, -1.0, -1.0, 0.0],
        np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    safe_anchor = feasible_anchor(a_dsp, cfg)
    zs = []
    seen = set()
    for anchor in anchors:
        z = project_from_anchor(anchor, safe_anchor, a_dsp, cfg)
        key = tuple(np.round(z, 8))
        if key not in seen and is_feasible(z, a_dsp, None, cfg):
            seen.add(key)
            zs.append(z)
    return np.asarray(zs)


def prior_augmented_pool(a_dsp: np.ndarray, rng: np.random.Generator, n: int,
                         cfg: DSPConfig = DSPConfig(), jitter_scale: float = 0.25) -> np.ndarray:
    """Build a feasible candidate pool around prescription anchors plus random coverage."""
    priors = prior_candidates(a_dsp, cfg)
    safe_anchor = feasible_anchor(a_dsp, cfg)
    zs = [z.copy() for z in priors]
    seen = {tuple(np.round(z, 8)) for z in zs}
    attempts = 0
    while len(zs) < n and attempts < 50 * n:
        attempts += 1
        base = priors[int(rng.integers(len(priors)))]
        cand = project_from_anchor(base + rng.normal(0.0, jitter_scale, Z_DIM), safe_anchor, a_dsp, cfg)
        key = tuple(np.round(cand, 8))
        if key not in seen and is_feasible(cand, a_dsp, None, cfg):
            seen.add(key)
            zs.append(cand)
    if len(zs) < n:
        random_fill = sample_feasible(a_dsp, rng, n - len(zs), cfg)
        zs.extend(random_fill)
    return np.asarray(zs[:n])


def local_prior_pool(a_dsp: np.ndarray, rng: np.random.Generator, n: int,
                     cfg: DSPConfig = DSPConfig(), jitter_scale: float = 0.12) -> np.ndarray:
    """Build a local feasible pool around the strongest clinical prior anchors."""
    priors = prior_candidates(a_dsp, cfg)
    safe_anchor = feasible_anchor(a_dsp, cfg)
    bases = priors[1:] if len(priors) > 1 else priors
    seeds = [z.copy() for z in priors]

    linear = project_from_anchor(np.r_[0.0, 0.0, -1.0, -1.0, -1.0, 0.0], safe_anchor, a_dsp, cfg)
    for z3 in [-1.0, -0.85, -0.7, -0.5]:
        for z6 in [0.0, 0.5, 1.0]:
            seeds.append(project_from_anchor(np.r_[0.0, 0.0, z3, -1.0, -1.0, z6], safe_anchor, a_dsp, cfg))
    slow = project_from_anchor(np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], safe_anchor, a_dsp, cfg)
    for z3 in [-0.8, -0.5, -0.2, 0.0]:
        for z4z5 in [-1.0, -0.5, 0.0]:
            seeds.append(project_from_anchor(np.r_[0.0, 0.0, z3, z4z5, z4z5, 1.0], safe_anchor, a_dsp, cfg))

    zs = []
    seen = set()
    for z in seeds:
        key = tuple(np.round(z, 8))
        if key not in seen and is_feasible(z, a_dsp, None, cfg):
            seen.add(key)
            zs.append(z)

    attempts = 0
    while len(zs) < n and attempts < 80 * n:
        attempts += 1
        base = bases[int(rng.integers(len(bases)))]
        scale = jitter_scale * (0.5 + rng.random())
        cand = project_from_anchor(base + rng.normal(0.0, scale, Z_DIM), safe_anchor, a_dsp, cfg)
        key = tuple(np.round(cand, 8))
        if key not in seen and is_feasible(cand, a_dsp, None, cfg):
            seen.add(key)
            zs.append(cand)
    if len(zs) < n:
        zs.extend(sample_feasible(a_dsp, rng, n - len(zs), cfg))
    return np.asarray(zs[:n])


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



def run_prior_preference_twin(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                              n_queries: int, seed: int, n_candidates: int = 512,
                              kappa: float = 0.25, cfg: DSPConfig = DSPConfig()) -> PreferenceRun:
    """Run AudioTwin with clinical anchors in the query and recommendation set."""
    rng = np.random.default_rng(seed)
    priors = prior_candidates(a_dsp, cfg)
    pool = prior_augmented_pool(a_dsp, rng, max(n_candidates, 2 * n_queries + len(priors) + 8), cfg)
    X: list[np.ndarray] = []
    index: dict[tuple[float, ...], int] = {}
    comparisons: list[tuple[int, int]] = []
    used_pairs: set[tuple[int, int]] = set()

    prior_pairs = [(i, j) for i in range(len(priors)) for j in range(i + 1, len(priors))]
    for ia_prior, ib_prior in prior_pairs[:n_queries]:
        za = priors[ia_prior]
        zb = priors[ib_prior]
        ia = _add_point(za, X, index)
        ib = _add_point(zb, X, index)
        chose_a = query(za, zb, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))
        used_pairs.add(tuple(sorted((ia, ib))))

    if not comparisons:
        za, zb = pool[0], pool[1]
        ia = _add_point(za, X, index)
        ib = _add_point(zb, X, index)
        chose_a = query(za, zb, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))
        used_pairs.add(tuple(sorted((ia, ib))))

    model = _fit_model(X, comparisons)
    while len(comparisons) < n_queries:
        mean = model.predict_mean(pool)
        std = np.sqrt(model.predict_variance(pool))
        best_pool_idx = int(np.argmax(mean))
        score = mean + kappa * std
        score[best_pool_idx] = -np.inf
        selected = None
        for cand_idx in np.argsort(score)[::-1]:
            za = pool[best_pool_idx]
            zb = pool[int(cand_idx)]
            ia = _add_point(za, X, index)
            ib = _add_point(zb, X, index)
            pair = tuple(sorted((ia, ib)))
            if ia != ib and pair not in used_pairs:
                selected = za, zb, ia, ib, pair
                break
        if selected is None:
            ia_pool, ib_pool = rng.choice(len(pool), 2, replace=False)
            za = pool[int(ia_pool)]
            zb = pool[int(ib_pool)]
            ia = _add_point(za, X, index)
            ib = _add_point(zb, X, index)
            pair = tuple(sorted((ia, ib)))
            if ia == ib or pair in used_pairs:
                continue
        else:
            za, zb, ia, ib, pair = selected
        chose_a = query(za, zb, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((ia, ib) if chose_a else (ib, ia))
        used_pairs.add(pair)
        model = _fit_model(X, comparisons)

    rec_pool = prior_augmented_pool(a_dsp, rng, n_candidates, cfg)
    z_rec = rec_pool[int(np.argmax(model.predict_mean(rec_pool)))]
    u_rec = utility(z_rec, listener, ctx, family, a_dsp, cfg)
    return PreferenceRun(X=np.asarray(X), comparisons=np.asarray(comparisons, dtype=int),
                         z_rec=z_rec, u_rec=float(u_rec), model=model)


def run_local_prior_preference_twin(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                                    n_queries: int, seed: int, n_candidates: int = 512,
                                    kappa: float = 0.05, recommend_margin: float = 0.10,
                                    cfg: DSPConfig = DSPConfig()) -> PreferenceRun:
    """Run a conservative local search around the linear/slow clinical priors."""
    rng = np.random.default_rng(seed)
    safe_anchor = feasible_anchor(a_dsp, cfg)
    incumbent = project_from_anchor(np.r_[0.0, 0.0, -1.0, -1.0, -1.0, 0.0], safe_anchor, a_dsp, cfg)
    priors = prior_candidates(a_dsp, cfg)
    pool = local_prior_pool(a_dsp, rng, max(n_candidates, 2 * n_queries + len(priors) + 16), cfg)

    X: list[np.ndarray] = []
    index: dict[tuple[float, ...], int] = {}
    comparisons: list[tuple[int, int]] = []
    used_pairs: set[tuple[int, int]] = set()
    inc_idx = _add_point(incumbent, X, index)

    warm_starts = [z for z in priors if not np.allclose(z, incumbent)]
    warm_starts.extend(pool[:max(0, n_queries - len(warm_starts))])
    for challenger in warm_starts:
        if len(comparisons) >= n_queries:
            break
        ch_idx = _add_point(challenger, X, index)
        pair = tuple(sorted((inc_idx, ch_idx)))
        if inc_idx == ch_idx or pair in used_pairs:
            continue
        chose_incumbent = query(incumbent, challenger, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((inc_idx, ch_idx) if chose_incumbent else (ch_idx, inc_idx))
        used_pairs.add(pair)

    if not comparisons:
        challenger = pool[0]
        ch_idx = _add_point(challenger, X, index)
        chose_incumbent = query(incumbent, challenger, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((inc_idx, ch_idx) if chose_incumbent else (ch_idx, inc_idx))
        used_pairs.add(tuple(sorted((inc_idx, ch_idx))))

    model = _fit_model(X, comparisons)
    while len(comparisons) < n_queries:
        mean = model.predict_mean(pool)
        std = np.sqrt(model.predict_variance(pool))
        score = mean + kappa * std
        selected = None
        for cand_idx in np.argsort(score)[::-1]:
            challenger = pool[int(cand_idx)]
            ch_idx = _add_point(challenger, X, index)
            pair = tuple(sorted((inc_idx, ch_idx)))
            if inc_idx != ch_idx and pair not in used_pairs:
                selected = challenger, ch_idx, pair
                break
        if selected is None:
            break
        challenger, ch_idx, pair = selected
        chose_incumbent = query(incumbent, challenger, listener, ctx, family, a_dsp, rng, cfg)
        comparisons.append((inc_idx, ch_idx) if chose_incumbent else (ch_idx, inc_idx))
        used_pairs.add(pair)
        model = _fit_model(X, comparisons)

    rec_pool = local_prior_pool(a_dsp, rng, n_candidates, cfg)
    rec_pool = np.vstack([incumbent, rec_pool])
    mean = model.predict_mean(rec_pool)
    best_idx = int(np.argmax(mean))
    z_rec = rec_pool[best_idx] if mean[best_idx] >= mean[0] + recommend_margin else incumbent
    u_rec = utility(z_rec, listener, ctx, family, a_dsp, cfg)
    return PreferenceRun(X=np.asarray(X), comparisons=np.asarray(comparisons, dtype=int),
                         z_rec=z_rec, u_rec=float(u_rec), model=model)

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
