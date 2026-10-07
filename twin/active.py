"""AudioTwin active-preference loop: Laplace preference GP + qEUBO / BALD / random query selection.

All three acquisition rules share the SAME model, candidate pools, minimum pair separation and oracle
stream layout, so any difference between them is attributable to query selection alone.
No setting is hard-coded from earlier diagnostics: the only prior is the soft pull toward the
prescription (z = 0, or the nearest feasible anchor) built into the GP mean.
Candidates are always feasible (project_from_anchor), so safety holds by construction.
One run records the recommendation at several query budgets (checkpoints) to give a budget sweep.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsp.params import DSPConfig, Z_DIM
from oracle.contexts import Context
from oracle.listener import Listener
from oracle.preference import query
from .acquisition import bald_matrix, best_pair, eubo_matrix
from .laplace_gp import LaplacePreferenceGP, fit_with_selection
from .optimizer import feasible_anchor, project_from_anchor, sample_feasible

ACQUISITIONS = ("qeubo", "bald", "random")


@dataclass
class ActiveRun:
    X: np.ndarray
    comparisons: np.ndarray
    recs: dict            # query budget -> recommended z
    acquisition: str


def _local(center, rng, n, scale, anchor, a_dsp, cfg):
    return np.array([project_from_anchor(center + rng.normal(0.0, scale, Z_DIM), anchor, a_dsp, cfg)
                     for _ in range(n)])


def _random_pair(Z, rng, min_sep):
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(Z), k=1)
    ok = np.flatnonzero(dist[iu] >= min_sep)
    if len(ok) == 0:
        raise ValueError("no admissible pair in candidate pool")
    k = int(rng.choice(ok))
    return int(iu[0][k]), int(iu[1][k])


def run_laplace_twin(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                     n_queries: int, seed: int, acquisition: str = "qeubo",
                     checkpoints=(0, 5, 10, 20, 30, 50), n_uniform: int = 192, n_local: int = 64,
                     n_rec: int = 512, min_sep: float = 0.4, hyper_every: int = 5,
                     cfg: DSPConfig = DSPConfig()) -> ActiveRun:
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {ACQUISITIONS}")
    rng = np.random.default_rng(seed)                          # candidate pools, random pairs
    rng_oracle = np.random.default_rng(seed + 7_000_003)       # simulated listener responses
    anchor = feasible_anchor(a_dsp, cfg)
    ckpts = sorted({int(c) for c in checkpoints if 0 <= c <= n_queries} | {n_queries})

    X: list[np.ndarray] = []
    index: dict[tuple, int] = {}
    comps: list[tuple[int, int]] = []
    hyper: dict | None = None
    recs: dict = {}

    def add(z) -> int:
        key = tuple(np.round(z, 10))
        if key not in index:
            index[key] = len(X)
            X.append(np.asarray(z, dtype=float))
        return index[key]

    for t in range(n_queries + 1):
        # ---- model on the t comparisons so far
        if not comps:
            model = LaplacePreferenceGP()
        elif t % hyper_every == 0:
            model = fit_with_selection(np.array(X), comps)
            hyper = dict(lengthscale=model.lengthscale, signal_var=model.signal_var, noise_scale=model.noise_scale)
        else:
            model = LaplacePreferenceGP(**(hyper or {})).fit(np.array(X), comps)

        incumbent = anchor if not X else np.array(X)[int(np.argmax(model.predict_mean(np.array(X))))]

        # ---- recommendation at checkpoints
        if t in ckpts:
            pool = np.vstack([anchor[None, :], sample_feasible(a_dsp, rng, n_rec // 2, cfg),
                              _local(incumbent, rng, n_rec // 2, 0.25, anchor, a_dsp, cfg)]
                             + ([np.array(X)] if X else []))
            recs[t] = pool[int(np.argmax(model.predict_mean(pool)))].copy()
        if t == n_queries:
            break

        # ---- choose the next pair from a feasible pool
        parts = [anchor[None, :], sample_feasible(a_dsp, rng, n_uniform, cfg)]
        if X:
            parts.append(_local(incumbent, rng, n_local, 0.25, anchor, a_dsp, cfg))
        Z = np.vstack(parts)
        if acquisition == "random":
            pair = None
            for sep in (min_sep, min_sep / 2, 0.0):
                try:
                    pair = _random_pair(Z, rng, sep)
                    break
                except ValueError:
                    continue
        else:
            mu, C = model.posterior(Z)
            score = eubo_matrix(mu, C) if acquisition == "qeubo" else bald_matrix(mu, C, model.noise_scale)
            pair = None
            for sep in (min_sep, min_sep / 2, 0.0):
                try:
                    pair = best_pair(score, Z, sep)
                    break
                except ValueError:
                    continue
        if pair is None:
            raise RuntimeError("could not find an admissible query pair")
        za, zb = Z[pair[0]], Z[pair[1]]
        ia, ib = add(za), add(zb)
        chose_a = query(za, zb, listener, ctx, family, a_dsp, rng_oracle, cfg)
        comps.append((ia, ib) if chose_a else (ib, ia))

    return ActiveRun(X=np.array(X), comparisons=np.array(comps, dtype=int), recs=recs, acquisition=acquisition)
