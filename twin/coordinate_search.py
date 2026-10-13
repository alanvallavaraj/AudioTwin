"""Simple development-only coordinate-search preference baseline.

This comparator was added after the frozen final evaluation as a post-final exploratory baseline. It is not part
of the historical primary analysis. One pairwise query compares the current incumbent with a
single projected coordinate perturbation. Coordinates and directions are visited deterministically;
the step size halves after each complete +/- sweep over all six latent dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsp.params import DSPConfig, Z_DIM
from oracle.contexts import Context
from oracle.listener import Listener
from oracle.preference import query
from .optimizer import feasible_anchor, project_from_anchor


@dataclass
class CoordinateSearchRun:
    recs: dict[int, np.ndarray]
    comparisons: int
    queried_pairs: list[tuple[np.ndarray, np.ndarray]]


def run_coordinate_search(listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
                          n_queries: int, seed: int, checkpoints=(0, 5, 10, 20, 30, 50),
                          initial_step: float = 0.5, cfg: DSPConfig = DSPConfig()) -> CoordinateSearchRun:
    rng_oracle = np.random.default_rng(seed + 7_000_003)
    anchor = feasible_anchor(a_dsp, cfg)
    incumbent = project_from_anchor(np.zeros(Z_DIM), anchor, a_dsp, cfg)
    ckpts = sorted({int(c) for c in checkpoints if 0 <= c <= n_queries} | {n_queries})
    recs: dict[int, np.ndarray] = {0: incumbent.copy()} if 0 in ckpts else {}
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for t in range(n_queries):
        sweep = t // (2 * Z_DIM)
        pos = t % (2 * Z_DIM)
        dim = pos // 2
        sign = 1.0 if pos % 2 == 0 else -1.0
        step = float(initial_step * (0.5 ** sweep))

        proposal = incumbent.copy()
        proposal[dim] += sign * step
        proposal = project_from_anchor(proposal, anchor, a_dsp, cfg)
        if np.allclose(proposal, incumbent):
            proposal = incumbent.copy()
            proposal[dim] -= sign * step
            proposal = project_from_anchor(proposal, anchor, a_dsp, cfg)

        pairs.append((incumbent.copy(), proposal.copy()))
        if not np.allclose(proposal, incumbent):
            chose_incumbent = query(incumbent, proposal, listener, ctx, family, a_dsp, rng_oracle, cfg)
            if not chose_incumbent:
                incumbent = proposal

        q = t + 1
        if q in ckpts:
            recs[q] = incumbent.copy()

    return CoordinateSearchRun(recs=recs, comparisons=n_queries, queried_pairs=pairs)
