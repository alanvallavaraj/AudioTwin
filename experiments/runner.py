"""Hydra experiment runner for AudioTwin oracle-regret experiments."""
from __future__ import annotations

import json
import subprocess
import zlib
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dsp.params import Z_DIM
from oracle import CONTEXTS, FAMILIES, POST_HOC_FAMILIES, find_optimum, observe_audiogram, sample_population, utility
from oracle.preference import OracleOptimum
from twin import run_active_preference_twin, run_local_prior_preference_twin, run_prior_preference_twin, run_random_preference_twin, sample_feasible
from twin.active import run_laplace_twin
from twin.optimizer import feasible_anchor, project_from_anchor


FINAL_SEED = 777000111   # RESERVED (docs/protocol.md): refuse to run unless allow_final=true
LAPLACE_METHODS = {"audiotwin_qeubo": "qeubo", "audiotwin_bald": "bald", "audiotwin_gprandom": "random"}
DEFAULT_BUDGETS = (0, 5, 10, 20, 30, 50)


def stable_seed(base: int, *keys) -> int:
    """Seed derived from stable keys only, so a cell's random streams never depend on which other
    methods/contexts are in the config or on iteration order."""
    return (int(base) + zlib.crc32("|".join(map(str, keys)).encode())) % (2**31 - 1)


STATIC_Z = {
    "prescription": np.zeros(Z_DIM),
    "prescription_linear": np.r_[0.0, 0.0, -1.0, -1.0, -1.0, 0.0],
    "prescription_slow": np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _cfg_json(cfg: DictConfig) -> str:
    return json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)


def _method_setting(method: str, listener, ctx, family: str, a_hat: np.ndarray, seed: int, cfg: DictConfig):
    if method in STATIC_Z:
        return STATIC_Z[method].copy(), utility(STATIC_Z[method], listener, ctx, family, a_hat)
    if method == "random_feasible":
        rng = np.random.default_rng(seed)
        z = sample_feasible(a_hat, rng, 1)[0]
        return z, utility(z, listener, ctx, family, a_hat)
    if method == "audiotwin_random":
        run = run_random_preference_twin(
            listener, ctx, family, a_hat,
            n_queries=int(cfg.n_queries), seed=seed, n_candidates=int(cfg.n_candidates),
        )
        return run.z_rec, run.u_rec
    if method == "audiotwin_local_prior":
        run = run_local_prior_preference_twin(
            listener, ctx, family, a_hat,
            n_queries=int(cfg.n_queries), seed=seed, n_candidates=int(cfg.n_candidates),
        )
        return run.z_rec, run.u_rec
    if method == "audiotwin_prior":
        run = run_prior_preference_twin(
            listener, ctx, family, a_hat,
            n_queries=int(cfg.n_queries), seed=seed, n_candidates=int(cfg.n_candidates),
        )
        return run.z_rec, run.u_rec
    if method == "audiotwin_active":
        run = run_active_preference_twin(
            listener, ctx, family, a_hat,
            n_queries=int(cfg.n_queries), seed=seed, n_candidates=int(cfg.n_candidates),
        )
        return run.z_rec, run.u_rec
    if method == "audiotwin_exploit":
        run = run_active_preference_twin(
            listener, ctx, family, a_hat,
            n_queries=int(cfg.n_queries), seed=seed, n_candidates=int(cfg.n_candidates),
            n_initial=3, kappa=0.05,
        )
        return run.z_rec, run.u_rec
    raise ValueError(f"unknown method: {method}")


def _optimum_for(listener, ctx, family: str, a_hat: np.ndarray, seed: int, cfg: DictConfig) -> OracleOptimum | None:
    if not bool(cfg.compute_optimum):
        return None
    return find_optimum(
        listener, ctx, family, a_hat, seed=seed,
        n_random=int(cfg.optimum.n_random), maxiter=int(cfg.optimum.maxiter),
        popsize=int(cfg.optimum.popsize),
    )


def method_outputs(method: str, listener, ctx, family: str, a_hat: np.ndarray, seed: int, cfg: DictConfig):
    """Return [(n_queries, z, projected)] for one method in one cell.

    Static baselines are projected into the SAME safe set AudioTwin must respect (`projected` records
    whether that changed them). Laplace-GP methods return one entry per query budget from a single run."""
    if method in STATIC_Z:
        anchor = feasible_anchor(a_hat)
        z = project_from_anchor(STATIC_Z[method], anchor, a_hat)
        return [(0, z, not np.allclose(z, STATIC_Z[method]))]
    if method in LAPLACE_METHODS:
        budgets = [int(b) for b in cfg.get("budgets", DEFAULT_BUDGETS)]
        run = run_laplace_twin(listener, ctx, family, a_hat, n_queries=max(budgets), seed=seed,
                               acquisition=LAPLACE_METHODS[method], checkpoints=tuple(budgets))
        return [(b, run.recs[b], False) for b in sorted(set(budgets))]
    z, _ = _method_setting(method, listener, ctx, family, a_hat, seed, cfg)   # legacy variants
    return [(int(cfg.n_queries) if method.startswith("audiotwin") else 0, z, False)]


def compute_row(listener, global_listener_id: int, context_name: str, family: str, method: str,
                a_hat: np.ndarray, seed: int, cfg: DictConfig, commit: str, config_json: str,
                z: np.ndarray, n_queries: int, projected: bool, opt: OracleOptimum | None = None) -> dict:
    ctx = CONTEXTS[context_name]
    u = utility(z, listener, ctx, family, a_hat)
    u_opt = np.nan
    u_lo = np.nan
    regret = np.nan
    z_opt = np.full(Z_DIM, np.nan)
    if opt is not None:
        u_opt, u_lo, regret, z_opt = opt.u_opt, opt.u_lo, opt.regret(u), opt.z_opt

    row = {
        "git_hash": commit,
        "config_json": config_json,
        "task_id": int(cfg.task_id),
        "seed": int(seed),
        "base_seed": int(cfg.seed),
        "listener_id": int(global_listener_id),
        "context": context_name,
        "family": family,
        "post_hoc_family": bool(family in POST_HOC_FAMILIES),
        "method": method,
        "sigma_a": float(cfg.sigma_a),
        "n_queries": int(n_queries),
        "projected": bool(projected),
        "utility": float(u),
        "u_opt": float(u_opt),
        "u_lo": float(u_lo),
        "regret": float(regret),
    }
    for i, val in enumerate(np.asarray(z, dtype=float)):
        row[f"z{i + 1}"] = float(val)
    for i, val in enumerate(np.asarray(z_opt, dtype=float)):
        row[f"zopt{i + 1}"] = float(val)
    for hz, val in zip([250, 500, 1000, 2000, 4000, 6000, 8000], listener.audiogram):
        row[f"audiogram_{hz}"] = float(val)
    for hz, val in zip([250, 500, 1000, 2000, 4000, 6000, 8000], a_hat):
        row[f"a_hat_{hz}"] = float(val)
    return row


def run_task(cfg: DictConfig) -> Path:
    if int(cfg.seed) == FINAL_SEED and not bool(cfg.get("allow_final", False)):
        raise RuntimeError("seed 777000111 is the RESERVED final-evaluation seed (docs/protocol.md); "
                           "freeze and tag the protocol first, then pass allow_final=true")
    out_dir = Path(str(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"task_{int(cfg.task_id):05d}.parquet"
    if out_path.exists() and not bool(cfg.overwrite):
        return out_path

    base_seed = int(cfg.seed) + 1009 * int(cfg.task_id)
    rng = np.random.default_rng(base_seed)
    listeners = sample_population(int(cfg.n_listeners), seed=base_seed)
    commit = git_hash()
    config_json = _cfg_json(cfg)
    rows = []

    for local_id, listener in enumerate(listeners):
        global_listener_id = int(cfg.task_id) * int(cfg.n_listeners) + local_id
        a_hat = observe_audiogram(listener, float(cfg.sigma_a), rng)
        for context_name in list(cfg.contexts):
            if context_name not in CONTEXTS:
                raise ValueError(f"unknown context: {context_name}")
            ctx = CONTEXTS[context_name]
            for family in list(cfg.families):
                if family not in FAMILIES:
                    raise ValueError(f"unknown family: {family}")
                opt = _optimum_for(listener, ctx, family, a_hat,
                                   stable_seed(base_seed, "opt", global_listener_id, context_name, family), cfg)
                for method in list(cfg.methods):
                    if method in LAPLACE_METHODS:   # paired: identical pool/oracle streams across acquisitions
                        method_seed = stable_seed(base_seed, "gp", global_listener_id, context_name, family)
                    else:
                        method_seed = stable_seed(base_seed, method, global_listener_id, context_name, family)
                    for n_q, z, projected in method_outputs(method, listener, ctx, family, a_hat, method_seed, cfg):
                        rows.append(compute_row(
                            listener, global_listener_id, context_name, family, method,
                            a_hat, method_seed, cfg, commit, config_json, z, n_q, projected, opt,
                        ))

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return out_path


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    path = run_task(cfg)
    print(path)


if __name__ == "__main__":
    main()
