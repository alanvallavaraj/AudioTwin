"""Hydra experiment runner for AudioTwin oracle-regret experiments."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dsp.params import Z_DIM
from oracle import CONTEXTS, FAMILIES, find_optimum, observe_audiogram, sample_population, utility
from oracle.preference import OracleOptimum
from twin import run_active_preference_twin, run_random_preference_twin, sample_feasible


STATIC_Z = {
    "prescription": np.zeros(Z_DIM),
    "prescription_linear": np.array([0.0, 0.0, -1.0, -1.0, -1.0]),
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
    if method == "audiotwin_active":
        run = run_active_preference_twin(
            listener, ctx, family, a_hat,
            n_queries=int(cfg.n_queries), seed=seed, n_candidates=int(cfg.n_candidates),
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


def compute_row(listener, global_listener_id: int, context_name: str, family: str, method: str,
                a_hat: np.ndarray, seed: int, cfg: DictConfig, commit: str, config_json: str,
                opt: OracleOptimum | None = None) -> dict:
    ctx = CONTEXTS[context_name]
    z, u = _method_setting(method, listener, ctx, family, a_hat, seed, cfg)
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
        "listener_id": int(global_listener_id),
        "context": context_name,
        "family": family,
        "method": method,
        "sigma_a": float(cfg.sigma_a),
        "n_queries": int(cfg.n_queries) if method.startswith("audiotwin") else 0,
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
                opt_seed = base_seed + 10_000 + 193 * global_listener_id + 29 * len(rows)
                opt = _optimum_for(listener, ctx, family, a_hat, opt_seed, cfg)
                for method in list(cfg.methods):
                    method_seed = base_seed + 17 * global_listener_id + 101 * len(rows)
                    rows.append(compute_row(
                        listener, global_listener_id, context_name, family, method,
                        a_hat, method_seed, cfg, commit, config_json, opt,
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
