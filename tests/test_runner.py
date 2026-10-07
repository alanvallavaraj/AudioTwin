import pandas as pd
from omegaconf import OmegaConf

from experiments.runner import run_task


def _cfg(tmp_path):
    return OmegaConf.create({
        "seed": 123,
        "task_id": 0,
        "out_dir": str(tmp_path),
        "overwrite": False,
        "n_listeners": 1,
        "sigma_a": 5.0,
        "contexts": ["noisy_conversation"],
        "families": ["balanced"],
        "methods": ["prescription", "prescription_linear", "audiotwin_random", "audiotwin_prior", "audiotwin_local_prior"],
        "n_queries": 3,
        "n_candidates": 32,
        "compute_optimum": False,
        "optimum": {"n_random": 20, "maxiter": 3, "popsize": 4},
    })


def test_runner_writes_idempotent_parquet(tmp_path):
    cfg = _cfg(tmp_path)
    path = run_task(cfg)
    assert path.exists()
    df = pd.read_parquet(path)
    assert set(df["method"]) == {"prescription", "prescription_linear", "audiotwin_random", "audiotwin_prior", "audiotwin_local_prior"}
    assert len(df) == 5
    assert {"git_hash", "config_json", "utility", "regret", "z1", "z5"}.issubset(df.columns)
    assert df.loc[df["method"] == "audiotwin_random", "n_queries"].item() == 3
    assert df.loc[df["method"] == "audiotwin_prior", "n_queries"].item() == 3
    assert df.loc[df["method"] == "audiotwin_local_prior", "n_queries"].item() == 3
    assert run_task(cfg) == path
