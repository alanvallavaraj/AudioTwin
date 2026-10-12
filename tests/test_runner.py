import numpy as np
import pytest
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


def _gp_cfg(tmp_path, methods):
    c = _cfg(tmp_path)
    c.methods = methods
    c.budgets = [0, 2, 4]
    return c


def test_laplace_methods_emit_one_row_per_budget_with_paired_streams(tmp_path):
    methods = ["prescription", "prescription_linear", "prescription_slow",
               "audiotwin_qeubo", "audiotwin_bald", "audiotwin_gprandom"]
    df = pd.read_parquet(run_task(_gp_cfg(tmp_path, methods)))
    assert len(df) == 3 + 3 * 3
    for m in ("audiotwin_qeubo", "audiotwin_bald", "audiotwin_gprandom"):
        assert sorted(df.loc[df.method == m, "n_queries"]) == [0, 2, 4]
    assert (df.loc[df.method.str.startswith("prescription"), "n_queries"] == 0).all()
    z0 = df[(df.n_queries == 0) & df.method.str.startswith("audiotwin")][[f"z{i}" for i in range(1, 7)]].to_numpy()
    assert np.allclose(z0, z0[0])          # zero-query recommendation is acquisition-independent (same prior, same pool)
    zc = [f"z{i}" for i in range(1, 7)]
    z_rx = df[df.method == "prescription"][zc].to_numpy()[0]
    assert np.allclose(z0[0], z_rx)       # fairness: at T=0 the twin IS the (projected) prescription baseline
    assert {"projected", "post_hoc_family", "base_seed"}.issubset(df.columns)


def test_cell_results_do_not_depend_on_other_methods_in_the_config(tmp_path):
    a = pd.read_parquet(run_task(_gp_cfg(tmp_path / "a", ["audiotwin_qeubo"])))
    b = pd.read_parquet(run_task(_gp_cfg(tmp_path / "b", ["prescription", "audiotwin_bald", "audiotwin_qeubo"])))
    zc = [f"z{i}" for i in range(1, 7)]
    qa = a[a.method == "audiotwin_qeubo"].sort_values("n_queries")[zc].to_numpy()
    qb = b[b.method == "audiotwin_qeubo"].sort_values("n_queries")[zc].to_numpy()
    assert np.allclose(qa, qb)


def test_reserved_final_seed_is_refused(tmp_path):
    c = _gp_cfg(tmp_path, ["prescription"])
    c.seed = 777000111
    with pytest.raises(RuntimeError, match="RESERVED"):
        run_task(c)
