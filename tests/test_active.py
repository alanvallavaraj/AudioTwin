import numpy as np
import pytest

from dsp import DSPConfig, Z_DIM, is_feasible
from oracle import CONTEXTS, observe_audiogram, sample_population
from twin.active import ACQUISITIONS, run_laplace_twin
from twin.optimizer import feasible_anchor

CFG = DSPConfig()


@pytest.fixture(scope="module")
def setting():
    L = sample_population(1, seed=41)[0]
    a_hat = observe_audiogram(L, 5.0, np.random.default_rng(0))
    return L, CONTEXTS["noisy_conversation"], a_hat


@pytest.mark.parametrize("acq", ACQUISITIONS)
def test_run_is_feasible_deterministic_and_checkpointed(setting, acq):
    L, ctx, a_hat = setting
    kw = dict(n_queries=8, seed=5, acquisition=acq, checkpoints=(0, 4, 8))
    r1 = run_laplace_twin(L, ctx, "balanced", a_hat, **kw)
    r2 = run_laplace_twin(L, ctx, "balanced", a_hat, **kw)
    assert sorted(r1.recs) == [0, 4, 8]
    assert len(r1.comparisons) == 8
    for z in list(r1.X) + list(r1.recs.values()):
        assert is_feasible(z, a_hat, None, CFG) and z.shape == (Z_DIM,)
    assert np.array_equal(r1.X, r2.X) and np.array_equal(r1.comparisons, r2.comparisons)
    for k in r1.recs:
        assert np.array_equal(r1.recs[k], r2.recs[k])
    # zero-query recommendation is the prescription anchor (the GP prior mean peaks there)
    assert np.allclose(r1.recs[0], feasible_anchor(a_hat, CFG))


def test_acquisitions_pick_different_queries_but_share_the_pools(setting):
    L, ctx, a_hat = setting
    runs = {a: run_laplace_twin(L, ctx, "balanced", a_hat, n_queries=6, seed=9, acquisition=a, checkpoints=(6,))
            for a in ACQUISITIONS}
    assert not np.array_equal(runs["qeubo"].X, runs["random"].X)
    assert not np.array_equal(runs["qeubo"].X, runs["bald"].X)


def test_rejects_unknown_acquisition(setting):
    L, ctx, a_hat = setting
    with pytest.raises(ValueError):
        run_laplace_twin(L, ctx, "balanced", a_hat, n_queries=2, seed=0, acquisition="nope")
