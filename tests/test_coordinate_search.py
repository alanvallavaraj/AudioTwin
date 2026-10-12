import numpy as np

from dsp.safety import is_feasible
from oracle import CONTEXTS, observe_audiogram, sample_population
from twin.coordinate_search import run_coordinate_search


def test_coordinate_search_is_deterministic_safe_and_checkpointed():
    listener = sample_population(1, seed=424242)[0]
    rng = np.random.default_rng(99)
    a_hat = observe_audiogram(listener, 5.0, rng)
    kwargs = dict(listener=listener, ctx=CONTEXTS["noisy_conversation"], family="balanced",
                  a_dsp=a_hat, n_queries=20, seed=123, checkpoints=(0, 5, 10, 20))
    a = run_coordinate_search(**kwargs)
    b = run_coordinate_search(**kwargs)
    assert sorted(a.recs) == [0, 5, 10, 20]
    assert a.comparisons == 20
    assert np.allclose(a.recs[0], b.recs[0])
    for q in a.recs:
        assert np.allclose(a.recs[q], b.recs[q])
        assert is_feasible(a.recs[q], a_hat)
