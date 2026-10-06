import numpy as np

from dsp import Z_DIM
from dsp.safety import is_feasible
from oracle import CONTEXTS, observe_audiogram, sample_population, utility
from twin import PairwisePreferenceGP, run_active_preference_twin, run_random_preference_twin, sample_feasible


def test_pairwise_gp_ranks_observed_winner_higher_and_has_variance():
    X = np.zeros((4, Z_DIM))
    X[:, 0] = [-0.8, -0.2, 0.2, 0.8]
    comparisons = np.array([[3, 0], [3, 1], [2, 0], [2, 1]])
    gp = PairwisePreferenceGP(lengthscale=1.0).fit(X, comparisons)
    pred = gp.predict_mean(X)
    var = gp.predict_variance(X)
    assert pred[3] > pred[1]
    assert pred[2] > pred[0]
    assert np.all(var >= 0.0)
    assert np.max(var) < gp.prior_var


def test_sample_feasible_returns_safe_points():
    listener = sample_population(1, seed=30)[0]
    rng = np.random.default_rng(31)
    X = sample_feasible(listener.audiogram, rng, 16)
    assert X.shape == (16, Z_DIM)
    assert all(is_feasible(z, listener.audiogram) for z in X)


def test_sample_feasible_handles_unsafe_prescription_anchor():
    listener = sample_population(1, seed=20261006 + 1009 * 11)[0]
    rng = np.random.default_rng(20261006 + 1009 * 11)
    a_hat = observe_audiogram(listener, 5.0, rng)
    assert not is_feasible(np.zeros(Z_DIM), a_hat)
    X = sample_feasible(a_hat, np.random.default_rng(123), 8)
    assert X.shape == (8, Z_DIM)
    assert all(is_feasible(z, a_hat) for z in X)


def test_random_preference_twin_run_is_reproducible_and_safe():
    listener = sample_population(1, seed=32)[0]
    ctx = CONTEXTS["noisy_conversation"]
    run1 = run_random_preference_twin(listener, ctx, "balanced", listener.audiogram, n_queries=6, seed=33)
    run2 = run_random_preference_twin(listener, ctx, "balanced", listener.audiogram, n_queries=6, seed=33)
    assert run1.comparisons.shape == (6, 2)
    assert np.allclose(run1.z_rec, run2.z_rec)
    assert is_feasible(run1.z_rec, listener.audiogram)
    assert np.isfinite(run1.u_rec)
    assert run1.u_rec == utility(run1.z_rec, listener, ctx, "balanced", listener.audiogram)


def test_active_preference_twin_run_is_reproducible_and_safe():
    listener = sample_population(1, seed=42)[0]
    ctx = CONTEXTS["noisy_conversation"]
    run1 = run_active_preference_twin(listener, ctx, "balanced", listener.audiogram, n_queries=8, seed=43, n_candidates=64)
    run2 = run_active_preference_twin(listener, ctx, "balanced", listener.audiogram, n_queries=8, seed=43, n_candidates=64)
    assert run1.comparisons.shape == (8, 2)
    assert np.allclose(run1.z_rec, run2.z_rec)
    assert is_feasible(run1.z_rec, listener.audiogram)
    assert np.isfinite(run1.u_rec)


def test_random_preference_twin_handles_task_21_seed():
    seed = 20261006 + 1009 * 21
    listener = sample_population(1, seed=seed)[0]
    rng = np.random.default_rng(seed)
    a_hat = observe_audiogram(listener, 5.0, rng)
    ctx = CONTEXTS["noisy_conversation"]
    run = run_random_preference_twin(listener, ctx, "balanced", a_hat, n_queries=12, seed=seed, n_candidates=64)
    assert is_feasible(run.z_rec, a_hat)
    assert np.isfinite(run.u_rec)
