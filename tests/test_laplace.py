import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import pearsonr, spearmanr

from twin.acquisition import bald_matrix, best_pair, eubo_matrix
from twin.laplace_gp import LaplacePreferenceGP, fit_with_selection

D = 6


def synth(n_pts=60, n_pairs=90, sigma=0.05, seed=0):
    """Ground truth u(z) = -||z - z*||^2 / 5 with probit comparisons (noise sigma)."""
    rng = np.random.default_rng(seed)
    zstar = rng.uniform(-0.5, 0.5, D)
    u = lambda Z: -np.sum((np.atleast_2d(Z) - zstar) ** 2, axis=1) / 5.0
    X = rng.uniform(-1, 1, (n_pts, D))
    comps = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_pts, 2, replace=False)
        p = ndtr((u(X[i])[0] - u(X[j])[0]) / (np.sqrt(2) * sigma))
        comps.append((i, j) if rng.random() < p else (j, i))
    return X, np.array(comps), u, zstar, rng


def test_map_matches_brute_force_minimisation():
    X, comps, _, _, _ = synth(n_pts=12, n_pairs=20, seed=1)
    gp = LaplacePreferenceGP(lengthscale=1.0, signal_var=0.3, noise_scale=0.1).fit(X, comps)
    K_inv, mu, s = gp.K_inv, gp.mu_X, gp._s
    w, l = comps[:, 0], comps[:, 1]
    obj = lambda f: 0.5 * (f - mu) @ K_inv @ (f - mu) - log_ndtr((f[w] - f[l]) / s).sum()
    ref = minimize(obj, np.zeros(len(X)), method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    assert obj(gp.f_map) <= obj(ref.x) + 1e-7
    assert np.max(np.abs(gp.f_map - ref.x)) < 1e-3


def test_no_data_posterior_is_the_prior():
    gp = LaplacePreferenceGP()
    Z = np.random.default_rng(0).uniform(-1, 1, (7, D))
    mean, cov = gp.posterior(Z)
    assert np.allclose(mean, gp.prior_mean(Z))
    assert np.allclose(cov, gp.kernel(Z, Z))


def test_posterior_covariance_is_valid_and_shrinks_with_data():
    X, comps, _, _, rng = synth(seed=2)
    Z = rng.uniform(-1, 1, (25, D))
    prior_var = np.diag(LaplacePreferenceGP().kernel(Z, Z))
    gp = fit_with_selection(X, comps)
    _, C = gp.posterior(Z)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-8
    assert np.all(np.diag(C) <= np.diag(gp.kernel(Z, Z)) + 1e-8)       # data never increases variance
    assert np.mean(np.diag(C)) < np.mean(np.diag(gp.kernel(Z, Z)))     # and on average reduces it
    assert prior_var.min() > 0


def test_recovers_known_utility():
    X, comps, u, zstar, rng = synth(seed=3)
    gp = fit_with_selection(X, comps)
    Z = rng.uniform(-1, 1, (400, D))
    r, _ = pearsonr(gp.predict_mean(Z), u(Z))
    assert r > 0.6, r
    # the model's best pool point should be among the better true-utility points
    best = Z[np.argmax(gp.predict_mean(Z))]
    assert u(best)[0] > np.percentile(u(Z), 80)


def test_eubo_matches_monte_carlo():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    C = A @ A.T * 0.1 + 0.05 * np.eye(5)
    mu = rng.normal(0, 0.3, 5)
    E = eubo_matrix(mu, C)
    S = rng.multivariate_normal(mu, C, size=400_000)
    for i, j in [(0, 1), (2, 4), (1, 3)]:
        mc = np.maximum(S[:, i], S[:, j]).mean()
        assert E[i, j] == pytest.approx(mc, abs=3e-3)
        assert E[i, j] == pytest.approx(E[j, i])
        assert E[i, j] >= max(mu[i], mu[j]) - 1e-12      # E[max] >= max of means (Jensen)
    assert np.isneginf(E[0, 0])


def test_bald_matches_monte_carlo_mutual_information():
    rng = np.random.default_rng(1)
    noise = 0.1
    s = np.sqrt(2) * noise
    h = lambda p: -p * np.log(np.clip(p, 1e-12, 1)) - (1 - p) * np.log(np.clip(1 - p, 1e-12, 1))
    for mu_d, theta in [(0.0, 0.1), (0.05, 0.2), (0.3, 0.15), (-0.1, 0.4)]:
        mu = np.array([mu_d, 0.0]); C = np.array([[theta**2, 0.0], [0.0, 1e-12]])
        approx = bald_matrix(mu, C, noise)[0, 1]
        f = rng.normal(mu_d, theta, 400_000)
        p = ndtr(f / s)
        mc = h(p.mean()) - h(p).mean()
        assert approx == pytest.approx(mc, abs=0.02), (mu_d, theta, approx, mc)
    # more posterior uncertainty about the difference -> more information
    lo = bald_matrix(np.zeros(2), np.diag([0.01, 1e-12]), noise)[0, 1]
    hi = bald_matrix(np.zeros(2), np.diag([0.5, 1e-12]), noise)[0, 1]
    assert hi > lo


def test_best_pair_respects_min_separation():
    Z = np.array([[0.0, 0], [0.05, 0], [1.0, 1.0], [-1.0, -1.0]])
    score = np.array([[-np.inf, 9, 1, 2], [9, -np.inf, 3, 4], [1, 3, -np.inf, 5], [2, 4, 5, -np.inf]], float)
    assert best_pair(score, Z, min_sep=0.0) == (0, 1)
    assert best_pair(score, Z, min_sep=0.3) == (2, 3)


def test_agrees_with_botorch_pairwise_gp():
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
    X, comps, u, _, rng = synth(seed=4)
    Z = rng.uniform(-1, 1, (300, D))
    mine = fit_with_selection(X, comps).predict_mean(Z)
    model = PairwiseGP(torch.tensor(X, dtype=torch.double), torch.tensor(comps, dtype=torch.long))
    fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model))
    theirs = model.posterior(torch.tensor(Z, dtype=torch.double)).mean.squeeze(-1).detach().numpy()
    truth = u(Z)
    assert spearmanr(mine, truth)[0] > 0.6 and spearmanr(theirs, truth)[0] > 0.6
    assert spearmanr(mine, theirs)[0] > 0.7
