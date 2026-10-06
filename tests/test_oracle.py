import numpy as np
import pytest

from dsp import DSPConfig, Z_DIM
from dsp.safety import violations
from oracle import (CONTEXTS, FAMILIES, VERIFIED_BISGAARD, bisgaard_to_audiogram, find_optimum,
                    observe_audiogram, p_prefer_a, pta4, query, sample_population, utility)

CFG = DSPConfig()


def test_population_reproducible_and_in_range():
    p1 = sample_population(20, seed=7)
    p2 = sample_population(20, seed=7)
    for a, b in zip(p1, p2):
        assert np.array_equal(a.audiogram, b.audiogram)
        assert 20.0 <= pta4(a.audiogram) <= 55.0
        assert np.all((a.audiogram >= 0) & (a.audiogram <= 90))
        assert a.sigma_n > 0 and 0 <= a.lapse < 0.1
        for w in a.weights.values():
            assert w.sum() == pytest.approx(1.0)
    p3 = sample_population(20, seed=8)
    assert not all(np.array_equal(a.audiogram, b.audiogram) for a, b in zip(p1, p3))


def test_bisgaard_verified_rows_and_mapping():
    assert VERIFIED_BISGAARD["N4"][0] == 55 and VERIFIED_BISGAARD["N6"][-1] == 100
    a = bisgaard_to_audiogram(VERIFIED_BISGAARD["N4"])
    assert a.shape == (7,)
    assert a[-1] == 80.0          # 8 kHz holds the 6 kHz value
    assert a[0] == 55.0 and a[2] == 55.0   # 250 Hz, 1 kHz taken directly


def test_observe_audiogram_noise():
    L = sample_population(1, seed=1)[0]
    rng = np.random.default_rng(0)
    a0 = observe_audiogram(L, 0.0, rng)
    assert np.all(np.abs(a0 - L.audiogram) <= 2.5 + 1e-9)   # only 5 dB rounding
    a1 = observe_audiogram(L, 8.0, rng)
    assert a1.shape == (7,)


def test_preference_probabilities():
    assert p_prefer_a(0.5, -0.5, 0.05, 0.02) == pytest.approx(0.99, abs=0.005)  # 1 - lapse/2
    assert p_prefer_a(0.1, 0.1, 0.1, 0.04) == pytest.approx(0.5)
    assert p_prefer_a(-0.5, 0.5, 0.05, 0.02) == pytest.approx(0.01, abs=0.005)
    L = sample_population(1, seed=2)[0]
    a = L.audiogram
    ctx = CONTEXTS["noisy_conversation"]
    zA, zB = np.zeros(Z_DIM), np.array([0.9, 0.9, 0.9, -0.9, -0.9])
    rng = np.random.default_rng(3)
    n = 2000
    emp = np.mean([query(zA, zB, L, ctx, "balanced", a, rng) for _ in range(n)])
    ua = utility(zA, L, ctx, "balanced", a)
    ub = utility(zB, L, ctx, "balanced", a)
    assert emp == pytest.approx(p_prefer_a(ua, ub, L.sigma_n, L.lapse), abs=0.04)


def test_utility_bounded_and_finite():
    rng = np.random.default_rng(0)
    for L in sample_population(5, seed=4):
        for ctx in CONTEXTS.values():
            for fam in FAMILIES:
                z = rng.uniform(-1, 1, Z_DIM)
                u = utility(z, L, ctx, fam, L.audiogram)
                assert np.isfinite(u) and -1.0 <= u <= 1.0


def test_optimum_beats_prescription_and_random_and_is_feasible():
    L = sample_population(1, seed=5)[0]
    ctx = CONTEXTS["noisy_conversation"]
    opt = find_optimum(L, ctx, "balanced", L.audiogram, seed=0)
    assert all(v <= 1e-6 for v in violations(opt.z_opt, L.audiogram, None, CFG).values())
    u0 = utility(np.zeros(Z_DIM), L, ctx, "balanced", L.audiogram)
    assert opt.u_opt >= u0 - 1e-9
    assert opt.u_opt > opt.u_lo
    assert opt.regret(opt.u_opt) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= opt.regret(opt.u_lo) <= 1.5


def test_families_disagree_about_best_setting():
    """If all oracle families chose the same z, the 'multiple oracles' robustness claim is empty."""
    L = sample_population(1, seed=6)[0]
    ctx = CONTEXTS["noisy_conversation"]
    zs = {f: find_optimum(L, ctx, f, L.audiogram, seed=0).z_opt for f in FAMILIES}
    d = [np.linalg.norm(zs[a] - zs[b]) for a in FAMILIES for b in FAMILIES if a < b]
    assert max(d) > 0.2, f"families give near-identical optima: {d}"
