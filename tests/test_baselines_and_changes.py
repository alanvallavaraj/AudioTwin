import numpy as np
import pytest

from baselines import (STATIC_BASELINES, fixed_eq, no_correction, prescription,
                       prescription_linear, prescription_slow)
from dsp import DSPConfig, Z_DIM, z_to_settings
from oracle import CONTEXTS, FAMILIES, POST_HOC_FAMILIES, find_optimum, sample_population, utility

CFG = DSPConfig()
SEVERE = np.array([40, 45, 55, 60, 65, 70, 75], float)
MILD = np.array([15, 15, 20, 25, 30, 35, 35], float)


def test_widened_compression_reaches_linear_even_for_severe_loss():
    z = np.zeros(Z_DIM); z[2] = -1.0
    s = z_to_settings(z, SEVERE, CFG)
    assert np.all(s.cr == 1.0)                                   # was ~1.25+ before the change
    z[2] = +1.0
    assert np.all(z_to_settings(z, SEVERE, CFG).cr <= CFG.cr_max)
    assert np.all(z_to_settings(np.zeros(Z_DIM), SEVERE, CFG).cr > 1.0)  # z=0 still compresses


def test_static_baselines():
    assert no_correction() is None
    assert set(STATIC_BASELINES) == {"no_correction", "fixed_eq", "prescription",
                                     "prescription_linear", "prescription_slow"}
    p, lin, slow = prescription(SEVERE), prescription_linear(SEVERE), prescription_slow(SEVERE)
    assert np.array_equal(p.gains_db, lin.gains_db) and np.array_equal(p.gains_db, slow.gains_db)
    assert np.all(lin.cr == 1.0) and lin.nr == 0.0 and lin.se == 0.0
    assert slow.attack_ms == 50.0 and slow.release_ms == 500.0 and np.array_equal(slow.cr, p.cr)
    assert np.array_equal(p.gains_db, z_to_settings(np.zeros(Z_DIM), SEVERE, CFG).gains_db)
    assert np.array_equal(fixed_eq().gains_db, fixed_eq(CFG).gains_db)   # audiogram-independent


def test_post_hoc_family_registered_and_penalises_compression_only_in_noise():
    assert "noise_sensitive" in FAMILIES and POST_HOC_FAMILIES == {"noise_sensitive"}
    L = sample_population(1, seed=11)[0]
    L.weights["noise_sensitive"] = L.weights["balanced"].copy()   # identical weights: isolate the new term
    a = L.audiogram
    hi_cr = np.array([0, 0, 1.0, -1, -1])
    for ctx_name in ("noisy_conversation", "outdoor", "quiet", "music"):
        ctx = CONTEXTS[ctx_name]
        extra = (utility(hi_cr, L, ctx, "balanced", a) - utility(hi_cr, L, ctx, "noise_sensitive", a))
        if ctx.snr_db >= 15.0:      # noise_factor = 0 -> family identical to balanced
            assert extra == pytest.approx(0.0, abs=1e-12), (ctx_name, extra)
        else:                       # noisy: strictly lower utility for high compression
            assert extra > 1e-6, (ctx_name, extra)
    # and low compression is not penalised by the extra term (log2(CR)=0 when CR=1)
    lo_cr = np.array([0, 0, -1.0, -1, -1])
    ctx = CONTEXTS["noisy_conversation"]
    assert utility(lo_cr, L, ctx, "balanced", a) == pytest.approx(
        utility(lo_cr, L, ctx, "noise_sensitive", a), abs=1e-12)


def test_post_hoc_family_optimum_prefers_lower_compression_in_noise():
    ctx = CONTEXTS["noisy_conversation"]
    pop = sample_population(6, seed=21)
    z3 = {f: np.mean([find_optimum(L, ctx, f, L.audiogram, seed=L.id).z_opt[2] for L in pop])
          for f in ("balanced", "noise_sensitive")}
    assert z3["noise_sensitive"] < z3["balanced"] - 0.05, z3
