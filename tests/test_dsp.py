import numpy as np
import pytest

from dsp import (BANDS_HZ, DSPConfig, DSPSettings, Z_DIM, band_masks, is_feasible,
                 process, project_to_S, z_to_settings)

FS = 24000
CFG = DSPConfig()
AUD = np.array([20, 25, 35, 45, 55, 60, 65], dtype=float)  # mild-moderate sloping


def tone(f, spl, dur=2.0, fs=FS, cal=CFG.cal_db):
    t = np.arange(int(dur * fs)) / fs
    rms = 10 ** ((spl - cal) / 20)
    return np.sqrt(2) * rms * np.sin(2 * np.pi * f * t)


def level(x, cal=CFG.cal_db):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12) + cal


def flat_settings(gain=0.0, cr=1.0, mpo=140.0, nr=0.0):
    nb = len(BANDS_HZ)
    return DSPSettings(np.full(nb, gain), np.full(nb, cr), nr, 0.0, 5.0, 50.0, np.full(nb, mpo))


def test_masks_partition_of_unity():
    m = band_masks(4800, FS)
    assert np.allclose(m.sum(axis=0), 1.0, atol=1e-12)


def test_unity_gain_is_transparent():
    rng = np.random.default_rng(0)
    x = 0.05 * rng.standard_normal(FS)
    y = process(x, FS, flat_settings())
    assert np.max(np.abs(y - x)) < 1e-9


def test_linear_gain_20db():
    x = tone(1000, 60)
    y = process(x, FS, flat_settings(gain=20.0))
    assert level(y[FS // 2:]) - level(x[FS // 2:]) == pytest.approx(20.0, abs=0.3)


def test_compression_slope():
    s = flat_settings(gain=0.0, cr=2.0)
    lo = process(tone(1000, 60), FS, s)
    hi = process(tone(1000, 80), FS, s)
    delta = level(hi[FS:]) - level(lo[FS:])
    assert delta == pytest.approx(10.0, abs=1.5)  # 20 dB in -> ~10 dB out at CR=2


def test_limiter():
    s = flat_settings(gain=30.0, mpo=90.0)
    y = process(tone(1000, 80), FS, s)
    assert level(y[FS:]) <= 90.5


def test_z0_is_prescription_and_bounds():
    s = z_to_settings(np.zeros(Z_DIM), AUD, CFG)
    assert np.all(s.gains_db >= 0) and np.all(s.cr >= 1) and np.all(s.cr <= CFG.cr_max)
    assert s.attack_ms == pytest.approx(CFG.attack_ms)
    assert s.release_ms == pytest.approx(CFG.release_ms)
    s2 = z_to_settings(np.ones(Z_DIM) * 5.0, AUD, CFG)  # out-of-range z is clipped
    assert np.all(s2.gains_db <= 50.0 + 1e-9)


def test_z6_reaches_slow_compression_times():
    z = np.zeros(Z_DIM)
    z[5] = 1.0
    s = z_to_settings(z, AUD, CFG)
    assert s.attack_ms == pytest.approx(CFG.slow_attack_ms)
    assert s.release_ms == pytest.approx(CFG.slow_release_ms)


def test_prescription_is_feasible_and_projection_works():
    assert is_feasible(np.zeros(Z_DIM), AUD, None, CFG)
    z = np.ones(Z_DIM)
    zp = project_to_S(z, AUD, None, CFG)
    assert is_feasible(zp, AUD, None, CFG)
    assert np.all(np.abs(zp) <= np.abs(z) + 1e-12)
