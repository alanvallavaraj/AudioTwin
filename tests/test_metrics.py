import numpy as np
import pytest

from eval.metrics import estoi, stoi_classic

FS = 16000


def speechlike(seed=0, dur=4.0):
    """Amplitude-modulated noise with syllable-rate (4 Hz) envelope: crude speech stand-in
    for sanity checks only, NOT for reported results."""
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    return env * rng.standard_normal(n) * 0.1


def test_estoi_identity_and_monotone_in_noise():
    x = speechlike()
    rng = np.random.default_rng(1)
    assert estoi(x, x, FS) == pytest.approx(1.0, abs=1e-3)
    scores = []
    for snr in (20, 5, -5):
        noise = rng.standard_normal(len(x))
        noise *= np.sqrt(np.mean(x**2) / np.mean(noise**2)) * 10 ** (-snr / 20)
        scores.append(estoi(x, x + noise, FS))
    assert scores[0] > scores[1] > scores[2]


def test_stoi_is_gain_invariant():
    """Documents WHY audibility must come from a hearing-loss model, not from STOI itself."""
    x = speechlike(2)
    assert stoi_classic(x, 0.1 * x, FS) == pytest.approx(stoi_classic(x, 10 * x, FS), abs=0.01)
