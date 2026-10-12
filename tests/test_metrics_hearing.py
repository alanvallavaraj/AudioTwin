"""Slow sanity tests for HASPI/HASQI wiring. Need data/sanity_speech_24k.wav (git-ignored;
LibriSpeech SLR12 excerpt via librosa.example('libri1'), CC-BY-4.0, rescaled to 65 dB SPL)."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from dsp import DSPConfig, Z_DIM, z_to_settings
from eval.metrics import estoi, haspi, hasqi
from eval.render import render, speech_shaped_noise

DATA = Path(__file__).resolve().parents[1] / "data" / "sanity_speech_24k.wav"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not DATA.exists(), reason="sanity speech missing")]
CFG = DSPConfig()
AUD = np.array([25, 30, 40, 50, 60, 65, 70], float)


@pytest.fixture(scope="module")
def clip():
    import soundfile as sf
    sp, fs = sf.read(DATA)
    sp = sp[: 6 * fs]
    return sp, fs, speech_shaped_noise(sp, np.random.default_rng(0))


def _scores(sp, noise, fs, snr, s):
    ref, proc = render(sp, noise if snr is not None else None, fs, snr, s, CFG)
    return haspi(ref, proc, fs, AUD, CFG.cal_db), hasqi(ref, proc, fs, AUD, CFG.cal_db), estoi(ref, proc, fs)


def test_quiet_amplification_helps_and_estoi_is_audibility_blind(clip):
    sp, fs, noise = clip
    h0, q0, e0 = _scores(sp, noise, fs, None, None)
    s = z_to_settings(np.zeros(Z_DIM), AUD, CFG)
    h1, q1, e1 = _scores(sp, noise, fs, None, s)
    assert h1 >= h0 and q1 > q0 + 0.2          # audiogram-aware metrics reward audibility
    assert e0 == pytest.approx(1.0, abs=1e-3)   # ESTOI cannot see the unaided audibility loss


def test_noise_linear_gain_beats_unaided_but_fast_compression_hurts(clip):
    """Documents a real finding (see docs/findings.md): fast WDRC lowers HASPI in noise."""
    sp, fs, noise = clip
    base = z_to_settings(np.zeros(Z_DIM), AUD, CFG)
    lin = dataclasses.replace(base, cr=np.ones_like(base.cr))
    h_none = _scores(sp, noise, fs, 5.0, None)[0]
    h_lin = _scores(sp, noise, fs, 5.0, lin)[0]
    h_fast = _scores(sp, noise, fs, 5.0, base)[0]
    assert h_lin > h_none
    assert h_fast < h_lin - 0.1
