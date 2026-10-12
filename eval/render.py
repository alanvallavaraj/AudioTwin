"""Build reference / processed signals for evaluation."""
from __future__ import annotations

import numpy as np

from dsp import DSPConfig, DSPSettings, process


def speech_shaped_noise(speech: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random-phase noise with the speech's long-term magnitude spectrum (sanity use only;
    real experiments should use DEMAND or similar recorded noise)."""
    S = np.fft.rfft(speech)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=S.shape))
    return np.fft.irfft(np.abs(S) * phase, n=len(speech))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise (not speech) so that speech-to-noise RMS ratio equals snr_db."""
    n = noise[: len(speech)]
    g = np.sqrt(np.mean(speech**2) / np.mean(n**2)) * 10 ** (-snr_db / 20.0)
    return speech + g * n


def render(speech: np.ndarray, noise: np.ndarray | None, fs: int, snr_db: float | None,
           settings: DSPSettings | None, cfg: DSPConfig = DSPConfig()):
    """Return (reference, processed). reference = clean speech at presentation level;
    processed = DSP(speech + noise), or the unprocessed mixture if settings is None."""
    mix = speech if noise is None else mix_at_snr(speech, noise, snr_db)
    proc = mix if settings is None else process(mix, fs, settings, cfg)
    return speech, proc
