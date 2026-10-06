"""Offline multiband processor: partition-of-unity band split, per-band gain,
compression, simple expander-style noise reduction, and output limiting.

Simulation-only: zero-phase, non-causal (no device latency modelled).
Speech enhancement (settings.se) is a hook and currently an identity.
"""
from __future__ import annotations

import numpy as np

from .params import BANDS_HZ, DSPConfig, DSPSettings

_EPS = 1e-12


def band_masks(n_samples: int, fs: float, centers_hz: np.ndarray = BANDS_HZ) -> np.ndarray:
    """Raised-cosine masks over log-frequency; shape (n_bands, n_fft_bins); sum to 1 exactly."""
    f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    logf = np.log2(np.maximum(f, 1e-6))
    logc = np.log2(centers_hz)
    nb = len(centers_hz)
    masks = np.zeros((nb, len(f)))
    # below first centre / above last centre: full weight to edge band
    masks[0, logf <= logc[0]] = 1.0
    masks[-1, logf >= logc[-1]] = 1.0
    for k in range(nb - 1):
        sel = (logf >= logc[k]) & (logf < logc[k + 1])
        t = (logf[sel] - logc[k]) / (logc[k + 1] - logc[k])
        w_hi = 0.5 - 0.5 * np.cos(np.pi * t)
        masks[k + 1, sel] = w_hi
        masks[k, sel] = 1.0 - w_hi
    return masks


def split_bands(x: np.ndarray, fs: float, centers_hz: np.ndarray = BANDS_HZ) -> np.ndarray:
    X = np.fft.rfft(x)
    masks = band_masks(len(x), fs, centers_hz)
    return np.fft.irfft(masks * X[None, :], n=len(x), axis=1)


def _frame_levels_db(b: np.ndarray, frame: int, cal_db: float) -> np.ndarray:
    n_frames = int(np.ceil(len(b) / frame))
    pad = n_frames * frame - len(b)
    bp = np.pad(b, (0, pad), mode="edge") if pad else b
    rms = np.sqrt(np.mean(bp.reshape(n_frames, frame) ** 2, axis=1))
    return 20.0 * np.log10(rms + _EPS) + cal_db


def _smooth_levels(L: np.ndarray, hop_s: float, attack_ms: float, release_ms: float) -> np.ndarray:
    a_att = np.exp(-hop_s / (attack_ms * 1e-3))
    a_rel = np.exp(-hop_s / (release_ms * 1e-3))
    out = np.empty_like(L)
    s = L[0]
    for i, v in enumerate(L):
        a = a_att if v > s else a_rel
        s = a * s + (1.0 - a) * v
        out[i] = s
    return out


def process(
    x: np.ndarray,
    fs: float,
    settings: DSPSettings,
    cfg: DSPConfig = DSPConfig(),
    frame_ms: float = 5.0,
) -> np.ndarray:
    """Apply multiband gain/compression/NR/limiting to mono signal x (float, full scale = +/-1)."""
    x = np.asarray(x, dtype=float)
    bands = split_bands(x, fs)
    frame = max(int(round(frame_ms * 1e-3 * fs)), 1)
    hop_s = frame / fs
    n = len(x)
    n_frames = int(np.ceil(n / frame))
    centres = (np.arange(n_frames) + 0.5) * frame
    t = np.arange(n)

    y = np.zeros(n)
    for k in range(bands.shape[0]):
        L = _smooth_levels(_frame_levels_db(bands[k], frame, cfg.cal_db), hop_s, settings.attack_ms, settings.release_ms)
        comp = -(1.0 - 1.0 / settings.cr[k]) * np.maximum(L - cfg.comp_threshold_db_spl, 0.0)
        gain = settings.gains_db[k] + comp
        if settings.nr > 0:  # simple expander: attenuate very low-level band energy
            gain = gain - settings.nr * 0.5 * np.minimum(np.maximum(30.0 - L, 0.0), 20.0)
        over = (L + gain) - settings.mpo_db_spl[k]
        gain = gain - np.maximum(over, 0.0)  # output limiter
        g_samples = np.interp(t, centres, gain)
        y += bands[k] * 10.0 ** (g_samples / 20.0)
    # se hook: identity (speech-enhancement stage to be added)
    return y
