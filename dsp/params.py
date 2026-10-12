"""Latent parameterisation z -> DSP settings (docs/methods.md).

Numeric values are explicit simulation assumptions, not clinical prescription constants.
They are enumerated in docs/simulator_parameters.md and the manuscript parameter table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BANDS_HZ = np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000], dtype=float)
AUDIOGRAM_HZ = np.array([250, 500, 1000, 2000, 4000, 6000, 8000], dtype=float)
Z_DIM = 6  # gain offset, tilt, compression scale, NR, SE, compression speed


@dataclass(frozen=True)
class DSPConfig:
    beta_gain_db: float = 10.0       # dB per unit z1
    beta_tilt_db_oct: float = 6.0    # dB/octave per unit z2 (pivot 1 kHz)
    half_gain_factor: float = 0.5    # prescription: g_rx = factor * threshold
    gmax_slope: float = 0.8          # g_max = clip(slope*thr + offset, 0, gmax_cap)
    gmax_offset_db: float = 5.0
    gmax_cap_db: float = 50.0
    cr_max: float = 4.0
    cr_exponent_scale: float = 2.0   # CR = CR_rx * 2^(scale*z3); scale 2 lets z3=-1 reach CR=1
    comp_threshold_db_spl: float = 45.0   # compression kneepoint (band level)
    ucl_db_spl: float = 100.0             # predicted-output ceiling per band
    mpo_db_spl: float = 105.0             # hard output limiter per band
    ref_input_db_spl: float = 65.0        # reference band level for safety checks
    smooth_delta_db: float = 12.0         # max gain difference between adjacent bands
    change_rate_delta: float = 0.5        # max |z - z_prev|_inf
    attack_ms: float = 5.0                # z6=0 baseline attack time
    release_ms: float = 50.0              # z6=0 baseline release time
    slow_attack_ms: float = 50.0          # z6=1 slow-compression attack time
    slow_release_ms: float = 500.0        # z6=1 slow-compression release time
    nr_max: float = 1.0
    se_max: float = 1.0
    cal_db: float = 100.0                 # dB SPL of a full-scale-RMS signal


@dataclass
class DSPSettings:
    gains_db: np.ndarray
    cr: np.ndarray
    nr: float
    se: float
    attack_ms: float
    release_ms: float
    mpo_db_spl: np.ndarray = field(default_factory=lambda: np.full(len(BANDS_HZ), 105.0))


def interp_audiogram(audiogram_db_hl: np.ndarray) -> np.ndarray:
    """Interpolate 7-point audiogram to the 10 DSP bands (linear in log-frequency)."""
    return np.interp(np.log2(BANDS_HZ), np.log2(AUDIOGRAM_HZ), audiogram_db_hl)


def prescription_gain(audiogram_db_hl: np.ndarray, cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    """Synthetic half-gain prescription used in all reported experiments.

This is not NAL-NL2 or NAL-R: gain is 0.5 times the interpolated threshold.
"""
    thr = interp_audiogram(audiogram_db_hl)
    return np.clip(cfg.half_gain_factor * thr, 0.0, None)


def max_gain(audiogram_db_hl: np.ndarray, cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    thr = interp_audiogram(audiogram_db_hl)
    return np.clip(cfg.gmax_slope * thr + cfg.gmax_offset_db, 0.0, cfg.gmax_cap_db)


def prescription_cr(audiogram_db_hl: np.ndarray, cfg: DSPConfig = DSPConfig()) -> np.ndarray:
    thr = interp_audiogram(audiogram_db_hl)
    return 1.0 + 1.5 * np.clip((thr - 20.0) / 40.0, 0.0, 1.0)


def _time_from_z(z6: float, fast_ms: float, slow_ms: float) -> float:
    # z6=0 keeps the original fast baseline; z6=1 reaches the slow-compression variant.
    exponent = np.clip(z6, -1.0, 1.0)
    return float(fast_ms * (slow_ms / fast_ms) ** exponent)


def z_to_settings(z: np.ndarray, audiogram_db_hl: np.ndarray, cfg: DSPConfig = DSPConfig()) -> DSPSettings:
    z = np.clip(np.asarray(z, dtype=float), -1.0, 1.0)
    if z.shape != (Z_DIM,):
        raise ValueError(f"z must have shape ({Z_DIM},)")
    g_rx = prescription_gain(audiogram_db_hl, cfg)
    g_max = max_gain(audiogram_db_hl, cfg)
    gains = np.clip(
        g_rx + cfg.beta_gain_db * z[0] + cfg.beta_tilt_db_oct * z[1] * np.log2(BANDS_HZ / 1000.0),
        0.0,
        g_max,
    )
    cr = np.clip(prescription_cr(audiogram_db_hl, cfg) * 2.0 ** (cfg.cr_exponent_scale * z[2]), 1.0, cfg.cr_max)
    return DSPSettings(
        gains_db=gains,
        cr=cr,
        nr=cfg.nr_max * (z[3] + 1.0) / 2.0,
        se=cfg.se_max * (z[4] + 1.0) / 2.0,
        attack_ms=_time_from_z(z[5], cfg.attack_ms, cfg.slow_attack_ms),
        release_ms=_time_from_z(z[5], cfg.release_ms, cfg.slow_release_ms),
        mpo_db_spl=np.full(len(BANDS_HZ), cfg.mpo_db_spl),
    )
