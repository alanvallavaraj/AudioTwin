"""Static (non-interactive) baselines. Each returns DSPSettings, or None for 'no correction'
(eval.render.render treats None as the unprocessed mixture).

Interactive baselines (simulated manual self-adjustment, in-situ audiometry fit with its own
audiogram error, AudioTwin variants) live in the experiment runner stage.

Fairness note (docs/findings.md): fast compression lowers HASPI in noise, so the prescription is
reported THREE ways (as built, linear, slow compression). AudioTwin must beat the best of them.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dsp.params import BANDS_HZ, DSPConfig, DSPSettings, Z_DIM, z_to_settings


def no_correction() -> None:
    return None


def fixed_eq(cfg: DSPConfig = DSPConfig()) -> DSPSettings:
    """Audiogram-independent consumer EQ: gentle high-frequency lift, no compression.
    PLACEHOLDER curve (3 dB/octave above 1 kHz, capped at 12 dB); justify or replace."""
    gains = np.clip(3.0 * np.log2(BANDS_HZ / 1000.0), 0.0, 12.0)
    nb = len(BANDS_HZ)
    return DSPSettings(gains, np.ones(nb), 0.0, 0.0, cfg.attack_ms, cfg.release_ms,
                       np.full(nb, cfg.mpo_db_spl))


def prescription(a_hat: np.ndarray, cfg: DSPConfig = DSPConfig()) -> DSPSettings:
    """z = 0: half-gain gains + prescribed CR, default (fast) compression times."""
    return z_to_settings(np.zeros(Z_DIM), a_hat, cfg)


def prescription_linear(a_hat: np.ndarray, cfg: DSPConfig = DSPConfig()) -> DSPSettings:
    s = prescription(a_hat, cfg)
    return dataclasses.replace(s, cr=np.ones_like(s.cr), nr=0.0, se=0.0)


def prescription_slow(a_hat: np.ndarray, cfg: DSPConfig = DSPConfig(),
                      attack_ms: float = 50.0, release_ms: float = 500.0) -> DSPSettings:
    s = prescription(a_hat, cfg)
    return dataclasses.replace(s, attack_ms=attack_ms, release_ms=release_ms)


STATIC_BASELINES = {
    "no_correction": lambda a, cfg=DSPConfig(): None,
    "fixed_eq": lambda a, cfg=DSPConfig(): fixed_eq(cfg),
    "prescription": prescription,
    "prescription_linear": prescription_linear,
    "prescription_slow": prescription_slow,
}
