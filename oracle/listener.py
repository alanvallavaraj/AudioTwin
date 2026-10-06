"""Simulated listeners (hidden ground truth; never visible to the twin)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dsp.params import AUDIOGRAM_HZ, BANDS_HZ, interp_audiogram
from .audiograms import sample_parametric

# PLACEHOLDER reference equivalent threshold SPLs (dB) for AUDIOGRAM_HZ; replace with verified
# ISO 389-1 values for the chosen transducer before submission.
RETSPL_PLACEHOLDER = np.array([26.0, 12.0, 7.0, 9.0, 10.0, 15.0, 13.0])

FAMILIES = {  # Dirichlet alphas over weights (audibility, comfort, ideal-curve, artifacts)
    "intelligibility": np.array([6.0, 1.0, 1.0, 1.0]),
    "comfort": np.array([1.0, 5.0, 2.0, 1.0]),
    "balanced": np.array([3.0, 3.0, 2.0, 2.0]),
}


@dataclass
class Listener:
    id: int
    audiogram: np.ndarray          # true dB HL at AUDIOGRAM_HZ
    ucl_db_spl: np.ndarray         # per DSP band
    ideal_offset_db: np.ndarray    # per DSP band, listener-specific preferred gain offset
    cr_pref: float                 # preferred log2 mean compression ratio
    sigma_n: float                 # choice noise (utility units)
    lapse: float                   # lapse rate
    weights: dict = field(default_factory=dict)   # family -> weights (A, C, D, Q)

    def thr_spl(self) -> np.ndarray:
        return interp_audiogram(self.audiogram) + np.interp(
            np.log2(BANDS_HZ), np.log2(AUDIOGRAM_HZ), RETSPL_PLACEHOLDER
        )


def sample_listener(rng: np.random.Generator, lid: int, pta_range=(20.0, 55.0)) -> Listener:
    a = sample_parametric(rng, pta_range)
    thr = interp_audiogram(a) + np.interp(np.log2(BANDS_HZ), np.log2(AUDIOGRAM_HZ), RETSPL_PLACEHOLDER)
    ucl = np.clip(np.maximum(thr + 25.0, 100.0 + rng.normal(0, 4, size=thr.shape)), None, 110.0)
    knots = rng.normal(0, 4, size=3)
    off = np.interp(np.log2(BANDS_HZ), np.log2([250.0, 1000.0, 8000.0]), knots) + rng.normal(0, 3)
    return Listener(
        id=lid,
        audiogram=a,
        ucl_db_spl=ucl,
        ideal_offset_db=off,
        cr_pref=float(rng.uniform(0.0, 1.0)),
        sigma_n=float(rng.uniform(0.05, 0.15)),
        lapse=float(rng.uniform(0.0, 0.05)),
        weights={k: rng.dirichlet(v) for k, v in FAMILIES.items()},
    )


def sample_population(n: int, seed: int, pta_range=(20.0, 55.0)) -> list[Listener]:
    rng = np.random.default_rng(seed)
    return [sample_listener(rng, i, pta_range) for i in range(n)]


def observe_audiogram(listener: Listener, sigma_a: float, rng: np.random.Generator) -> np.ndarray:
    """What the twin knows: true audiogram + measurement error, rounded to 5 dB, clipped."""
    a_hat = listener.audiogram + rng.normal(0, sigma_a, size=listener.audiogram.shape)
    return np.clip(np.round(a_hat / 5.0) * 5.0, 0, 100)
