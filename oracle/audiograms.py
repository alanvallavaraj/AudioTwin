"""Audiogram sources.

1. Parametric synthetic family (documented; NOT Bisgaard-derived). Main population.
2. Bisgaard et al. (2010) standard audiograms: only N4 and N6 are VERIFIED here
   (copied from a published table of the Bisgaard values, 250-6000 Hz, 9 frequencies).
   The other eight (N1-N3, N5, N7, S1-S3) must be filled from the paper's table via
   configs/bisgaard_standard_audiograms.csv before they are used. Do not guess them.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dsp.params import AUDIOGRAM_HZ

BISGAARD_HZ = np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000], dtype=float)

VERIFIED_BISGAARD = {
    "N4": np.array([55, 55, 55, 55, 60, 65, 70, 75, 80], dtype=float),
    "N6": np.array([75, 80, 85, 85, 90, 90, 95, 100, 100], dtype=float),
}


def bisgaard_to_audiogram(row9: np.ndarray, extrapolation: str = "hold") -> np.ndarray:
    """Map the 9-frequency Bisgaard row (250-6k) to AUDIOGRAM_HZ (250..8k).

    8 kHz is outside the standard; policy 'hold' repeats the 6 kHz value (state this in the paper).
    """
    if extrapolation != "hold":
        raise ValueError("only 'hold' extrapolation is implemented")
    out = np.interp(np.log2(AUDIOGRAM_HZ), np.log2(BISGAARD_HZ), row9)
    out[AUDIOGRAM_HZ > BISGAARD_HZ[-1]] = row9[-1]
    return out


def load_bisgaard_csv(path: str | Path) -> dict[str, np.ndarray]:
    """CSV columns: name,250,500,750,1000,1500,2000,3000,4000,6000 (values dB HL)."""
    out: dict[str, np.ndarray] = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [str(int(f)) for f in BISGAARD_HZ]
        for r in rd:
            out[r["name"]] = np.array([float(r[c]) for c in cols])
    return out


def pta4(audiogram: np.ndarray) -> float:
    """Mean of 500, 1000, 2000, 4000 Hz thresholds (AUDIOGRAM_HZ order)."""
    idx = [list(AUDIOGRAM_HZ).index(f) for f in (500, 1000, 2000, 4000)]
    return float(np.mean(audiogram[idx]))


def sample_parametric(rng: np.random.Generator, pta_range=(20.0, 55.0)) -> np.ndarray:
    """Synthetic mild-to-moderate audiogram (documented assumptions; rounded to 5 dB)."""
    while True:
        base = rng.uniform(15, 55)            # level at 1 kHz
        slope_hf = rng.uniform(0, 12)         # dB/octave above 1 kHz
        slope_lf = rng.uniform(0, 4)          # dB/octave rise toward low freq (lower loss)
        octs = np.log2(AUDIOGRAM_HZ / 1000.0)
        a = base + np.where(octs > 0, slope_hf * octs, slope_lf * octs)
        a = a + rng.normal(0, 4, size=a.shape)
        a = np.clip(np.round(a / 5.0) * 5.0, 0, 90)
        if pta_range[0] <= pta4(a) <= pta_range[1]:
            return a
