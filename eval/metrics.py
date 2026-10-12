"""Evaluation metric wrappers (REPORTED metrics; kept disjoint from the oracle utility).

IMPORTANT: STOI/ESTOI/PESQ are essentially insensitive to overall gain and know nothing about
the listener's thresholds. For hearing-aid evaluation they must be computed on the output of a
hearing-loss simulation (so audibility matters), or they cannot distinguish a good fitting from
no fitting. HASPI/HASQI take the audiogram directly. Document which route each metric uses.
"""
from __future__ import annotations

import numpy as np


def _align(ref: np.ndarray, proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(proc))
    return np.asarray(ref[:n], dtype=float), np.asarray(proc[:n], dtype=float)


def estoi(ref: np.ndarray, proc: np.ndarray, fs: int) -> float:
    """Extended STOI (pystoi). ref/proc must already be on the same level/audibility footing."""
    from pystoi import stoi

    r, p = _align(ref, proc)
    return float(stoi(r, p, fs, extended=True))


def stoi_classic(ref: np.ndarray, proc: np.ndarray, fs: int) -> float:
    from pystoi import stoi

    r, p = _align(ref, proc)
    return float(stoi(r, p, fs, extended=False))


def pesq_wb(ref: np.ndarray, proc: np.ndarray, fs: int) -> float:
    """Wide-band PESQ (needs fs 16000). Lazy import so the package works without it."""
    try:
        from pesq import pesq
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install pesq") from e
    if fs != 16000:
        raise ValueError("PESQ wide-band requires fs=16000; resample first")
    r, p = _align(ref, proc)
    return float(pesq(fs, r, p, "wb"))


# ---------------------------------------------------------------------------------------
# HASPI / HASQI via pyclarity (Kates & Arehart models). Audiogram-aware, so these are the
# PRIMARY reported metrics. Conventions (from pyclarity docs):
#   * reference = CLEAN speech, no amplification, no noise (HASQI can add NAL-R itself)
#   * processed = hearing-aid output (including any noise / distortion)
#   * level1 = dB SPL that corresponds to signal RMS = 1  -> must equal DSPConfig.cal_db
#   * audiogram: 6 audiometric frequencies 250..6000 Hz (8 kHz is NOT used by these metrics;
#     state this in the paper)
# ---------------------------------------------------------------------------------------
_METRIC_FREQS_HZ = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0])


def _audiogram6(audiogram_db_hl_7):
    from clarity.utils.audiogram import Audiogram

    a = np.asarray(audiogram_db_hl_7, dtype=float)
    if a.shape != (7,):
        raise ValueError("expected the 7-point AUDIOGRAM_HZ audiogram (250..8000 Hz)")
    return Audiogram(levels=a[:6], frequencies=_METRIC_FREQS_HZ)


def haspi(ref: np.ndarray, proc: np.ndarray, fs: int, audiogram_db_hl_7, level1: float) -> float:
    from clarity.evaluator.haspi import haspi_v2

    r, p = _align(ref, proc)
    score, _ = haspi_v2(r, fs, p, fs, _audiogram6(audiogram_db_hl_7), level1=level1)
    return float(score)


def hasqi(ref: np.ndarray, proc: np.ndarray, fs: int, audiogram_db_hl_7, level1: float) -> float:
    """Combined HASQI v2 (NAL-R equalisation added to the reference by the model)."""
    from clarity.evaluator.hasqi import hasqi_v2

    r, p = _align(ref, proc)
    combined, _, _, _ = hasqi_v2(r, fs, p, fs, _audiogram6(audiogram_db_hl_7), equalisation=1, level1=level1)
    return float(combined)
