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
