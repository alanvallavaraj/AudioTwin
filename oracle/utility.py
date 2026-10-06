"""Oracle utility u*(z; listener, context). Analytic (no audio rendering), cheap for HPC.

Deliberately built from an SII-like audibility term, a loudness-comfort term, an
ideal-gain-curve term and an artifact term, so it stays disjoint from the REPORTED
evaluation metrics (HASPI/HASQI/PESQ/ESTOI). All constants are placeholders to justify.
"""
from __future__ import annotations

import numpy as np

from dsp.params import DSPConfig, prescription_gain, z_to_settings
from .contexts import IMPORTANCE, Context
from .listener import Listener

PEAK_FACTOR_DB = 12.0


def _pen(x: float) -> float:
    return 1.0 - np.exp(-x)


def utility(z, listener: Listener, ctx: Context, family: str, a_dsp: np.ndarray,
            cfg: DSPConfig = DSPConfig()) -> float:
    """Utility of latent setting z when the DSP was fitted from audiogram a_dsp
    (the twin's estimate) but the listener's true thresholds/limits apply."""
    s = z_to_settings(z, a_dsp, cfg)
    wA, wC, wD, wQ = listener.weights[family]
    L_sp = ctx.speech_band_db()
    comp = -(1.0 - 1.0 / s.cr) * np.maximum(L_sp - cfg.comp_threshold_db_spl, 0.0)
    sp_out = np.minimum(L_sp + s.gains_db + comp, s.mpo_db_spl)

    thr = listener.thr_spl()
    snr_out = ctx.snr_db + s.nr * ctx.nr_benefit_db + s.se * ctx.se_benefit_db
    aud = np.clip((np.minimum(snr_out, sp_out + PEAK_FACTOR_DB - thr) + 15.0) / 30.0, 0.0, 1.0)
    A = float(np.sum(IMPORTANCE * aud))

    mcl = thr + 0.5 * (listener.ucl_db_spl - thr)
    d = (sp_out - mcl) / 10.0
    over = np.maximum(sp_out + PEAK_FACTOR_DB - listener.ucl_db_spl, 0.0) / 5.0
    comfort = float(np.mean(np.where(d > 0, 2.0 * d**2, d**2)) + np.mean(over**2))

    g_ideal = prescription_gain(listener.audiogram, cfg) + listener.ideal_offset_db
    D = float(np.mean(((s.gains_db - g_ideal) / 10.0) ** 2))

    art_w = 1.0 if ctx.speech_like else 2.0
    Q = float(art_w * (0.5 * s.nr**2 + 0.7 * s.se**2)
              + (np.log2(np.mean(s.cr)) - listener.cr_pref) ** 2)
    if family == "noise_sensitive":  # post-hoc: compression artefacts matter more in noise
        noise_factor = float(np.clip((15.0 - ctx.snr_db) / 15.0, 0.0, 1.0))
        Q += 3.0 * noise_factor * float(np.log2(np.mean(s.cr))) ** 2

    return float(wA * A - wC * _pen(comfort) - wD * _pen(D) - wQ * _pen(Q))
