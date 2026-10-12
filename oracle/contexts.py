"""Listening contexts (all numbers are PLACEHOLDERS; justify or replace before submission)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Placeholder long-term speech band levels (dB SPL) at 65 dB overall, per DSP band.
LTASS_65_DB = np.array([55.0, 55.0, 54.0, 52.0, 49.0, 47.0, 44.0, 41.0, 36.0, 31.0])
# Placeholder speech-importance weights per DSP band (sum to 1).
_I = np.array([0.06, 0.09, 0.09, 0.12, 0.12, 0.14, 0.14, 0.12, 0.08, 0.04])
IMPORTANCE = _I / _I.sum()


@dataclass(frozen=True)
class Context:
    name: str
    speech_db_spl: float       # overall speech level
    snr_db: float              # in-band SNR (positive = speech above noise)
    speech_like: bool          # False for music: NR/SE give no benefit, only artifacts
    nr_benefit_db: float       # SNR improvement at nr=1
    se_benefit_db: float       # SNR improvement at se=1

    def speech_band_db(self) -> np.ndarray:
        return LTASS_65_DB + (self.speech_db_spl - 65.0)

    def noise_band_db(self) -> np.ndarray:
        return self.speech_band_db() - self.snr_db


CONTEXTS = {
    "quiet": Context("quiet", 60.0, 40.0, True, 0.0, 0.0),
    "noisy_conversation": Context("noisy_conversation", 68.0, 3.0, True, 3.0, 4.0),
    "lecture": Context("lecture", 62.0, 12.0, True, 2.0, 3.0),
    "call": Context("call", 62.0, 20.0, True, 1.5, 2.0),
    "outdoor": Context("outdoor", 66.0, 5.0, True, 3.0, 3.0),
    "music": Context("music", 72.0, 30.0, False, 0.0, 0.0),
    "live": Context("live", 80.0, 25.0, False, 0.0, 0.0),
}
