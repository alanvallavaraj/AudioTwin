"""Why does the prescription score below 'no correction' on HASPI at 5 dB SNR?"""
import dataclasses, numpy as np, soundfile as sf
from dsp import DSPConfig, Z_DIM, z_to_settings
from eval.metrics import estoi, haspi, hasqi
from eval.render import render, speech_shaped_noise

CFG = DSPConfig()
sp, fs = sf.read("data/sanity_speech_24k.wav"); sp = sp[: 6 * fs]
AUD = np.array([25, 30, 40, 50, 60, 65, 70], float)
noise = speech_shaped_noise(sp, np.random.default_rng(0))
base = z_to_settings(np.zeros(Z_DIM), AUD, CFG)
variants = {
    "no correction": None,
    "prescription (as built)": base,
    "same, CR=1 (linear)": dataclasses.replace(base, cr=np.ones_like(base.cr)),
    "same, CR=1, nr=se=0": dataclasses.replace(base, cr=np.ones_like(base.cr), nr=0.0, se=0.0),
    "slow comp (50/500 ms)": dataclasses.replace(base, attack_ms=50.0, release_ms=500.0),
    "CR as built, nr=se=0": dataclasses.replace(base, nr=0.0, se=0.0),
}
print("gains dB:", np.round(base.gains_db, 1), " CR:", np.round(base.cr, 2), " nr/se:", base.nr, base.se)
print(f"{'variant':26s} {'HASPI':>7s} {'HASQI':>7s} {'ESTOI':>7s}")
for name, s in variants.items():
    ref, proc = render(sp, noise, fs, 5.0, s, CFG)
    print(f"{name:26s} {haspi(ref, proc, fs, AUD, CFG.cal_db):7.3f} {hasqi(ref, proc, fs, AUD, CFG.cal_db):7.3f} {estoi(ref, proc, fs):7.3f}")
