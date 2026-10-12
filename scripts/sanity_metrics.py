"""Do HASPI/HASQI/ESTOI move sensibly with amplification for a hearing-impaired audiogram?"""
import numpy as np, soundfile as sf
from dsp import DSPConfig, Z_DIM, z_to_settings
from eval.metrics import estoi, haspi, hasqi
from eval.render import render, speech_shaped_noise

CFG = DSPConfig()
sp, fs = sf.read("data/sanity_speech_24k.wav")
sp = sp[: 6 * fs]
AUD = np.array([25, 30, 40, 50, 60, 65, 70], float)   # mild->moderate sloping (synthetic)
rng = np.random.default_rng(0)
noise = speech_shaped_noise(sp, rng)

def conds():
    yield "no correction", None
    for name, z in [("prescription z=0", np.zeros(Z_DIM)),
                    ("gain -1 (z1=-1)", np.array([-1, 0, 0, -1, -1.])),
                    ("gain +1 (z1=+1)", np.array([1, 0, 0, -1, -1.]))]:
        yield name, z_to_settings(z, AUD, CFG)
    yield "half prescription", None  # constructed below

print(f"{'condition':22s} {'SNR':>5s} {'HASPI':>7s} {'HASQI':>7s} {'ESTOI':>7s}")
for snr in (None, 5.0):
    for name, s in [("no correction", None),
                    ("prescription z=0", z_to_settings(np.zeros(Z_DIM), AUD, CFG)),
                    ("gain z1=-1", z_to_settings(np.array([-1, 0, 0, -1, -1.]), AUD, CFG)),
                    ("gain z1=+1", z_to_settings(np.array([1, 0, 0, -1, -1.]), AUD, CFG))]:
        ref, proc = render(sp, noise if snr is not None else None, fs, snr, s, CFG)
        h = haspi(ref, proc, fs, AUD, CFG.cal_db)
        q = hasqi(ref, proc, fs, AUD, CFG.cal_db)
        e = estoi(ref, proc, fs)
        print(f"{name:22s} {('quiet' if snr is None else str(snr)):>5s} {h:7.3f} {q:7.3f} {e:7.3f}")
