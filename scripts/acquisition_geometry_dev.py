"""Development-only qEUBO/BALD query-geometry diagnostic (seed 424242)."""
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd
from experiments.runner import stable_seed
from oracle import CONTEXTS,observe_audiogram,sample_population
from twin.active import run_laplace_twin
DEV_SEED=424242; CONTEXTS_FINAL=('noisy_conversation','lecture','music'); FAMILIES=('intelligibility','comfort','balanced'); THRESH=0.9

def frac(run):
    pts=[]
    for a,b in run.comparisons: pts += [run.X[int(a)],run.X[int(b)]]
    Z=np.asarray(pts); return float(np.mean(np.max(np.abs(Z),axis=1)>=THRESH))

def one(t):
    base=DEV_SEED+1009*t; rng=np.random.default_rng(base); L=sample_population(1,seed=base)[0]; ah=observe_audiogram(L,5.0,rng); rows=[]
    for c in CONTEXTS_FINAL:
      for f in FAMILIES:
        seed=stable_seed(base,'gp',t,c,f); ctx=CONTEXTS[c]
        for acq in ('qeubo','bald'):
          r=run_laplace_twin(L,ctx,f,ah,n_queries=30,seed=seed,acquisition=acq,checkpoints=(30,))
          rows.append(dict(task_id=t,context=c,family=f,acquisition=acq,boundary_fraction=frac(r)))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',type=int,default=32); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--out-dir',default='publication/postfinal_diagnostics'); a=ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as ex: chunks=list(ex.map(one,range(a.tasks)))
    df=pd.DataFrame([r for ch in chunks for r in ch]); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'acquisition_geometry_dev32.csv',index=False)
    summary={'development_seed':DEV_SEED,'n_listeners':a.tasks,'endpoint_boundary_threshold':THRESH,
             'qeubo_boundary_fraction':float(df[df.acquisition=='qeubo'].boundary_fraction.mean()),
             'bald_boundary_fraction':float(df[df.acquisition=='bald'].boundary_fraction.mean())}
    (out/'acquisition_geometry_dev32_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
