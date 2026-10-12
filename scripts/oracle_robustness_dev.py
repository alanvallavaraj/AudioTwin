"""Development-only oracle/observation robustness checks for AudioTwin.

All scenarios use development seed 424242 and are governed by docs/protocol_v2_addendum.md.
They do not replace the frozen final N=128 result.
"""
from __future__ import annotations
import argparse, dataclasses, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from experiments.runner import stable_seed
from oracle import CONTEXTS, find_optimum, observe_audiogram, sample_population, utility
from twin.active import run_laplace_twin
from twin.optimizer import feasible_anchor, project_from_anchor

DEV_SEED=424242; CONTEXTS_FINAL=('noisy_conversation','lecture','music'); FAMILIES=('intelligibility','comfort','balanced')
SCENARIOS=('base','no_ideal_term','double_ideal_offset','high_choice_noise','high_audiogram_error')


def modify_listener(L,scenario):
    if scenario=='no_ideal_term':
        weights={}
        for k,w in L.weights.items():
            x=np.array(w,float); x[2]=0.0; x=x/x.sum(); weights[k]=x
        return dataclasses.replace(L,weights=weights)
    if scenario=='double_ideal_offset': return dataclasses.replace(L,ideal_offset_db=2.0*L.ideal_offset_db)
    if scenario=='high_choice_noise': return dataclasses.replace(L,sigma_n=min(0.30,2.0*L.sigma_n),lapse=min(0.10,2.0*L.lapse))
    return L


def one(task_scenario):
    task_id,scenario=task_scenario; base=DEV_SEED+1009*task_id
    rng=np.random.default_rng(base); L0=sample_population(1,seed=base)[0]; L=modify_listener(L0,scenario)
    sigma_a=10.0 if scenario=='high_audiogram_error' else 5.0
    a_hat=observe_audiogram(L,sigma_a,rng)
    try:
        anchor=feasible_anchor(a_hat)
    except ValueError:
        return {"task_id":task_id, "scenario":scenario, "feasible":False, "rows":[]}
    z0=project_from_anchor(np.zeros(6),anchor,a_hat); zlin=project_from_anchor(np.r_[0,0,-1,-1,-1,0],anchor,a_hat)
    rows=[]
    for c in CONTEXTS_FINAL:
      ctx=CONTEXTS[c]
      for f in FAMILIES:
        opt=find_optimum(L,ctx,f,a_hat,seed=stable_seed(base,'robust-opt',scenario,task_id,c,f),n_random=400,maxiter=60,popsize=12)
        static=min(opt.regret(utility(z0,L,ctx,f,a_hat)),opt.regret(utility(zlin,L,ctx,f,a_hat)))
        run=run_laplace_twin(L,ctx,f,a_hat,n_queries=20,seed=stable_seed(base,'robust-gp',scenario,task_id,c,f),acquisition='qeubo',checkpoints=(20,))
        q=opt.regret(utility(run.recs[20],L,ctx,f,a_hat))
        rows.append(dict(task_id=task_id,scenario=scenario,context=c,family=f,static_regret=static,qeubo_regret=q))
    return {"task_id":task_id, "scenario":scenario, "feasible":True, "rows":rows}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',type=int,default=12); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--out-dir',default='publication/postfinal_diagnostics'); a=ap.parse_args()
    jobs=[(t,s) for s in SCENARIOS for t in range(a.tasks)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex: results=list(ex.map(one,jobs))
    df=pd.DataFrame([r for res in results for r in res['rows']]); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'oracle_robustness_dev.csv',index=False)
    feasibility=pd.DataFrame([{k:res[k] for k in ('task_id','scenario','feasible')} for res in results])
    feasibility.to_csv(out/'oracle_robustness_feasibility.csv',index=False)
    pl=df.groupby(['scenario','task_id'])[['static_regret','qeubo_regret']].mean(); pl['delta']=pl.qeubo_regret-pl.static_regret
    summary=pl.groupby('scenario').delta.agg(['mean','std','count']).reset_index()
    beats=pl.groupby('scenario').apply(lambda g:int((g.delta<0).sum()),include_groups=False).rename('listeners_beating_static')
    summary=summary.merge(beats,on='scenario',how='left')
    feasible_counts=feasibility.groupby('scenario').feasible.agg(['sum','count']).reset_index().rename(columns={'sum':'feasible_listeners','count':'attempted_listeners'})
    summary=feasible_counts.merge(summary,on='scenario',how='left')
    summary.to_csv(out/'oracle_robustness_dev_summary.csv',index=False); print(summary.to_string(index=False))
if __name__=='__main__': main()
