"""Evaluate the post-final coordinate-search baseline on the frozen 32-listener dev sweep.

Requires the original development sweep directory as input because its stored oracle optima are reused;
no final-seed data are read. Governed by docs/protocol_v2_addendum.md.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.runner import stable_seed
from oracle import CONTEXTS, observe_audiogram, sample_population, utility
from twin.coordinate_search import run_coordinate_search

DEV_SEED=424242
CONTEXTS_FINAL=("noisy_conversation","lecture","music")
FAMILIES_PRIMARY=("intelligibility","comfort","balanced")


def _one(task_id:int, lookup:dict):
    base_seed=DEV_SEED+1009*task_id
    rng=np.random.default_rng(base_seed)
    listener=sample_population(1,seed=base_seed)[0]
    a_hat=observe_audiogram(listener,5.0,rng)
    rows=[]
    for c in CONTEXTS_FINAL:
        ctx=CONTEXTS[c]
        for f in FAMILIES_PRIMARY:
            ref=lookup[(task_id,c,f)]
            seed=stable_seed(base_seed,"gp",task_id,c,f)
            run=run_coordinate_search(listener,ctx,f,a_hat,n_queries=20,seed=seed,checkpoints=(20,))
            u=utility(run.recs[20],listener,ctx,f,a_hat)
            regret=max(ref[0]-u,0.0)/(ref[0]-ref[1]+1e-9)
            rows.append(dict(task_id=task_id,context=c,family=f,coordinate_t20_regret=regret))
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('dev_results'); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--out-dir',default='publication/postfinal_diagnostics')
    args=ap.parse_args()
    fs=sorted(Path(args.dev_results).glob('task_*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
    if set(df.base_seed.unique()) != {DEV_SEED} or df.task_id.nunique()!=32:
        raise RuntimeError('expected the 32-listener development sweep at base seed 424242')
    lookup={}
    for (t,c,f),g in df[df.family.isin(FAMILIES_PRIMARY)].groupby(['task_id','context','family']):
        r=g.iloc[0]; lookup[(int(t),c,f)]=(float(r.u_opt),float(r.u_lo))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        chunks=list(ex.map(_one, range(32), [lookup]*32))
    coord=pd.DataFrame([r for ch in chunks for r in ch])
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); coord.to_csv(out/'coordinate_search_dev32.csv',index=False)
    # merge with frozen dev qEUBO/static for exact same listeners/cells
    sub=df[df.family.isin(FAMILIES_PRIMARY)]
    static=sub[sub.method.isin(['prescription','prescription_linear','prescription_slow'])].groupby(['listener_id','context','family']).regret.min().rename('static')
    qe=sub[(sub.method=='audiotwin_qeubo') & (sub.n_queries==20)].set_index(['listener_id','context','family']).regret.rename('qeubo')
    co=coord.set_index(['task_id','context','family']).coordinate_t20_regret
    co.index=co.index.set_names(['listener_id','context','family'])
    cell=pd.concat([static,qe,co.rename('coordinate')],axis=1).dropna()
    per=cell.groupby(level=0).mean()
    summary=pd.DataFrame({
        'method':['coordinate','qeubo'],
        'mean_t20_minus_static':[(per.coordinate-per.static).mean(),(per.qeubo-per.static).mean()],
        'listeners_beating_static':[(per.coordinate<per.static).sum(),(per.qeubo<per.static).sum()],
        'n_listeners':[len(per),len(per)],
    })
    summary.to_csv(out/'coordinate_search_dev32_summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__': main()
