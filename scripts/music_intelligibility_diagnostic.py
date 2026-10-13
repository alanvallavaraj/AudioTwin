"""Development-only diagnostic for the music-intelligibility failure mode.

Tests two candidate explanations without touching the reserved final seed: (i) the optimum lies near the
NR/SE-off corner and is penalised by the GP centre prior; (ii) response noise is or is not sufficient
to explain the failure. Governed by docs/protocol_v2_addendum.md.
"""
from __future__ import annotations
import argparse, dataclasses, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.runner import stable_seed
from oracle import CONTEXTS, find_optimum, observe_audiogram, sample_population, utility
from twin.active import run_laplace_twin
from twin.laplace_gp import LaplacePreferenceGP

DEV_SEED=424242


def _regret(opt,z,L,ctx,a_hat):
    return opt.regret(utility(z,L,ctx,'intelligibility',a_hat))


def one(task_id:int):
    base=DEV_SEED+1009*task_id
    rng=np.random.default_rng(base)
    L=sample_population(1,seed=base)[0]
    a_hat=observe_audiogram(L,5.0,rng)
    ctx=CONTEXTS['music']
    opt=find_optimum(L,ctx,'intelligibility',a_hat,seed=stable_seed(base,'opt',task_id,'music','intelligibility'),n_random=400,maxiter=60,popsize=12)
    seed=stable_seed(base,'gp',task_id,'music','intelligibility')
    original=run_laplace_twin(L,ctx,'intelligibility',a_hat,n_queries=20,seed=seed,acquisition='qeubo',checkpoints=(20,),gp_gamma=0.1)
    noiseless_L=dataclasses.replace(L,sigma_n=1e-6,lapse=0.0)
    noiseless=run_laplace_twin(noiseless_L,ctx,'intelligibility',a_hat,n_queries=20,seed=seed,acquisition='qeubo',checkpoints=(20,),gp_gamma=0.1)
    no_prior=run_laplace_twin(L,ctx,'intelligibility',a_hat,n_queries=20,seed=seed,acquisition='qeubo',checkpoints=(20,),gp_gamma=0.0)
    prior=LaplacePreferenceGP(gamma=0.1)
    return {
        'task_id':task_id,
        'zopt4_nr':float(opt.z_opt[3]),'zopt5_se':float(opt.z_opt[4]),
        'nrse_corner':bool(opt.z_opt[3] <= -0.9 and opt.z_opt[4] <= -0.9),
        'prior_mean_opt':float(prior.prior_mean(opt.z_opt[None,:])[0]),
        'original_regret':_regret(opt,original.recs[20],L,ctx,a_hat),
        'noiseless_regret':_regret(opt,noiseless.recs[20],L,ctx,a_hat),
        'no_prior_regret':_regret(opt,no_prior.recs[20],L,ctx,a_hat),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',type=int,default=32); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--out-dir',default='publication/postfinal_diagnostics'); args=ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as ex: rows=list(ex.map(one,range(args.tasks)))
    df=pd.DataFrame(rows); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'music_intelligibility_dev32.csv',index=False)
    summary={
        'development_seed':DEV_SEED,'n_listeners':len(df),
        'nr_se_corner_fraction':float(df.nrse_corner.mean()),
        'mean_z4_nr':float(df.zopt4_nr.mean()),'mean_z5_se':float(df.zopt5_se.mean()),
        'mean_prior_at_optimum':float(df.prior_mean_opt.mean()),
        'original_t20_regret':float(df.original_regret.mean()),
        'noiseless_t20_regret':float(df.noiseless_regret.mean()),
        'zero_gamma_t20_regret':float(df.no_prior_regret.mean()),
        'noiseless_better_fraction':float((df.noiseless_regret < df.original_regret).mean()),
        'zero_gamma_better_fraction':float((df.no_prior_regret < df.original_regret).mean()),
    }
    (out/'music_intelligibility_dev32_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
