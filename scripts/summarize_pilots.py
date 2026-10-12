"""Summarise pilot result sets pulled from HPC (results_hpc/<pilot>/task_*.parquet).
Unit of analysis = listener (task); average over contexts/families within listener first."""
import glob, os, sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results_hpc"
FAMS_PRE = ["intelligibility", "comfort", "balanced"]
for d in sorted(glob.glob(os.path.join(root, "audiotwin_*"))):
    files = sorted(glob.glob(os.path.join(d, "task_*.parquet")))
    if not files:
        continue
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    hashes = sorted(df.git_hash.str[:7].unique())
    print(f"\n=== {os.path.basename(d)}  tasks={len(files)}  rows={len(df)}  code={hashes}  n_queries={sorted(df.n_queries.unique())}")
    for label, sub in [("ALL families", df), ("pre-specified families only", df[df.family.isin(FAMS_PRE)]),
                       ("noise_sensitive (post-hoc)", df[df.family == "noise_sensitive"])]:
        per_l = sub.groupby(["listener_id", "method"]).regret.mean().unstack("method")
        order = list(per_l.mean().sort_values().index)
        row = "  ".join(f"{m}={per_l[m].mean():.3f}" for m in order)
        print(f"  [{label}] mean regret (lower=better): {row}")
