# Released final synthetic dataset

`final_v1.parquet` is the consolidated 32,256-row final AudioTwin simulation dataset used by the frozen analysis.

- 128 independent analysable simulated listeners
- original task 55 was infeasible under the fixed safe-anchor rule and is absent
- deterministic replacement task 128 is included
- contexts: noisy_conversation, lecture, music
- primary families: intelligibility, comfort, balanced
- post-hoc family: noise_sensitive
- methods: three recorded static settings plus qEUBO, BALD and GP-random

SHA-256:
`eca6e5a8b5b679e2d0ffdaab467bb62c1a7ad4593a01241227ffc0f917cc61b8`

Recompute the frozen output with:

```bash
python scripts/analyze_final.py publication/data/final_v1.parquet
```

The consolidated-file output has been verified to match the analysis of the original 128 per-task parquet files.
