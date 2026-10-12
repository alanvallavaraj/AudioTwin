# Protocol provenance

The repository now exposes two different histories and they should not be conflated.

## Curated publication branch

`main` was assembled after the final experiment to provide a clean publication repository, released data, manuscript and post-final revisions. Its early commit timestamps are **not** evidence of when the experimental design was fixed.

## Original development history

The branch [`protocol-history`](../../tree/protocol-history) publishes the original AudioTwin development lineage up to the replacement-rule freeze. The original annotated tags are also public:

| Tag | Commit | Recorded purpose |
|---|---|---|
| `final-protocol-v1` | `52fcd8b` | Freeze scientific final-evaluation protocol |
| `final-protocol-v1.1` | `2017fe0` | QMUL environment-module launcher correction only |
| `final-protocol-v1.2` | `fc040b3` | Bash `nounset` launcher correction only; this is the experiment code hash stored in the final rows |
| `final-protocol-v1.3` | `7b3ab57` | Deterministic infeasible-listener replacement rule, recorded after task 55 failed but before final performance analysis |

The annotated tag records preserve the original development-repository tagger metadata. They are version-control provenance, not an external protocol registry. The paper therefore uses **pre-specified** terminology.

The final released dataset (`publication/data/final_v1.parquet`) contains `git_hash=fc040b3` in every row, linking the experimental outputs to `final-protocol-v1.2`. Task 55 generated no scientific result rows because it failed at the safe-anchor feasibility check; task 128 is the declared deterministic replacement described by v1.3.

## Post-final post-final diagnostic work

Post-final development-only diagnostics are governed by `docs/protocol_v2_addendum.md`. They use development seed `424242`, do not rerun seed `777000111`, and cannot redefine the frozen primary result.
