# Protocol provenance

The repository exposes two histories. They should not be conflated.

## Curated publication branch (`main`)

`main` was assembled after the final experiment to provide a clean publication repository with the released data, manuscript and post-final material. Its commit timestamps are **not** evidence of when the experimental design was fixed.

The frozen records `docs/protocol.md` and `docs/findings.md` are kept on `main` **verbatim** as they stood at commit `7b3ab57` (tag `final-protocol-v1.3`). Their original wording, including the term "pre-registered", is preserved deliberately; the clarifications below are made here instead of by editing those records.

## Original development history (`protocol-history`)

The branch [`protocol-history`](../../tree/protocol-history) publishes the original development lineage up to the replacement-rule freeze. The original annotated tags are public:

| Tag | Commit | Recorded purpose |
|---|---|---|
| `final-protocol-v1` | `52fcd8b` | Freeze scientific final-evaluation protocol |
| `final-protocol-v1.1` | `2017fe0` | QMUL environment-module launcher correction only |
| `final-protocol-v1.2` | `fc040b3` | Bash `nounset` launcher correction only; this is the experiment code hash stored in every final row |
| `final-protocol-v1.3` | `7b3ab57` | Deterministic infeasible-listener replacement rule, recorded after task 55 failed but before final performance analysis |

The manuscript cites these **commit hashes** as well as the tag names, because Git tags can be moved. On 12 October 2026 the `final-protocol-v1.3` tag briefly pointed at the curated snapshot `322a169`; it was restored to the original development commit `7b3ab57` the same day and will not be moved again.

The tags record version-control provenance, not an external protocol registry, and Git timestamps are self-asserted. The paper therefore uses the term **pre-specified** rather than "pre-registered".

The final released dataset (`publication/data/final_v1.parquet`) contains `git_hash=fc040b3` in every row, linking the outputs to `final-protocol-v1.2`. Task 55 generated no result rows because it failed the safe-anchor feasibility check; task 128 is the declared deterministic replacement described by v1.3.

## Clarifications to the frozen protocol text

* **Static comparator.** `docs/protocol.md` describes the primary comparator as "the BEST of prescription, prescription_linear, prescription_slow (per analysis cell)". The analysis implements exactly that minimum. Compression speed, however, has no term in the three primary utility families, so `prescription_slow` and `prescription` have identical utility there. The minimum therefore spans two distinct utility behaviours: the default prescription, and the linear setting with noise reduction and speech enhancement off. The manuscript describes the comparator in these terms.
* **Terminology.** The frozen text says "pre-registered families". These families were fixed in the private development protocol before final outcomes were inspected; they are reported as **pre-specified**.

## Post-final work

Post-final analyses are governed by `docs/protocol_v2_addendum.md` (sections v2 and v2.1). They use development seed `424242`, or descriptive re-analysis of the released final rows. They do not rerun seed `777000111` and cannot redefine the frozen primary result.

## Snapshot cited by the manuscript

The current BSPC submission manuscript corresponds to tag `manuscript-bspc-v1.0` on `main`. A preceding MLwA-targeted snapshot is retained in Git history. Scientific code under `twin/`, `dsp/`, `oracle/`, `baselines/` and `experiments/` is functionally identical to `fc040b3`. Later edits are limited to comments, documentation and two optional arguments of `twin.active.run_laplace_twin` (`gp_gamma`, `extra_base_points`) whose defaults reproduce the frozen behaviour exactly; a regression test checks this.
