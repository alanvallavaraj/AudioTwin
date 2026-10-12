# Post-final development-only diagnostics

These files are **post-final exploratory diagnostics** governed by `docs/protocol_v2_addendum.md`. They use development seed `424242` only and do not reuse final seed `777000111`.

## Coordinate search

`coordinate_search_dev32_summary.csv` compares a simple pairwise coordinate-search baseline against the static reference and frozen development qEUBO at T=20 on the same 32 development listeners.

## Acquisition geometry

`acquisition_geometry_dev32_summary.json` uses the explicit boundary criterion `max(abs(z)) >= 0.9`. Query endpoints over the first 30 comparisons were near the boundary for 22.9% of qEUBO endpoints and 87.6% of BALD endpoints.

## Music–intelligibility diagnostic

`music_intelligibility_dev32_summary.json` tests the observed corner optimum and two candidate explanations. All 32 development optima put NR/SE effectively at the off corner. Removing the quadratic GP centre prior did not improve regret, whereas removing simulated response noise/lapses materially improved it. This result is reported as a hypothesis test rather than used to tune the frozen method.

## Oracle robustness

`oracle_robustness_dev_summary.csv` and associated cell-level output report development-only perturbations of oracle/observation assumptions over 12 listeners. Mean qEUBO T=20 minus static was −0.0568 in the matched base rerun, −0.0594 with doubled ideal-gain offsets, −0.0481 under doubled audiogram error (11/12 feasible), −0.1771 with the ideal-distance term removed, and +0.0482 under doubled choice noise/lapse. Any infeasible safe-anchor draws are retained in the feasibility file and are not repaired by relaxing constraints.
