# Cover letter — Machine Learning with Applications

Dear Editor,

Please consider the manuscript **“AudioTwin: Safety-Constrained Preferential Bayesian Optimisation for Query-Efficient Personalised Hearing-Aid Fitting”** for publication as a Research Article in *Machine Learning with Applications*.

The manuscript presents an applied machine-learning framework for personalisation when the objective is an unknown human preference, feedback is limited to pairwise choices, and candidate decisions must remain within hard feasibility constraints. The methodological contribution combines a Laplace-approximated Gaussian-process preference model with qEUBO acquisition in a continuous six-dimensional constrained domain. The framework is evaluated through hearing-aid fitting, but the underlying problem—query-efficient constrained preference optimisation—is relevant to a broader class of interactive ML applications.

The final evaluation was frozen before exposure of the reserved test seed and contains 128 independent simulated listeners. At the pre-specified 20-query endpoint, qEUBO reduced mean normalised regret by 25.1% relative to a strengthened per-cell best-of-three static comparator. Against the same GP model using random pair acquisition, qEUBO also produced significantly lower regret after only five and ten comparisons, isolating a query-selection advantage rather than a benefit from model fitting alone. The manuscript reports negative and heterogeneous results transparently, including a music-intelligibility condition in which the static comparator remained superior.

The implementation, frozen protocol, compact result summaries and complete LaTeX manuscript sources are publicly available at https://github.com/alanvallavaraj/AudioTwin. The study is computational and does not claim clinical efficacy; human-listener validation is explicitly identified as future work.

This manuscript is original, is not under consideration elsewhere, and the author has approved its submission. There are no known competing interests.

Yours sincerely,

Dr Alan Immanuel Benjamin Vallavaraj
School of Computer Science and Engineering
University of Westminster, London, United Kingdom
