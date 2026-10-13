# Cover letter — Machine Learning with Applications

Dear Editor,

Please consider the manuscript **“AudioTwin: Safety-Constrained Preferential Bayesian Optimisation for Query-Efficient Personalised Hearing-Aid Fitting”** for publication as a Research Article in *Machine Learning with Applications*.

The manuscript addresses a recurring applied machine-learning problem: personalisation when the objective is an unknown human preference, feedback is limited to pairwise choices, and every candidate decision must stay within hard feasibility constraints. AudioTwin combines a Laplace-approximated Gaussian-process preference model with qEUBO acquisition in a continuous six-dimensional constrained domain. Hearing-aid fitting is the application, but the underlying problem—query-efficient constrained preference optimisation—arises across interactive ML.

The final evaluation of 128 simulated listeners was frozen before the reserved test seed was exposed. At the pre-specified 20-query endpoint, qEUBO reduced mean normalised regret by 25.1% relative to the per-cell best static half-gain fitting, and improved on it for 94 of 128 listeners. Against the same Gaussian-process model with random pair selection, qEUBO gave lower regret at the pre-specified five- and ten-comparison budgets, isolating a benefit of query selection rather than of model fitting alone.

We have tried to report the limits of the evidence as clearly as the gains. The manuscript states that the simulator's prescription is a half-gain rule and that two of the three recorded static settings are utility-equivalent in the primary analysis. It presents a music–intelligibility condition in which static fitting remained better, together with an analysis of why. It also shows, in clearly labelled exploratory development-seed checks, that the advantage over static fitting depends on preference-response noise.

The implementation, the complete released final dataset, analysis scripts that reproduce every reported statistic, the original protocol history and the LaTeX sources are publicly available at https://github.com/alanvallavaraj/AudioTwin. The study is computational and makes no claim of clinical efficacy; human-listener validation is identified as the next step.

This manuscript is original, is not under consideration elsewhere, and the author has approved its submission. There are no known competing interests.

Yours sincerely,

Dr Alan Immanuel Benjamin Vallavaraj
School of Computer Science and Engineering
University of Westminster, London, United Kingdom
