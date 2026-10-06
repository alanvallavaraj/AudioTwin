import numpy as np
import pandas as pd
import pytest

from eval.stats import (bootstrap_ci, compare_to_baseline, holm, paired_wilcoxon,
                        rank_biserial_paired)


def test_holm_known_values():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert np.all(holm([0.5, 0.9]) <= 1.0)


def test_bootstrap_ci_covers_mean_and_is_seeded():
    x = np.random.default_rng(0).normal(1.0, 1.0, 200)
    lo, hi = bootstrap_ci(x, seed=1)
    assert lo < x.mean() < hi
    assert bootstrap_ci(x, seed=1) == (lo, hi)
    assert (hi - lo) < 0.5


def test_rank_biserial_extremes():
    assert rank_biserial_paired([1, 2, 3, 4]) == 1.0
    assert rank_biserial_paired([-1, -2, -3]) == -1.0
    assert rank_biserial_paired([0, 0]) == 0.0


def test_paired_wilcoxon_detects_shift_and_handles_ties():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 40)
    _, p = paired_wilcoxon(a + 1.0, a)
    assert p < 1e-4
    assert paired_wilcoxon(a, a) == (0.0, 1.0)


def test_compare_to_baseline_table():
    rng = np.random.default_rng(1)
    rows = []
    for lid in range(30):
        base = rng.normal(0.5, 0.05)
        for seed in range(3):
            rows.append(dict(listener=lid, method="prescription", s=base + rng.normal(0, 0.01)))
            rows.append(dict(listener=lid, method="good", s=base + 0.1 + rng.normal(0, 0.01)))
            rows.append(dict(listener=lid, method="same", s=base + rng.normal(0, 0.01)))
    t = compare_to_baseline(pd.DataFrame(rows), "s")
    good = t[t.method == "good"].iloc[0]
    same = t[t.method == "same"].iloc[0]
    assert good.n_units == 30                       # seeds averaged within listener
    assert good.mean_diff == pytest.approx(0.1, abs=0.02)
    assert good.p_holm < 0.01 and good.effect_rb > 0.9
    assert same.p_holm > 0.05
    lower = compare_to_baseline(pd.DataFrame(rows), "s", higher_is_better=False)
    assert lower[lower.method == "good"].iloc[0].mean_diff < 0
