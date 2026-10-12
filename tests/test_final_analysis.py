import numpy as np
import pandas as pd
import pytest

from scripts.analyze_final import bootstrap_mean_ci, load_results, paired_delta_vs_best_static, validate_final_results


def _row(listener, context, family, method, q, regret):
    return {
        'listener_id': listener, 'context': context, 'family': family,
        'method': method, 'n_queries': q, 'regret': regret,
    }


def test_paired_delta_uses_best_static_per_cell_then_averages_within_listener():
    rows = []
    # Listener 0: the winning static baseline changes across cells.
    for method, vals in {
        'prescription': (0.20, 0.80),
        'prescription_linear': (0.40, 0.10),
        'prescription_slow': (0.30, 0.70),
    }.items():
        rows += [_row(0, 'c1', 'balanced', method, 0, vals[0]),
                 _row(0, 'c2', 'balanced', method, 0, vals[1])]
    rows += [_row(0, 'c1', 'balanced', 'audiotwin_qeubo', 20, 0.15),
             _row(0, 'c2', 'balanced', 'audiotwin_qeubo', 20, 0.20)]

    df = pd.DataFrame(rows)
    delta = paired_delta_vs_best_static(df, 'audiotwin_qeubo', 20, families=['balanced'])
    # Best static per cell = (0.20, 0.10), mean 0.15. qEUBO mean = 0.175.
    assert list(delta.index) == [0]
    assert np.isclose(delta.iloc[0], 0.025)


def test_bootstrap_ci_is_reproducible_for_fixed_analysis_seed():
    x = np.array([-0.2, -0.1, 0.0, 0.1])
    a = bootstrap_mean_ci(x, n_resamples=1000, seed=20261011)
    b = bootstrap_mean_ci(x, n_resamples=1000, seed=20261011)
    assert a == b


def _complete_structure(n_listeners=2):
    rows = []
    contexts = ("noisy_conversation", "lecture", "music")
    families = ("intelligibility", "comfort", "balanced", "noise_sensitive")
    static = ("prescription", "prescription_linear", "prescription_slow")
    learned = ("audiotwin_qeubo", "audiotwin_bald", "audiotwin_gprandom")
    budgets = (0, 5, 10, 20, 30, 50)
    for listener in range(n_listeners):
        for context in contexts:
            for family in families:
                for method in static:
                    rows.append({"task_id": listener, "listener_id": listener, "context": context,
                                 "family": family, "method": method, "n_queries": 0, "regret": 0.2})
                for method in learned:
                    for budget in budgets:
                        rows.append({"task_id": listener, "listener_id": listener, "context": context,
                                     "family": family, "method": method, "n_queries": budget, "regret": 0.2})
    return pd.DataFrame(rows)


def test_validate_final_results_rejects_missing_task():
    df = _complete_structure(2)
    with pytest.raises(ValueError, match="task IDs"):
        validate_final_results(df, expected_n=3)


def test_validate_final_results_rejects_missing_frozen_budget():
    df = _complete_structure(2)
    df = df[~((df.task_id == 1) & (df.method == "audiotwin_qeubo") & (df.n_queries == 50))]
    with pytest.raises(ValueError, match="frozen result grid"):
        validate_final_results(df, expected_n=2)


def test_validate_final_results_accepts_complete_frozen_grid():
    validate_final_results(_complete_structure(2), expected_n=2)


def test_validate_final_results_accepts_declared_replacement_task():
    import pandas as pd
    rows=[]
    contexts=['noisy_conversation','lecture','music']
    families=['intelligibility','comfort','balanced','noise_sensitive']
    static=['prescription','prescription_linear','prescription_slow']
    learned=['audiotwin_qeubo','audiotwin_bald','audiotwin_gprandom']
    for task in list(range(128)):
        if task == 55:
            continue
        for c in contexts:
            for f in families:
                for m in static:
                    rows.append({'task_id':task,'listener_id':task,'context':c,'family':f,'method':m,'n_queries':0,'regret':0.1,'git_hash':'x'})
                for m in learned:
                    for q in [0,5,10,20,30,50]:
                        rows.append({'task_id':task,'listener_id':task,'context':c,'family':f,'method':m,'n_queries':q,'regret':0.1,'git_hash':'x'})
    task=128
    for c in contexts:
        for f in families:
            for m in static:
                rows.append({'task_id':task,'listener_id':task,'context':c,'family':f,'method':m,'n_queries':0,'regret':0.1,'git_hash':'x'})
            for m in learned:
                for q in [0,5,10,20,30,50]:
                    rows.append({'task_id':task,'listener_id':task,'context':c,'family':f,'method':m,'n_queries':q,'regret':0.1,'git_hash':'x'})
    df=pd.DataFrame(rows)
    validate_final_results(df, replacement_map={55:128})


def test_load_results_accepts_consolidated_parquet(tmp_path):
    df = _complete_structure(2)
    path = tmp_path / "final_v1.parquet"
    df.to_parquet(path, index=False)
    loaded = load_results(path)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))
