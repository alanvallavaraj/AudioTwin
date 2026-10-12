from scripts.oracle_robustness_dev import one


def test_high_audiogram_error_infeasible_listener_is_recorded_not_relaxed():
    result = one((2, "high_audiogram_error"))
    assert result["scenario"] == "high_audiogram_error"
    assert result["task_id"] == 2
    assert result["feasible"] is False
    assert result["rows"] == []
