"""Unit tests for evaluation.metrics_collector."""
from unittest.mock import MagicMock

import pytest

from evaluation.metrics_collector import (
    _weighted_objective,
    _within_budget,
    aggregate,
    q1_metrics,
    q2_metrics,
    q3_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan_result(
    success=True,
    solvability="solved",
    duration_s=120.0,
    cost=5.0,
):
    r = MagicMock()
    r.success = success
    r.solvability = solvability
    r.duration_s = duration_s
    r.cost = cost
    r.plan_steps = []
    return r


def _make_prefix_sample(prefix_duration_s=None, full_duration_s=None):
    ps = MagicMock()
    ps.prefix_duration_s = prefix_duration_s
    ps.full_duration_s = full_duration_s
    return ps


# ---------------------------------------------------------------------------
# q1_metrics
# ---------------------------------------------------------------------------

class TestQ1Metrics:
    def test_q1_solved_true_when_success(self):
        m = q1_metrics(_make_plan_result(success=True))
        assert m["solved"] is True

    def test_q1_solved_false_when_failure(self):
        m = q1_metrics(_make_plan_result(success=False, solvability="unsolvable_structural", duration_s=None, cost=None))
        assert m["solved"] is False

    def test_q1_solvability_propagated(self):
        m = q1_metrics(_make_plan_result(solvability="timeout"))
        assert m["solvability"] == "timeout"

    def test_q1_weighted_objective_computed(self):
        m = q1_metrics(_make_plan_result(duration_s=100.0, cost=10.0), cost_weight=0.001)
        assert m["weighted_objective"] == pytest.approx(100.01)

    def test_q1_weighted_objective_none_when_no_plan(self):
        m = q1_metrics(_make_plan_result(success=False, duration_s=None, cost=None))
        assert m["weighted_objective"] is None

    def test_q1_plan_time_s_propagated(self):
        m = q1_metrics(_make_plan_result(duration_s=77.5))
        assert m["plan_time_s"] == pytest.approx(77.5)

    def test_q1_plan_cost_propagated(self):
        m = q1_metrics(_make_plan_result(cost=42.0))
        assert m["plan_cost"] == pytest.approx(42.0)

    def test_q1_custom_cost_weight(self):
        m = q1_metrics(_make_plan_result(duration_s=100.0, cost=100.0), cost_weight=0.5)
        assert m["weighted_objective"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# q2_metrics
# ---------------------------------------------------------------------------

class TestQ2Metrics:
    def test_q2_within_budget_true_when_plan_time_le_budget(self):
        ps = _make_prefix_sample(prefix_duration_s=1000.0, full_duration_s=4000.0)
        m = q2_metrics(_make_plan_result(duration_s=2000.0), ps)
        assert m["within_budget"] is True

    def test_q2_within_budget_false_when_over(self):
        ps = _make_prefix_sample(prefix_duration_s=3500.0, full_duration_s=4000.0)
        m = q2_metrics(_make_plan_result(duration_s=600.0), ps)
        assert m["within_budget"] is False

    def test_q2_within_budget_none_when_no_budget(self):
        ps = _make_prefix_sample()  # no timestamps
        m = q2_metrics(_make_plan_result(duration_s=100.0), ps)
        assert m["within_budget"] is None

    def test_q2_budget_s_equals_remaining(self):
        ps = _make_prefix_sample(prefix_duration_s=1000.0, full_duration_s=4000.0)
        m = q2_metrics(_make_plan_result(), ps)
        assert m["budget_s"] == pytest.approx(3000.0)

    def test_q2_budget_s_none_when_no_timestamps(self):
        ps = _make_prefix_sample()
        m = q2_metrics(_make_plan_result(), ps)
        assert m["budget_s"] is None

    def test_q2_weighted_objective_computed(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q2_metrics(_make_plan_result(duration_s=50.0, cost=10.0), ps, cost_weight=0.001)
        assert m["weighted_objective"] == pytest.approx(50.01)

    def test_q2_solved_propagated(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q2_metrics(_make_plan_result(success=False, solvability="timeout", duration_s=None, cost=None), ps)
        assert m["solved"] is False
        assert m["solvability"] == "timeout"


# ---------------------------------------------------------------------------
# q3_metrics
# ---------------------------------------------------------------------------

class TestQ3Metrics:
    def test_q3_correct_true_when_solved_matches_reachable_true(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(_make_plan_result(success=True), ps, [[]], ground_truth_reachable=True)
        assert m["correct"] is True

    def test_q3_correct_true_when_both_false(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(
            _make_plan_result(success=False, solvability="unsolvable_structural", duration_s=None, cost=None),
            ps, [[]], ground_truth_reachable=False,
        )
        assert m["correct"] is True

    def test_q3_correct_false_when_mismatch_solved_not_reachable(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(_make_plan_result(success=True), ps, [[]], ground_truth_reachable=False)
        assert m["correct"] is False

    def test_q3_correct_false_when_mismatch_reachable_not_solved(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(
            _make_plan_result(success=False, solvability="unsolvable_structural", duration_s=None, cost=None),
            ps, [[]], ground_truth_reachable=True,
        )
        assert m["correct"] is False

    def test_q3_ground_truth_reachable_propagated(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(_make_plan_result(), ps, [[]], ground_truth_reachable=True)
        assert m["ground_truth_reachable"] is True

    def test_q3_within_budget_present(self):
        ps = _make_prefix_sample(prefix_duration_s=1000.0, full_duration_s=4000.0)
        m = q3_metrics(_make_plan_result(duration_s=2500.0), ps, [[]], ground_truth_reachable=True)
        assert "within_budget" in m

    def test_q3_weighted_objective_present(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(_make_plan_result(duration_s=60.0, cost=5.0), ps, [[]], ground_truth_reachable=True)
        assert m["weighted_objective"] is not None


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_aggregate_mean_std(self):
        data = [
            {"plan_time_s": 10.0, "plan_cost": 1.0},
            {"plan_time_s": 20.0, "plan_cost": 3.0},
            {"plan_time_s": 30.0, "plan_cost": 5.0},
        ]
        result = aggregate(data)
        assert result["plan_time_s"]["mean"] == pytest.approx(20.0)
        assert result["plan_cost"]["mean"] == pytest.approx(3.0)
        assert result["plan_time_s"]["std"] > 0

    def test_aggregate_median(self):
        data = [
            {"plan_time_s": 10.0},
            {"plan_time_s": 20.0},
            {"plan_time_s": 100.0},
        ]
        result = aggregate(data)
        assert result["plan_time_s"]["median"] == pytest.approx(20.0)

    def test_aggregate_handles_none_values(self):
        data = [
            {"plan_time_s": 10.0},
            {"plan_time_s": None},
            {"plan_time_s": 30.0},
        ]
        result = aggregate(data)
        # None excluded → mean of [10, 30] = 20
        assert result["plan_time_s"]["mean"] == pytest.approx(20.0)

    def test_aggregate_all_none_field_excluded(self):
        data = [
            {"plan_time_s": None},
            {"plan_time_s": None},
        ]
        result = aggregate(data)
        assert "plan_time_s" not in result

    def test_aggregate_non_numeric_fields_excluded(self):
        data = [
            {"solvability": "solved", "plan_time_s": 10.0},
            {"solvability": "timeout", "plan_time_s": 20.0},
        ]
        result = aggregate(data)
        assert "solvability" not in result
        assert "plan_time_s" in result

    def test_aggregate_empty_list_returns_empty(self):
        assert aggregate([]) == {}

    def test_aggregate_single_entry_std_zero(self):
        data = [{"plan_time_s": 42.0}]
        result = aggregate(data)
        assert result["plan_time_s"]["mean"] == pytest.approx(42.0)
        assert result["plan_time_s"]["std"] == pytest.approx(0.0)

    def test_aggregate_bool_fields_excluded(self):
        data = [
            {"solved": True, "plan_time_s": 10.0},
            {"solved": False, "plan_time_s": 20.0},
        ]
        result = aggregate(data)
        assert "solved" not in result


# ---------------------------------------------------------------------------
# _weighted_objective / _within_budget
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_weighted_objective_none_time(self):
        assert _weighted_objective(None, 5.0, 0.001) is None

    def test_weighted_objective_none_cost_treated_as_zero(self):
        assert _weighted_objective(100.0, None, 0.001) == pytest.approx(100.0)

    def test_weighted_objective_computed(self):
        assert _weighted_objective(100.0, 10.0, 0.5) == pytest.approx(105.0)

    def test_within_budget_none_when_time_none(self):
        assert _within_budget(None, 3600.0) is None

    def test_within_budget_none_when_budget_none(self):
        assert _within_budget(100.0, None) is None

    def test_within_budget_true_at_exactly_budget(self):
        assert _within_budget(3600.0, 3600.0) is True

    def test_within_budget_false_when_over(self):
        assert _within_budget(3601.0, 3600.0) is False
