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
    sequence_alignment_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan_result(
    success=True,
    solvability="solved",
    duration_s=120.0,
    cost=5.0,
    search_time_s=10.0,
):
    r = MagicMock()
    r.success = success
    r.solvability = solvability
    r.duration_s = duration_s
    r.cost = cost
    r.search_time_s = search_time_s
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
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(success=True), ps)
        assert m["solved"] is True

    def test_q1_solved_false_when_failure(self):
        ps = _make_prefix_sample()
        m = q1_metrics(
            _make_plan_result(success=False, solvability="unsolvable_structural", duration_s=None, cost=None), ps,
        )
        assert m["solved"] is False

    def test_q1_solvability_propagated(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(solvability="timeout"), ps)
        assert m["solvability"] == "timeout"

    def test_q1_weighted_objective_computed(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(duration_s=100.0, cost=10.0), ps, cost_weight=0.001)
        assert m["weighted_objective"] == pytest.approx(100.01)

    def test_q1_weighted_objective_none_when_no_plan(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(success=False, duration_s=None, cost=None), ps)
        assert m["weighted_objective"] is None

    def test_q1_plan_time_s_propagated(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(duration_s=77.5), ps)
        assert m["plan_time_s"] == pytest.approx(77.5)

    def test_q1_plan_cost_propagated(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(cost=42.0), ps)
        assert m["plan_cost"] == pytest.approx(42.0)

    def test_q1_custom_cost_weight(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(duration_s=100.0, cost=100.0), ps, cost_weight=0.5)
        assert m["weighted_objective"] == pytest.approx(150.0)

    def test_q1_budget_s_equals_remaining(self):
        ps = _make_prefix_sample(prefix_duration_s=1000.0, full_duration_s=4000.0)
        m = q1_metrics(_make_plan_result(), ps)
        assert m["budget_s"] == pytest.approx(3000.0)

    def test_q1_budget_s_none_when_no_timestamps(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(), ps)
        assert m["budget_s"] is None

    def test_q1_within_budget_true_when_plan_time_le_budget(self):
        ps = _make_prefix_sample(prefix_duration_s=1000.0, full_duration_s=4000.0)
        m = q1_metrics(_make_plan_result(duration_s=2000.0), ps)
        assert m["within_budget"] is True

    def test_q1_within_budget_false_when_over(self):
        ps = _make_prefix_sample(prefix_duration_s=3500.0, full_duration_s=4000.0)
        m = q1_metrics(_make_plan_result(duration_s=600.0), ps)
        assert m["within_budget"] is False

    def test_q1_search_time_s_propagated(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(search_time_s=3.2), ps)
        assert m["search_time_s"] == pytest.approx(3.2)

    def test_q1_search_time_s_none_when_unavailable(self):
        ps = _make_prefix_sample()
        m = q1_metrics(_make_plan_result(search_time_s=None), ps)
        assert m["search_time_s"] is None


# ---------------------------------------------------------------------------
# q2_metrics
# ---------------------------------------------------------------------------

class TestQ2Metrics:
    def test_q2_search_time_s_propagated(self):
        ps = _make_prefix_sample()
        m = q2_metrics(_make_plan_result(search_time_s=4.4), ps)
        assert m["search_time_s"] == pytest.approx(4.4)

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
    def test_q3_search_time_s_propagated(self):
        ps = _make_prefix_sample(prefix_duration_s=0.0, full_duration_s=3600.0)
        m = q3_metrics(_make_plan_result(search_time_s=5.5), ps, [[]], ground_truth_reachable=True)
        assert m["search_time_s"] == pytest.approx(5.5)

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


# ---------------------------------------------------------------------------
# sequence_alignment_score
# ---------------------------------------------------------------------------

class TestSequenceAlignmentScore:
    def test_identical_sequences_score_one(self):
        seq = ["a", "b", "c"]
        assert sequence_alignment_score(seq, seq) == pytest.approx(1.0)

    def test_completely_different_sequences_score_zero(self):
        assert sequence_alignment_score(["a", "b"], ["x", "y"]) == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = sequence_alignment_score(["a", "b", "c"], ["a", "b", "x"])
        assert 0.0 < score < 1.0

    def test_both_empty_scores_one(self):
        assert sequence_alignment_score([], []) == pytest.approx(1.0)

    def test_one_empty_scores_zero(self):
        assert sequence_alignment_score(["a"], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sequence_alignment_score with REAL naming conventions
#
# sequence_alignment_score() itself does exact list-element comparison
# (difflib.SequenceMatcher), so its inputs must already speak the same
# "vocabulary". run_evaluation.py feeds it:
#   - result.plan_steps      <- planning.optic._parse_optic_stdout()
#   - suffix_activities      <- core_utils.sanitize_name(event["concept:name"])
# These tests derive plan_steps by running the real optic._parse_optic_stdout()
# on realistic OPTIC stdout (see tests/planning/test_optic.py::
# TestParseOpticStdoutActionNames for the naming-convention breakdown), so
# the two vocabularies are checked as they actually line up end to end, not
# just that the scoring function is correct in isolation on hand-picked
# strings.
# ---------------------------------------------------------------------------

class TestSequenceAlignmentScoreRealNaming:
    def test_perfect_behavioral_match_scores_high(self):
        """A plan that executes exactly the activities the trace suffix
        shows, in the same order, should score close to 1.0 — this is the
        baseline sanity check the feature exists for."""
        import planning.optic as optic_module

        stdout = (
            ";;; Solution Found\n"
            "0.000: (execute_admit_patient) [3.000]\n"
            "3.001: (execute_run_lab_tests_v2) [5.000]\n"
            "Time 0.01\n"
        )
        _, plan_steps, _ = optic_module._parse_optic_stdout(stdout)
        suffix_activities = ["admit_patient", "run_lab_tests"]

        assert sequence_alignment_score(plan_steps, suffix_activities) == pytest.approx(1.0)

    def test_silent_tau_actions_do_not_pollute_the_score(self):
        """Silent/tau actions are internal token-flow bookkeeping (see
        encoding/domain_builder.py::_build_prepared_tau_action) and are never
        present in the log — they must not appear in plan_steps at all, or
        they'll be scored as spurious/missing activities."""
        import planning.optic as optic_module

        stdout = (
            ";;; Solution Found\n"
            "0.000: (execute_admit_patient) [3.000]\n"
            "3.000: (execute_tau_1) [0.001]\n"
            "3.001: (execute_run_lab_tests) [5.000]\n"
            "Time 0.01\n"
        )
        _, plan_steps, _ = optic_module._parse_optic_stdout(stdout)
        suffix_activities = ["admit_patient", "run_lab_tests"]

        assert sequence_alignment_score(plan_steps, suffix_activities) == pytest.approx(1.0)

    def test_variant_suffix_does_not_create_a_false_mismatch(self):
        """Two variants of the same activity (_v1/_v2) must be counted as
        the same activity for alignment purposes — the log has no concept
        of PDDL variants, only activity names."""
        import planning.optic as optic_module

        stdout = ";;; Solution Found\n0.000: (execute_run_lab_tests_v2) [5.000]\nTime 0.01\n"
        _, plan_steps, _ = optic_module._parse_optic_stdout(stdout)
        suffix_activities = ["run_lab_tests"]

        assert sequence_alignment_score(plan_steps, suffix_activities) == pytest.approx(1.0)
