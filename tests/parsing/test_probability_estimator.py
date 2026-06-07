"""
Tests for parsing.probability_estimator.ProbabilityEstimator.

All tests use synthetic Petri net components and manually constructed
FiringStep / TraceExecution / PetriNetLog objects — no real event log,
pm4py discovery algorithm, or token-based replay is involved.
"""
import pytest
from collections import defaultdict
from typing import Set, TypedDict

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place, _arc
from parsing.probability_estimator import ProbabilityEstimator
from models import AttributeEffect, FiringStep, TraceExecution, PetriNetLog, XorSplitStats
from parsing.discretizer import Discretizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _XorNet(TypedDict):
    t_src: PetriNet.Transition
    t_b: PetriNet.Transition
    t_c: PetriNet.Transition
    p_in: PetriNet.Place
    p_xor: PetriNet.Place
    p_out_b: PetriNet.Place
    p_out_c: PetriNet.Place
    arcs: Set[PetriNet.Arc]


def _xor_net() -> _XorNet:
    """
    Build a minimal XOR-split net with explicit component references.

    Structure:
        p_in -> t_src -> p_xor -> t_b -> p_out_b
                               -> t_c -> p_out_c
    """
    t_src = _transition("t_src", "Src")
    t_b   = _transition("t_b",   "B")
    t_c   = _transition("t_c",   "C")
    p_in    = _place("p_in")
    p_xor   = _place("p_xor")
    p_out_b = _place("p_out_b")
    p_out_c = _place("p_out_c")
    arcs: Set[PetriNet.Arc] = {
        _arc(p_in,  t_src), _arc(t_src, p_xor),
        _arc(p_xor, t_b),   _arc(t_b,   p_out_b),
        _arc(p_xor, t_c),   _arc(t_c,   p_out_c),
    }
    return _XorNet(
        t_src=t_src, t_b=t_b, t_c=t_c,
        p_in=p_in, p_xor=p_xor, p_out_b=p_out_b, p_out_c=p_out_c,
        arcs=arcs,
    )


def _make_estimator(silent_transitions=None) -> ProbabilityEstimator:
    return ProbabilityEstimator(silent_transitions=silent_transitions or {})


def _make_step(transition: PetriNet.Transition, from_places) -> FiringStep:
    return FiringStep(
        transition=transition,
        activity_name=transition.label.lower() if transition.label else "tau",
        is_tau=transition.label is None,
        from_places=set(from_places),
        attributes={},
    )


def _make_log(executions) -> PetriNetLog:
    return PetriNetLog(
        executions=executions,
        net=PetriNet("test"),
        initial_marking=Marking(),
        final_marking=Marking(),
    )


# ===========================================================================
# compute_from_petri_net_log
# ===========================================================================

class TestComputeFromPetriNetLog:
    def test_branch_counts_reflect_from_places(self):
        """B chosen 2×, C chosen 1× — probabilities must reflect 2:1 ratio."""
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        executions = [
            TraceExecution("c1", [_make_step(net["t_src"], {net["p_in"]}),
                                   _make_step(net["t_b"],  {net["p_xor"]})]),
            TraceExecution("c2", [_make_step(net["t_src"], {net["p_in"]}),
                                   _make_step(net["t_b"],  {net["p_xor"]})]),
            TraceExecution("c3", [_make_step(net["t_src"], {net["p_in"]}),
                                   _make_step(net["t_c"],  {net["p_xor"]})]),
        ]
        result = _make_estimator().compute_from_petri_net_log(
            _make_log(executions), decision_points
        )

        stats = result[net["p_xor"].name]
        assert abs(stats.probabilities["b"] - round(2 / 3, 2)) < 0.01
        assert abs(stats.probabilities["c"] - round(1 / 3, 2)) < 0.01

    def test_total_executions_equals_number_of_xor_firings(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        executions = [
            TraceExecution("c1", [_make_step(net["t_b"], {net["p_xor"]})]),
            TraceExecution("c2", [_make_step(net["t_b"], {net["p_xor"]})]),
            TraceExecution("c3", [_make_step(net["t_c"], {net["p_xor"]})]),
        ]
        result = _make_estimator().compute_from_petri_net_log(
            _make_log(executions), decision_points
        )

        assert result[net["p_xor"].name].total_executions == 3

    def test_step_not_counted_when_xor_place_absent_from_from_places(self):
        """t_b fires but p_xor is not in from_places — no count for p_xor."""
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        executions = [
            TraceExecution("c1", [_make_step(net["t_b"], {net["p_in"]})]),
        ]
        result = _make_estimator().compute_from_petri_net_log(
            _make_log(executions), decision_points
        )

        assert result[net["p_xor"].name].total_executions == 0

    def test_empty_log_yields_equal_probability_fallback(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        result = _make_estimator().compute_from_petri_net_log(
            _make_log([]), decision_points
        )

        stats = result[net["p_xor"].name]
        assert stats.probabilities["b"] == 0.5
        assert stats.probabilities["c"] == 0.5
        assert stats.total_executions == 0

    def test_returns_xor_split_stats_instance(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}
        executions = [TraceExecution("c1", [_make_step(net["t_b"], {net["p_xor"]})])]

        result = _make_estimator().compute_from_petri_net_log(
            _make_log(executions), decision_points
        )

        assert isinstance(result[net["p_xor"].name], XorSplitStats)


# ===========================================================================
# _normalize_branch_counts
# ===========================================================================

class TestNormalizeBranchCounts:
    def _counts(self, place, t_b, t_c, n_b, n_c):
        d = defaultdict(int)
        d[t_b] = n_b
        d[t_c] = n_c
        return {place: d}

    def test_probabilities_reflect_branch_counts(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 3, 1)
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        stats = result[net["p_xor"].name]
        assert isinstance(stats, XorSplitStats)
        assert stats.probabilities["b"] == 0.75
        assert stats.probabilities["c"] == 0.25

    def test_total_executions_equals_sum_of_branch_counts(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 3, 1)
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        assert result[net["p_xor"].name].total_executions == 4

    def test_probabilities_sum_to_approximately_one(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 7, 3)
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        total = sum(result[net["p_xor"].name].probabilities.values())
        assert abs(total - 1.0) < 0.02

    def test_zero_counts_produce_equal_probability_fallback(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 0, 0)
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        stats = result[net["p_xor"].name]
        assert stats.probabilities["b"] == 0.5
        assert stats.probabilities["c"] == 0.5
        assert stats.total_executions == 0


# ===========================================================================
# _get_activity_name_for_transition
# ===========================================================================

class TestGetActivityNameForTransition:
    def test_labeled_transition_returns_sanitized_name(self):
        t = _transition("t1", "ER Registration")
        assert _make_estimator()._get_activity_name_for_transition(t) == "er_registration"

    def test_silent_transition_returns_known_tau_name(self):
        t = _transition("t1", None)
        name = _make_estimator({t: "tau_5"})._get_activity_name_for_transition(t)
        assert name == "tau_5"

    def test_silent_transition_not_in_dict_returns_tau_unknown_fallback(self):
        t = _transition("t1", None)
        name = _make_estimator()._get_activity_name_for_transition(t)
        assert name.startswith("tau_unknown_")


# ===========================================================================
# compute_attribute_effect_probabilities
# ===========================================================================

def _labeled_step(activity_name: str, attrs: dict) -> FiringStep:
    """Labeled FiringStep with the given activity name and attributes."""
    t = _transition(f"t_{activity_name}", activity_name)
    return FiringStep(
        transition=t,
        activity_name=activity_name,
        is_tau=False,
        from_places=set(),
        attributes=attrs,
    )


def _tau_step() -> FiringStep:
    """Silent FiringStep with empty attributes."""
    t = _transition("t_tau", None)
    return FiringStep(
        transition=t,
        activity_name="tau_1",
        is_tau=True,
        from_places=set(),
        attributes={},
    )


class TestComputeAttributeEffectProbabilities:
    def test_attribute_appearing_for_first_time_is_counted_as_effect(self):
        """First execution of A with crp=2.1 — crp was never in state, counts as effect."""
        executions = [TraceExecution("c1", [_labeled_step("a", {"crp": 2.1})])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert result["a"].presence_probabilities["crp"] == 1.0

    def test_attribute_changing_value_is_counted_as_effect(self):
        """A fires with crp=2.1, then B fires with crp=8.3 — B changes crp."""
        executions = [TraceExecution("c1", [
            _labeled_step("a", {"crp": 2.1}),
            _labeled_step("b", {"crp": 8.3}),
        ])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert result["b"].presence_probabilities["crp"] == 1.0

    def test_unchanged_attribute_not_counted_as_effect(self):
        """A fires with crp=2.1, B fires with crp=2.1 — same value, no effect."""
        executions = [TraceExecution("c1", [
            _labeled_step("a", {"crp": 2.1}),
            _labeled_step("b", {"crp": 2.1}),
        ])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert "crp" not in result["b"].presence_probabilities

    def test_attribute_absent_from_step_is_not_considered(self):
        """A fires with crp=2.1, B fires with empty attributes — crp not mentioned in B."""
        executions = [TraceExecution("c1", [
            _labeled_step("a", {"crp": 2.1}),
            _labeled_step("b", {}),
        ])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert "crp" not in result["b"].presence_probabilities

    def test_value_probability_reflects_value_distribution(self):
        """B fires 3 times: crp→5.0 twice, crp→3.0 once."""
        executions = [
            TraceExecution("c1", [_labeled_step("b", {"crp": 5.0})]),
            TraceExecution("c2", [_labeled_step("b", {"crp": 5.0})]),
            TraceExecution("c3", [_labeled_step("b", {"crp": 3.0})]),
        ]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        vp = result["b"].value_probabilities["crp"]
        assert abs(vp[5.0] - round(2 / 3, 2)) < 0.01
        assert abs(vp[3.0] - round(1 / 3, 2)) < 0.01

    def test_total_firings_counts_all_labeled_step_occurrences(self):
        """A fires in 3 independent executions → total_firings = 3."""
        executions = [
            TraceExecution("c1", [_labeled_step("a", {})]),
            TraceExecution("c2", [_labeled_step("a", {})]),
            TraceExecution("c3", [_labeled_step("a", {})]),
        ]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert result["a"].total_firings == 3

    def test_tau_steps_are_excluded_from_result(self):
        """Tau transitions must not appear as keys in the result."""
        executions = [TraceExecution("c1", [_tau_step()])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert "tau_1" not in result
        assert result == {}

    def test_state_resets_between_executions(self):
        """Each execution starts with empty state — first occurrence counts as effect."""
        executions = [
            TraceExecution("c1", [_labeled_step("a", {"crp": 2.1})]),
            TraceExecution("c2", [_labeled_step("a", {"crp": 2.1})]),
        ]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert result["a"].presence_probabilities["crp"] == 1.0
        assert result["a"].total_firings == 2

    def test_empty_log_returns_empty_dict(self):
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log([]))
        assert result == {}

    def test_returns_attribute_effect_instance(self):
        executions = [TraceExecution("c1", [_labeled_step("a", {"x": 1})])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert isinstance(result["a"], AttributeEffect)

    def test_attribute_names_are_sanitized(self):
        """Attribute key 'CRP Measurement' must appear as 'crp_measurement' in the result."""
        executions = [TraceExecution("c1", [_labeled_step("a", {"CRP Measurement": 2.1})])]
        result = _make_estimator().compute_attribute_effect_probabilities(_make_log(executions))

        assert "crp_measurement" in result["a"].presence_probabilities
        assert "CRP Measurement" not in result["a"].presence_probabilities

    def test_numerical_values_discretized_when_discretizer_provided(self):
        """30.0 and 80.0 with boundary at 50.0 produce different intervals → detected as change."""
        disc = Discretizer()
        disc.boundaries = {"crp": [50.0]}
        executions = [TraceExecution("c1", [
            _labeled_step("a", {"crp": 30.0}),
            _labeled_step("b", {"crp": 80.0}),
        ])]
        result = _make_estimator().compute_attribute_effect_probabilities(
            _make_log(executions), discretizer=disc
        )

        assert result["b"].presence_probabilities["crp"] == 1.0
        assert "gte_50_0" in result["b"].value_probabilities["crp"]

    def test_values_in_same_interval_treated_as_unchanged(self):
        """30.0 and 40.0 both map to 'lte_50_0' — no change detected in B."""
        disc = Discretizer()
        disc.boundaries = {"crp": [50.0]}
        executions = [TraceExecution("c1", [
            _labeled_step("a", {"crp": 30.0}),
            _labeled_step("b", {"crp": 40.0}),
        ])]
        result = _make_estimator().compute_attribute_effect_probabilities(
            _make_log(executions), discretizer=disc
        )

        assert "crp" not in result["b"].presence_probabilities
