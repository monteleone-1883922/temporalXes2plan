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
from models import FiringStep, TraceExecution, PetriNetLog, XorSplitStats


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
