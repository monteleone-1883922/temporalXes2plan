"""
Tests for parsing.probability_estimator.ProbabilityEstimator.

_count_branches_from_replay, _normalize_branch_counts, and
_get_activity_name_for_transition are tested with fully synthetic Petri net
components and fabricated replay results — no real event log or pm4py
discovery algorithm is involved.

_replay_log is tested with unittest.mock.patch to replace pm4py's conformance
checking with controlled fake results.
"""
import pytest
from collections import defaultdict
from typing import Set, TypedDict
from unittest.mock import patch

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place, _arc
from parsing.probability_estimator import ProbabilityEstimator
from models import AnalysisConfig


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

    Returns all components in a TypedDict for clear access in tests.
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


def _make_estimator(arcs, initial_marking, silent_transitions=None, config=None):
    return ProbabilityEstimator(
        petrinet=PetriNet("test"),
        initial_marking=initial_marking,
        final_marking=Marking(),
        edges=arcs,
        silent_transitions=silent_transitions or {},
        config=config or AnalysisConfig(),
    )


def _fake_replay(fitnesses):
    """Produce synthetic replay results with the given fitness scores."""
    return [{"trace_fitness": f, "activated_transitions": []} for f in fitnesses]


# ===========================================================================
# __init__ — arc map construction
# ===========================================================================

class TestInit:
    def test_trans_inputs_built_from_place_to_transition_arcs(self):
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        assert net["p_in"]  in est._trans_inputs[net["t_src"]]
        assert net["p_xor"] in est._trans_inputs[net["t_b"]]
        assert net["p_xor"] in est._trans_inputs[net["t_c"]]

    def test_trans_outputs_built_from_transition_to_place_arcs(self):
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        assert net["p_xor"]   in est._trans_outputs[net["t_src"]]
        assert net["p_out_b"] in est._trans_outputs[net["t_b"]]
        assert net["p_out_c"] in est._trans_outputs[net["t_c"]]


# ===========================================================================
# _count_branches_from_replay
# ===========================================================================

class TestCountBranchesFromReplay:
    def test_basic_xor_branch_counts_match_replay(self):
        """B chosen 2×, C chosen 1× → counts reflect this."""
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        replay_results = [
            {"activated_transitions": [net["t_src"], net["t_b"]]},
            {"activated_transitions": [net["t_src"], net["t_b"]]},
            {"activated_transitions": [net["t_src"], net["t_c"]]},
        ]
        counts = est._count_branches_from_replay(replay_results, decision_points)

        assert counts[net["p_xor"]][net["t_b"]] == 2
        assert counts[net["p_xor"]][net["t_c"]] == 1

    def test_branch_not_counted_when_xor_place_has_no_token(self):
        """
        If t_src never fires, p_xor never receives a token.
        t_b firing in that state must not be counted for p_xor.
        """
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        replay_results = [{"activated_transitions": [net["t_b"]]}]
        counts = est._count_branches_from_replay(replay_results, decision_points)

        assert counts[net["p_xor"]][net["t_b"]] == 0
        assert counts[net["p_xor"]][net["t_c"]] == 0

    def test_marking_resets_to_initial_between_traces(self):
        """
        Each trace must start from initial_marking independently.
        Two traces each firing t_src then t_b should yield count 2, not 1.
        """
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        replay_results = [
            {"activated_transitions": [net["t_src"], net["t_b"]]},
            {"activated_transitions": [net["t_src"], net["t_b"]]},
        ]
        counts = est._count_branches_from_replay(replay_results, decision_points)

        assert counts[net["p_xor"]][net["t_b"]] == 2

    def test_empty_replay_results_yield_zero_counts(self):
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        counts = est._count_branches_from_replay([], decision_points)

        assert counts[net["p_xor"]][net["t_b"]] == 0
        assert counts[net["p_xor"]][net["t_c"]] == 0


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
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 3, 1)
        result = est._normalize_branch_counts(branch_counts, decision_points)

        probs = result[net["p_xor"].name]
        assert probs["b"] == 0.75
        assert probs["c"] == 0.25

    def test_probabilities_sum_to_approximately_one(self):
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 7, 3)
        result = est._normalize_branch_counts(branch_counts, decision_points)

        total = sum(result[net["p_xor"].name].values())
        assert abs(total - 1.0) < 0.02

    def test_zero_counts_produce_equal_probability_fallback(self):
        net = _xor_net()
        est = _make_estimator(net["arcs"], Marking({net["p_in"]: 1}))
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = self._counts(net["p_xor"], net["t_b"], net["t_c"], 0, 0)
        result = est._normalize_branch_counts(branch_counts, decision_points)

        probs = result[net["p_xor"].name]
        assert probs["b"] == 0.5
        assert probs["c"] == 0.5


# ===========================================================================
# _get_activity_name_for_transition
# ===========================================================================

class TestGetActivityNameForTransition:
    def _est(self, silent=None):
        return _make_estimator(set(), Marking(), silent_transitions=silent or {})

    def test_labeled_transition_returns_sanitized_name(self):
        t = _transition("t1", "ER Registration")
        assert self._est()._get_activity_name_for_transition(t) == "er_registration"

    def test_silent_transition_returns_known_tau_name(self):
        t = _transition("t1", None)
        name = self._est({t: "tau_5"})._get_activity_name_for_transition(t)
        assert name == "tau_5"

    def test_silent_transition_not_in_dict_returns_tau_unknown_fallback(self):
        t = _transition("t1", None)
        name = self._est()._get_activity_name_for_transition(t)
        assert name.startswith("tau_unknown_")


# ===========================================================================
# _replay_log — fitness filtering (pm4py call is mocked)
# ===========================================================================

class TestReplayLog:
    def _est(self, config=None):
        net = _xor_net()
        return _make_estimator(net["arcs"], Marking({net["p_in"]: 1}), config=config)

    def test_all_traces_above_threshold_are_kept(self):
        with patch("pm4py.conformance_diagnostics_token_based_replay",
                   return_value=_fake_replay([1.0, 0.9, 0.85])):
            result = self._est()._replay_log(None)
        assert len(result) == 3

    def test_traces_below_default_threshold_are_filtered(self):
        with patch("pm4py.conformance_diagnostics_token_based_replay",
                   return_value=_fake_replay([1.0, 0.5, 0.3])):
            result = self._est()._replay_log(None)
        assert len(result) == 1
        assert result[0]["trace_fitness"] == 1.0

    def test_custom_threshold_in_config_is_respected(self):
        config = AnalysisConfig(replay_min_fitness=0.5)
        with patch("pm4py.conformance_diagnostics_token_based_replay",
                   return_value=_fake_replay([1.0, 0.6, 0.4])):
            result = self._est(config)._replay_log(None)
        assert len(result) == 2
