"""
Tests for parsing.probability_estimator.ProbabilityEstimator.

All tests use synthetic PreprocessedLog / TransitionFiringData objects —
no real event log, pm4py discovery, or token-based replay is involved.
"""
import pytest
from collections import defaultdict
from typing import Dict, List, Set, TypedDict

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place, _arc
from parsing.probability_estimator import ProbabilityEstimator
from models import (
    AnalysisConfig,
    AttributeEffect,
    PreprocessedLog,
    TransitionFiringData,
    XorSplitStats,
)


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


def _fd(activity_name: str, from_places=None, pre_state=None, changed_attrs=None) -> TransitionFiringData:
    """Build a TransitionFiringData with sensible defaults."""
    return TransitionFiringData(
        activity_name=activity_name,
        pre_state=pre_state or {},
        changed_attrs=changed_attrs or {},
        from_places=frozenset(from_places or []),
    )


def _preprocessed(
    transition_firings: Dict[str, List[TransitionFiringData]] = None,
    xor_firings: Dict[str, List[TransitionFiringData]] = None,
) -> PreprocessedLog:
    return PreprocessedLog(
        transition_firings=transition_firings or {},
        xor_firings=xor_firings or {},
        static_attributes=set(),
    )


# ===========================================================================
# compute_xor_probabilities
# ===========================================================================

class TestComputeXorProbabilities:
    def test_branch_counts_reflect_xor_firings(self):
        """B chosen 2x, C chosen 1x — probabilities must reflect 2:1 ratio."""
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        fd_b1 = _fd("B", from_places={net["p_xor"]})
        fd_b2 = _fd("B", from_places={net["p_xor"]})
        fd_c1 = _fd("C", from_places={net["p_xor"]})

        plog = _preprocessed(xor_firings={
            net["p_xor"].name: [fd_b1, fd_b2, fd_c1],
        })
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        stats = result[net["p_xor"].name]
        assert abs(stats.probabilities["B"] - round(2 / 3, 2)) < 0.01
        assert abs(stats.probabilities["C"] - round(1 / 3, 2)) < 0.01

    def test_total_executions_equals_number_of_xor_firings(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        plog = _preprocessed(xor_firings={
            net["p_xor"].name: [_fd("B"), _fd("B"), _fd("C")],
        })
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        assert result[net["p_xor"].name].total_executions == 3

    def test_place_not_in_xor_firings_yields_zero_total(self):
        """XOR place exists in decision_points but has no firings in preprocessed log."""
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        plog = _preprocessed(xor_firings={})
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        assert result[net["p_xor"].name].total_executions == 0

    def test_empty_xor_firings_yields_equal_probability_fallback(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        plog = _preprocessed(xor_firings={})
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        stats = result[net["p_xor"].name]
        assert stats.probabilities["B"] == 0.5
        assert stats.probabilities["C"] == 0.5
        assert stats.total_executions == 0

    def test_returns_xor_split_stats_instance(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        plog = _preprocessed(xor_firings={
            net["p_xor"].name: [_fd("B")],
        })
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        assert isinstance(result[net["p_xor"].name], XorSplitStats)

    def test_branch_with_zero_firings_gets_zero_probability(self):
        """Only B fires — C must still appear in result with probability 0."""
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        plog = _preprocessed(xor_firings={
            net["p_xor"].name: [_fd("B"), _fd("B")],
        })
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        stats = result[net["p_xor"].name]
        assert stats.probabilities["B"] == 1.0
        assert stats.probabilities["C"] == 0.0

    def test_rare_branch_probability_survives_rounding(self):
        """2 out of 500 firings (0.004) must not collapse to 0.0 -- rounding
        to only 2 decimals would silently make this branch appear
        impossible and, downstream, drop its -log(prob) action cost."""
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        firings = [_fd("B")] * 498 + [_fd("C")] * 2
        plog = _preprocessed(xor_firings={net["p_xor"].name: firings})
        result = _make_estimator().compute_xor_probabilities(plog, decision_points)

        stats = result[net["p_xor"].name]
        assert stats.probabilities["C"] == pytest.approx(0.004)
        assert stats.probabilities["C"] > 0.0


# ===========================================================================
# _normalize_branch_counts
# ===========================================================================

class TestNormalizeBranchCounts:
    def test_probabilities_reflect_branch_counts(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = {net["p_xor"].name: {"B": 3, "C": 1}}
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        stats = result[net["p_xor"].name]
        assert isinstance(stats, XorSplitStats)
        assert stats.probabilities["B"] == 0.75
        assert stats.probabilities["C"] == 0.25

    def test_total_executions_equals_sum_of_branch_counts(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = {net["p_xor"].name: {"B": 3, "C": 1}}
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        assert result[net["p_xor"].name].total_executions == 4

    def test_probabilities_sum_to_approximately_one(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = {net["p_xor"].name: {"B": 7, "C": 3}}
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        total = sum(result[net["p_xor"].name].probabilities.values())
        assert abs(total - 1.0) < 0.02

    def test_zero_counts_produce_equal_probability_fallback(self):
        net = _xor_net()
        decision_points = {net["p_xor"]: [net["t_b"], net["t_c"]]}

        branch_counts = {net["p_xor"].name: {}}
        result = _make_estimator()._normalize_branch_counts(branch_counts, decision_points)

        stats = result[net["p_xor"].name]
        assert stats.probabilities["B"] == 0.5
        assert stats.probabilities["C"] == 0.5
        assert stats.total_executions == 0


# ===========================================================================
# _get_activity_name_for_transition
# ===========================================================================

class TestGetActivityNameForTransition:
    def test_labeled_transition_returns_raw_label(self):
        t = _transition("t1", "ER Registration")
        assert _make_estimator()._get_activity_name_for_transition(t) == "ER Registration"

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

class TestComputeAttributeEffectProbabilities:
    def test_changed_attr_counted_as_effect(self):
        """changed_attrs contains crp — must appear in presence_probabilities."""
        plog = _preprocessed(transition_firings={
            "a": [_fd("a", changed_attrs={"crp": 2.1})],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert result["a"].presence_probabilities["crp"] == 1.0

    def test_unchanged_attr_not_counted(self):
        """Attr present in pre_state but NOT in changed_attrs → no effect."""
        plog = _preprocessed(transition_firings={
            "b": [_fd("b", pre_state={"crp": 2.1}, changed_attrs={})],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert "crp" not in result["b"].presence_probabilities

    def test_value_probability_reflects_value_distribution(self):
        """B fires 3 times: crp→5.0 twice, crp→3.0 once."""
        plog = _preprocessed(transition_firings={
            "b": [
                _fd("b", changed_attrs={"crp": 5.0}),
                _fd("b", changed_attrs={"crp": 5.0}),
                _fd("b", changed_attrs={"crp": 3.0}),
            ],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        vp = result["b"].value_probabilities["crp"]
        assert abs(vp[5.0] - round(2 / 3, 2)) < 0.01
        assert abs(vp[3.0] - round(1 / 3, 2)) < 0.01

    def test_total_firings_counts_all_firing_data_objects(self):
        """A appears 3 times in transition_firings → total_firings = 3."""
        plog = _preprocessed(transition_firings={
            "a": [_fd("a"), _fd("a"), _fd("a")],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert result["a"].total_firings == 3

    def test_empty_log_returns_empty_dict(self):
        plog = _preprocessed()
        result = _make_estimator().compute_attribute_effect_probabilities(plog)
        assert result == {}

    def test_returns_attribute_effect_instance(self):
        plog = _preprocessed(transition_firings={
            "a": [_fd("a", changed_attrs={"x": 1})],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert isinstance(result["a"], AttributeEffect)

    def test_presence_probability_partial(self):
        """crp changes in 1 of 2 firings → presence = 0.5."""
        plog = _preprocessed(transition_firings={
            "a": [
                _fd("a", changed_attrs={"crp": 2.1}),
                _fd("a", changed_attrs={}),
            ],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert result["a"].presence_probabilities["crp"] == 0.5

    def test_rare_presence_probability_survives_rounding(self):
        """crp changes in 2 of 500 firings (0.004) must not collapse to 0.0
        -- see the XOR-branch rounding regression test above for why."""
        firings = [_fd("a", changed_attrs={"crp": 2.1})] * 2
        firings += [_fd("a", changed_attrs={})] * 498
        plog = _preprocessed(transition_firings={"a": firings})
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert result["a"].presence_probabilities["crp"] == pytest.approx(0.004)
        assert result["a"].presence_probabilities["crp"] > 0.0

    def test_multiple_attrs_in_same_firing(self):
        """Two attributes change in the same firing — both counted."""
        plog = _preprocessed(transition_firings={
            "a": [_fd("a", changed_attrs={"crp": 2.1, "age": 65})],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert result["a"].presence_probabilities["crp"] == 1.0
        assert result["a"].presence_probabilities["age"] == 1.0

    def test_discretized_values_used_as_is(self):
        """PreprocessedLog already contains discretized values — estimator stores them directly."""
        plog = _preprocessed(transition_firings={
            "a": [
                _fd("a", changed_attrs={"crp": "lte_50_0"}),
                _fd("a", changed_attrs={"crp": "gte_50_0"}),
            ],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        vp = result["a"].value_probabilities["crp"]
        assert "lte_50_0" in vp
        assert "gte_50_0" in vp

    def test_multiple_transitions(self):
        """Two different transitions each get their own AttributeEffect."""
        plog = _preprocessed(transition_firings={
            "a": [_fd("a", changed_attrs={"x": 1})],
            "b": [_fd("b", changed_attrs={"y": 2}), _fd("b", changed_attrs={})],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert "a" in result and "b" in result
        assert result["a"].total_firings == 1
        assert result["b"].total_firings == 2
        assert result["a"].presence_probabilities["x"] == 1.0
        assert result["b"].presence_probabilities["y"] == 0.5

    def test_no_changed_attrs_still_counted_in_total(self):
        """Firings with no changed attrs still contribute to total_firings."""
        plog = _preprocessed(transition_firings={
            "a": [_fd("a", changed_attrs={}), _fd("a", changed_attrs={})],
        })
        result = _make_estimator().compute_attribute_effect_probabilities(plog)

        assert result["a"].total_firings == 2
        assert result["a"].presence_probabilities == {}
