"""Unit tests for evaluation.query_builder."""
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from evaluation.query_builder import (
    QuerySpec,
    _goal_from_final_event,
    _remaining_budget,
    build_q1,
    build_q2,
    build_q3,
    compute_min_time_to_end,
    is_q3_reachable,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_prefix_sample(
    init_places=None,
    init_effects=None,
    final_attrs=None,
    prefix_duration_s=None,
    full_duration_s=None,
) -> MagicMock:
    ps = MagicMock()
    ps.init_places = init_places or ["p_1"]
    ps.init_effects = init_effects or []
    ps.final_event_attributes = final_attrs or {}
    ps.prefix_duration_s = prefix_duration_s
    ps.full_duration_s = full_duration_s
    return ps


_CATALOG_MIXED = {
    "status": {
        "type": "categorical",
        "possible_values": ["admitted", "discharged"],
        "bin_boundaries": [],
    },
    "critical": {
        "type": "boolean",
        "possible_values": ["true", "false"],
        "bin_boundaries": [],
    },
    "score": {
        "type": "numerical",
        "possible_values": ["lte_50_0", "gte_50_0"],
        "bin_boundaries": [50.0],
    },
}

_SERIALIZED_EMPTY: Dict[str, Any] = {
    "graph": {"nodes": [], "edges": []},
    "transitions": {},
    "attribute_catalog": _CATALOG_MIXED,
    "metadata": {"start_place": "p_start", "end_place": "p_end"},
}


def _transition(input_places, assignments=None, duration=None):
    return {
        "input_places": input_places,
        "preconditions": [],
        "effect_groups": (
            [{"assignments": assignments, "guard": [], "probability": 1.0}]
            if assignments else []
        ),
        "cost": 0.0,
        "duration": duration,
    }


def _duration(effective_min: float, effective_max: float) -> Dict[str, float]:
    return {"effective_min": effective_min, "effective_max": effective_max}


def _serialized_with_effect(attr: str, value: str) -> Dict[str, Any]:
    """Minimal reachable net: p_start -> act_a -> p_end, act_a sets attr=value."""
    return {
        "graph": {
            "nodes": [
                {"id": "p_start", "type": "place", "label": "p_start"},
                {"id": "t_a", "type": "transition", "label": "act_a"},
                {"id": "p_end", "type": "place", "label": "p_end"},
            ],
            "edges": [
                {"source": "p_start", "target": "t_a"},
                {"source": "t_a", "target": "p_end"},
            ],
        },
        "transitions": {
            "act_a": _transition(["p_start"], [{"attribute": attr, "value": value}]),
        },
        "attribute_catalog": _CATALOG_MIXED,
        "metadata": {"start_place": "p_start", "end_place": "p_end"},
    }


def _serialized_with_unreachable_effect(attr: str, value: str) -> Dict[str, Any]:
    """act_a (sets attr=value) only fires from p_isolated, which the marking
    starting at p_start never reaches — p_start reaches p_end via act_b
    instead, whose effect does not match the goal."""
    return {
        "graph": {
            "nodes": [
                {"id": "p_start", "type": "place", "label": "p_start"},
                {"id": "t_b", "type": "transition", "label": "act_b"},
                {"id": "p_end", "type": "place", "label": "p_end"},
                {"id": "p_isolated", "type": "place", "label": "p_isolated"},
                {"id": "t_a", "type": "transition", "label": "act_a"},
                {"id": "p_after_a", "type": "place", "label": "p_after_a"},
            ],
            "edges": [
                {"source": "p_start", "target": "t_b"},
                {"source": "t_b", "target": "p_end"},
                {"source": "p_isolated", "target": "t_a"},
                {"source": "t_a", "target": "p_after_a"},
            ],
        },
        "transitions": {
            "act_b": _transition(["p_start"]),
            "act_a": _transition(["p_isolated"], [{"attribute": attr, "value": value}]),
        },
        "attribute_catalog": _CATALOG_MIXED,
        "metadata": {"start_place": "p_start", "end_place": "p_end"},
    }


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------

class TestBuildQ1:
    def test_q1_has_no_deadline(self):
        ps = _make_prefix_sample()
        spec = build_q1(ps)
        assert spec.deadline is None

    def test_q1_query_type(self):
        spec = build_q1(_make_prefix_sample())
        assert spec.query_type == "Q1"

    def test_q1_require_completion(self):
        spec = build_q1(_make_prefix_sample())
        assert spec.require_completion is True

    def test_q1_uses_minimize_weighted_metric(self):
        spec = build_q1(_make_prefix_sample())
        assert spec.metric == "minimize_weighted"

    def test_q1_cost_weight_propagated(self):
        spec = build_q1(_make_prefix_sample(), cost_weight=0.5)
        assert spec.cost_weight == pytest.approx(0.5)

    def test_q1_default_cost_weight(self):
        spec = build_q1(_make_prefix_sample())
        assert spec.cost_weight == pytest.approx(0.001)

    def test_q1_init_places_from_sample(self):
        ps = _make_prefix_sample(init_places=["p_tok"])
        spec = build_q1(ps)
        assert spec.init_places == ["p_tok"]

    def test_q1_init_effects_from_sample(self):
        effects = [{"attribute": "status", "value": "admitted"}]
        ps = _make_prefix_sample(init_effects=effects)
        spec = build_q1(ps)
        assert spec.init_effects == effects


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------

class TestBuildQ2:
    def test_q2_deadline_equals_remaining_budget(self):
        ps = _make_prefix_sample(prefix_duration_s=1800.0, full_duration_s=3600.0)
        spec = build_q2(ps)
        assert spec is not None
        assert spec.deadline == pytest.approx(1800.0)

    def test_q2_skip_when_no_timestamps(self):
        ps = _make_prefix_sample()  # both durations None
        assert build_q2(ps) is None

    def test_q2_skip_when_prefix_duration_none(self):
        ps = _make_prefix_sample(full_duration_s=3600.0)
        assert build_q2(ps) is None

    def test_q2_skip_when_full_duration_none(self):
        ps = _make_prefix_sample(prefix_duration_s=600.0)
        assert build_q2(ps) is None

    def test_q2_query_type(self):
        ps = _make_prefix_sample(prefix_duration_s=600.0, full_duration_s=3600.0)
        spec = build_q2(ps)
        assert spec is not None
        assert spec.query_type == "Q2"

    def test_q2_uses_minimize_weighted_metric(self):
        ps = _make_prefix_sample(prefix_duration_s=600.0, full_duration_s=3600.0)
        spec = build_q2(ps)
        assert spec.metric == "minimize_weighted"

    def test_q2_require_completion(self):
        ps = _make_prefix_sample(prefix_duration_s=600.0, full_duration_s=3600.0)
        spec = build_q2(ps)
        assert spec.require_completion is True

    def test_q2_cost_weight_propagated(self):
        ps = _make_prefix_sample(prefix_duration_s=600.0, full_duration_s=3600.0)
        spec = build_q2(ps, cost_weight=0.01)
        assert spec.cost_weight == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------

class TestBuildQ3:
    def test_q3_built_even_without_timestamps(self):
        # Q3 has no deadline (see test_q3_deadline_is_always_none below), so
        # it no longer needs timestamp data at all -- only the attribute
        # goal matters, which comes from replayed attribute state, not time.
        ps = _make_prefix_sample(final_attrs={"status": "discharged"})
        assert ps.prefix_duration_s is None and ps.full_duration_s is None
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec is not None
        assert spec.deadline is None

    def test_q3_returns_none_when_no_attributes(self):
        # final event has no attributes in the catalog
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"unknown_field": "x"},
        )
        assert build_q3(ps, _SERIALIZED_EMPTY) is None

    def test_q3_returns_none_when_final_attrs_empty(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={},
        )
        assert build_q3(ps, _SERIALIZED_EMPTY) is None

    def test_q3_goal_includes_final_attributes(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"status": "discharged"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec is not None
        conditions = spec.goal_sop[0]
        assert any(c["attribute"] == "status" and c["value"] == "discharged"
                   for c in conditions)

    def test_q3_query_type(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"status": "admitted"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec is not None
        assert spec.query_type == "Q3"

    def test_q3_deadline_is_always_none(self):
        # Q3 has no time budget at all -- deadline stays None regardless of
        # how much real time was available (unlike Q2, which uses it).
        ps = _make_prefix_sample(
            prefix_duration_s=900.0,
            full_duration_s=3600.0,
            final_attrs={"status": "discharged"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec is not None
        assert spec.deadline is None

    def test_q3_uses_minimize_weighted_metric(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"status": "discharged"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec.metric == "minimize_weighted"

    def test_q3_require_completion(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"status": "discharged"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec.require_completion is True

    def test_q3_cost_weight_propagated(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"status": "discharged"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY, cost_weight=0.05)
        assert spec.cost_weight == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# _remaining_budget
# ---------------------------------------------------------------------------

class TestRemainingBudget:
    def test_budget_computed_correctly(self):
        ps = _make_prefix_sample(prefix_duration_s=1000.0, full_duration_s=4000.0)
        assert _remaining_budget(ps) == pytest.approx(3000.0)

    def test_budget_none_when_both_none(self):
        assert _remaining_budget(_make_prefix_sample()) is None

    def test_budget_none_when_prefix_none(self):
        ps = _make_prefix_sample(full_duration_s=3600.0)
        assert _remaining_budget(ps) is None

    def test_budget_none_when_full_none(self):
        ps = _make_prefix_sample(prefix_duration_s=600.0)
        assert _remaining_budget(ps) is None


# ---------------------------------------------------------------------------
# _goal_from_final_event
# ---------------------------------------------------------------------------

class TestGoalFromFinalEvent:
    def test_categorical_attr_included(self):
        result = _goal_from_final_event({"status": "discharged"}, _CATALOG_MIXED)
        assert result == [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]

    def test_unknown_attr_skipped(self):
        result = _goal_from_final_event({"unknown": "x"}, _CATALOG_MIXED)
        assert result == [[]]

    def test_attr_not_in_possible_values_skipped(self):
        result = _goal_from_final_event({"status": "unknown_val"}, _CATALOG_MIXED)
        assert result == [[]]

    def test_categorical_attr_matches_original_case(self):
        """possible_values holds the raw, unsanitized values observed in the
        log (e.g. "NIL", "A") — a raw_value with the same original casing
        must match directly, without lowercasing either side."""
        catalog = {"dismissal": {
            "type": "categorical", "possible_values": ["NIL", "A"], "bin_boundaries": [],
        }}
        result = _goal_from_final_event({"dismissal": "NIL"}, catalog)
        assert result == [[{"attribute": "dismissal", "predicate": "=", "value": "NIL"}]]

    def test_boolean_attr_included(self):
        result = _goal_from_final_event({"critical": "true"}, _CATALOG_MIXED)
        assert result == [[{"attribute": "critical", "predicate": "=", "value": "true"}]]

    def test_numerical_attr_binned(self):
        # final_attrs holds values already discretized into their bin label
        # by TraceReplayer._normalize_event_value (never a raw float) — see
        # _goal_from_final_event's docstring.
        result = _goal_from_final_event({"score": "gte_50_0"}, _CATALOG_MIXED)
        conds = result[0]
        assert len(conds) == 1
        assert conds[0]["attribute"] == "score"
        assert conds[0]["value"] == "gte_50_0"

    def test_numerical_attr_first_bin(self):
        result = _goal_from_final_event({"score": "lte_50_0"}, _CATALOG_MIXED)
        conds = result[0]
        assert conds[0]["value"] == "lte_50_0"

    def test_numerical_attr_unknown_bin_label_skipped(self):
        """A bin label not among the catalog's possible_values (e.g. stale
        cache, mismatched discretizer boundaries) is skipped, not an error."""
        result = _goal_from_final_event({"score": "gte_999_0"}, _CATALOG_MIXED)
        assert result == [[]]

    def test_multiple_attrs_combined(self):
        result = _goal_from_final_event(
            {"status": "admitted", "critical": "false"}, _CATALOG_MIXED
        )
        conds = result[0]
        attrs = {c["attribute"] for c in conds}
        assert "status" in attrs
        assert "critical" in attrs

    def test_empty_final_attrs(self):
        result = _goal_from_final_event({}, _CATALOG_MIXED)
        assert result == [[]]


# ---------------------------------------------------------------------------
# is_q3_reachable
# ---------------------------------------------------------------------------

class TestIsQ3Reachable:
    def test_reachable_true_when_effect_exists(self):
        serialized = _serialized_with_effect("status", "discharged")
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is True

    def test_reachable_false_when_no_effect(self):
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, _SERIALIZED_EMPTY, ["p_start"]) is False

    def test_reachable_false_when_wrong_value(self):
        serialized = _serialized_with_effect("status", "admitted")
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is False

    def test_reachable_true_when_one_clause_covered(self):
        # Two clauses: first unsatisfied, second satisfied
        serialized = _serialized_with_effect("critical", "true")
        goal = [
            [{"attribute": "status", "predicate": "=", "value": "discharged"}],
            [{"attribute": "critical", "predicate": "=", "value": "true"}],
        ]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is True

    def test_reachable_false_when_partial_clause_match(self):
        # Clause needs both attrs; only one is reachable
        serialized = _serialized_with_effect("status", "discharged")
        goal = [[
            {"attribute": "status", "predicate": "=", "value": "discharged"},
            {"attribute": "critical", "predicate": "=", "value": "true"},
        ]]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is False

    def test_empty_clause_skipped(self):
        goal = [[]]
        assert is_q3_reachable(goal, _SERIALIZED_EMPTY, ["p_start"]) is False

    def test_empty_goal_returns_false(self):
        assert is_q3_reachable([], _SERIALIZED_EMPTY, ["p_start"]) is False

    def test_reachable_false_when_effect_transition_not_reachable_from_marking(self):
        """The effect exists somewhere in the net, but only on a transition
        gated behind a place the current marking never reaches — this is the
        exact case the structural pre-check must catch: an effect existing
        'somewhere' is not enough, it must be reachable from init_places."""
        serialized = _serialized_with_unreachable_effect("status", "discharged")
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is False

    def test_reachable_true_when_marking_starts_past_the_gate(self):
        """Same net as above, but the marking already sits at p_isolated, so
        act_a (and its effect) is reachable and p_after_a — reused here as
        the end place — is reached."""
        serialized = _serialized_with_unreachable_effect("status", "discharged")
        serialized["metadata"]["end_place"] = "p_after_a"
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, ["p_isolated"]) is True

    def test_reachable_false_when_end_place_unreachable(self):
        """The attribute effect is reachable, but the declared end place is
        not connected to the net at all — the goal must not be reported
        reachable if the process can never complete."""
        serialized = _serialized_with_effect("status", "discharged")
        serialized["metadata"]["end_place"] = "p_unreachable_end"
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is False

    def test_reachable_true_through_multi_hop_chain_in_adversarial_node_order(self):
        """t_2 needs the output of t_1, but t_2 is listed BEFORE t_1 in the
        graph's node order — the worklist BFS must still propagate p_start
        (via t_1 -> p_mid) to unblock t_2, regardless of node/edge order.
        Also exercises a transition (t_1) with no attribute effect of its
        own sitting in the middle of the chain."""
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "t_2", "type": "transition", "label": "act_2"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                    {"id": "p_mid", "type": "place", "label": "p_mid"},
                    {"id": "t_1", "type": "transition", "label": "act_1"},
                    {"id": "p_start", "type": "place", "label": "p_start"},
                ],
                "edges": [
                    {"source": "p_mid", "target": "t_2"},
                    {"source": "t_2", "target": "p_end"},
                    {"source": "p_start", "target": "t_1"},
                    {"source": "t_1", "target": "p_mid"},
                ],
            },
            "transitions": {
                "act_1": _transition(["p_start"]),
                "act_2": _transition(["p_mid"], [{"attribute": "status", "value": "discharged"}]),
            },
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, ["p_start"]) is True

    def test_source_transition_with_no_input_places_fires_unconditionally(self):
        """A transition with no input places at all (a net source) must
        still fire and propagate its output, even from an empty marking."""
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "t_src", "type": "transition", "label": "act_src"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                ],
                "edges": [
                    {"source": "t_src", "target": "p_end"},
                ],
            },
            "transitions": {
                "act_src": _transition([], [{"attribute": "status", "value": "discharged"}]),
            },
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized, []) is True

    def test_all_queries_use_minimize_weighted(self):
        ps = _make_prefix_sample(
            prefix_duration_s=600.0,
            full_duration_s=3600.0,
            final_attrs={"status": "admitted"},
        )
        q1 = build_q1(ps)
        q2 = build_q2(ps)
        q3 = build_q3(ps, _SERIALIZED_EMPTY)
        assert q1.metric == "minimize_weighted"
        assert q2.metric == "minimize_weighted"
        assert q3.metric == "minimize_weighted"


# ---------------------------------------------------------------------------
# compute_min_time_to_end
# ---------------------------------------------------------------------------

class TestComputeMinTimeToEnd:
    def test_end_place_is_zero(self):
        serialized = _serialized_with_effect("status", "discharged")
        result = compute_min_time_to_end(serialized, ["p_end"])
        assert result == 0.0

    def test_single_hop_uses_effective_max(self):
        # effective_min/max are not planner-chosen -- they are the observed
        # range of real execution time for that activity, and a plan is
        # only reliable if it meets the deadline even in the worst
        # realistic case, so the WORST-CASE (effective_max) bound is used,
        # not the best case.
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t_a", "type": "transition", "label": "act_a"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t_a"},
                    {"source": "t_a", "target": "p_end"},
                ],
            },
            "transitions": {"act_a": _transition(["p_start"], duration=_duration(5.0, 10.0))},
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        assert compute_min_time_to_end(serialized, ["p_start"]) == 10.0  # effective_max, not effective_min
        assert compute_min_time_to_end(serialized, ["p_end"]) == 0.0

    def test_picks_the_faster_of_two_alternative_paths(self):
        # p_start -> t1(10s worst-case) -> p_mid -> t2(6s worst-case) -> p_end  (total 16s)
        # p_start -> t3(200s worst-case) -> p_end                               (total 200s, slower)
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "p_mid", "type": "place", "label": "p_mid"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                    {"id": "t1", "type": "transition", "label": "act1"},
                    {"id": "t2", "type": "transition", "label": "act2"},
                    {"id": "t3", "type": "transition", "label": "act3"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t1"}, {"source": "t1", "target": "p_mid"},
                    {"source": "p_mid", "target": "t2"}, {"source": "t2", "target": "p_end"},
                    {"source": "p_start", "target": "t3"}, {"source": "t3", "target": "p_end"},
                ],
            },
            "transitions": {
                "act1": _transition(["p_start"], duration=_duration(5.0, 10.0)),
                "act2": _transition(["p_mid"], duration=_duration(3.0, 6.0)),
                "act3": _transition(["p_start"], duration=_duration(100.0, 200.0)),
            },
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        assert compute_min_time_to_end(serialized, ["p_start"]) == 16.0
        assert compute_min_time_to_end(serialized, ["p_mid"]) == 6.0

    def test_and_join_waits_for_the_slower_input_not_the_faster(self):
        # t_join needs BOTH p_a (via t1, 10s worst-case) and p_b (via t2,
        # 30s worst-case) -- it cannot fire until the SLOWER of the two
        # arrives (max, not min), then adds its own 2s worst-case duration.
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "p_a", "type": "place", "label": "p_a"},
                    {"id": "p_b", "type": "place", "label": "p_b"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                    {"id": "t1", "type": "transition", "label": "act1"},
                    {"id": "t2", "type": "transition", "label": "act2"},
                    {"id": "t_join", "type": "transition", "label": "act_join"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t1"}, {"source": "t1", "target": "p_a"},
                    {"source": "p_start", "target": "t2"}, {"source": "t2", "target": "p_b"},
                    {"source": "p_a", "target": "t_join"}, {"source": "p_b", "target": "t_join"},
                    {"source": "t_join", "target": "p_end"},
                ],
            },
            "transitions": {
                "act1": _transition(["p_start"], duration=_duration(5.0, 10.0)),
                "act2": _transition(["p_start"], duration=_duration(20.0, 30.0)),
                "act_join": _transition(["p_a", "p_b"], duration=_duration(1.0, 2.0)),
            },
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        # max(10, 30) + 2 = 32 -- NOT min(10, 30) + 2 = 12.
        assert compute_min_time_to_end(serialized, ["p_start"]) == 32.0

    def test_and_join_satisfied_by_a_different_concurrent_branch_of_the_marking(self):
        # Same net as above, but the marking already holds tokens in BOTH
        # p_a and p_b directly (as if two concurrent branches already
        # completed) -- t_join must fire using max(0, 0) + 2 = 2, not
        # treat p_a/p_b as isolated unreachable starting points.
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "p_a", "type": "place", "label": "p_a"},
                    {"id": "p_b", "type": "place", "label": "p_b"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                    {"id": "t_join", "type": "transition", "label": "act_join"},
                ],
                "edges": [
                    {"source": "p_a", "target": "t_join"}, {"source": "p_b", "target": "t_join"},
                    {"source": "t_join", "target": "p_end"},
                ],
            },
            "transitions": {
                "act_join": _transition(["p_a", "p_b"], duration=_duration(1.0, 2.0)),
            },
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_a", "end_place": "p_end"},
        }
        assert compute_min_time_to_end(serialized, ["p_a", "p_b"]) == 2.0

    def test_missing_duration_data_contributes_zero(self):
        serialized = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t_a", "type": "transition", "label": "act_a"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t_a"},
                    {"source": "t_a", "target": "p_end"},
                ],
            },
            "transitions": {"act_a": _transition(["p_start"])},  # duration=None
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        assert compute_min_time_to_end(serialized, ["p_start"]) == 0.0

    def test_unreachable_marking_returns_infinity(self):
        serialized = _serialized_with_unreachable_effect("status", "discharged")
        # p_isolated feeds act_a -> p_after_a, never connected to p_end (the
        # declared end place) in this fixture -- structurally can never
        # reach the end.
        assert compute_min_time_to_end(serialized, ["p_isolated"]) == float("inf")

    def test_infinity_when_no_end_place_declared(self):
        serialized = _serialized_with_effect("status", "discharged")
        serialized["metadata"] = {}
        assert compute_min_time_to_end(serialized, ["p_start"]) == float("inf")


class TestComputeMinTimeToEndCache:
    """cache is keyed on the exact marking (frozenset(init_places)), not on
    individual places -- see compute_min_time_to_end's own docstring on why
    a per-place cache would silently break AND-join correctness."""

    def _serialized(self):
        return {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t_a", "type": "transition", "label": "act_a"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t_a"},
                    {"source": "t_a", "target": "p_end"},
                ],
            },
            "transitions": {"act_a": _transition(["p_start"], duration=_duration(5.0, 10.0))},
            "attribute_catalog": _CATALOG_MIXED,
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }

    def test_cache_populated_after_first_call(self):
        serialized = self._serialized()
        cache: dict = {}
        result = compute_min_time_to_end(serialized, ["p_start"], cache=cache)
        assert result == 10.0
        assert cache[frozenset(["p_start"])] == 10.0

    def test_second_call_with_same_marking_reuses_cached_value(self):
        serialized = self._serialized()
        cache: dict = {}
        first = compute_min_time_to_end(serialized, ["p_start"], cache=cache)
        # Even if the underlying graph changed after the first call, a hit
        # must return the CACHED value, proving it didn't recompute.
        mutated = dict(serialized)
        mutated["metadata"] = {}  # would make a fresh call return +inf
        second = compute_min_time_to_end(mutated, ["p_start"], cache=cache)
        assert second == first == 10.0

    def test_marking_order_does_not_affect_cache_key(self):
        serialized = self._serialized()
        cache: dict = {}
        compute_min_time_to_end(serialized, ["p_start"], cache=cache)
        assert frozenset(["p_start"]) in cache
        # A differently-ordered but identical marking must hit the same key.
        assert len(cache) == 1

    def test_different_markings_get_independent_cache_entries(self):
        serialized = self._serialized()
        cache: dict = {}
        compute_min_time_to_end(serialized, ["p_start"], cache=cache)
        compute_min_time_to_end(serialized, ["p_end"], cache=cache)
        assert cache[frozenset(["p_start"])] == 10.0
        assert cache[frozenset(["p_end"])] == 0.0
        assert len(cache) == 2

    def test_no_cache_means_no_memoization_and_no_error(self):
        serialized = self._serialized()
        # cache=None (default) -- must behave exactly like before caching existed.
        assert compute_min_time_to_end(serialized, ["p_start"]) == 10.0
