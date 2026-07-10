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


def _serialized_with_effect(attr: str, value: str) -> Dict[str, Any]:
    return {
        **_SERIALIZED_EMPTY,
        "transitions": {
            "act_a": {
                "effects": {attr: {value: {"probability": 1.0, "guard": []}}},
                "effect_groups": [],
            }
        },
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
    def test_q3_returns_none_when_no_timestamps(self):
        ps = _make_prefix_sample(final_attrs={"status": "discharged"})
        assert build_q3(ps, _SERIALIZED_EMPTY) is None

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

    def test_q3_deadline_equals_remaining_budget(self):
        ps = _make_prefix_sample(
            prefix_duration_s=900.0,
            full_duration_s=3600.0,
            final_attrs={"status": "discharged"},
        )
        spec = build_q3(ps, _SERIALIZED_EMPTY)
        assert spec is not None
        assert spec.deadline == pytest.approx(2700.0)

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
        assert is_q3_reachable(goal, serialized) is True

    def test_reachable_false_when_no_effect(self):
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, _SERIALIZED_EMPTY) is False

    def test_reachable_false_when_wrong_value(self):
        serialized = _serialized_with_effect("status", "admitted")
        goal = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        assert is_q3_reachable(goal, serialized) is False

    def test_reachable_true_when_one_clause_covered(self):
        # Two clauses: first unsatisfied, second satisfied
        serialized = _serialized_with_effect("critical", "true")
        goal = [
            [{"attribute": "status", "predicate": "=", "value": "discharged"}],
            [{"attribute": "critical", "predicate": "=", "value": "true"}],
        ]
        assert is_q3_reachable(goal, serialized) is True

    def test_reachable_false_when_partial_clause_match(self):
        # Clause needs both attrs; only one is reachable
        serialized = _serialized_with_effect("status", "discharged")
        goal = [[
            {"attribute": "status", "predicate": "=", "value": "discharged"},
            {"attribute": "critical", "predicate": "=", "value": "true"},
        ]]
        assert is_q3_reachable(goal, serialized) is False

    def test_empty_clause_skipped(self):
        goal = [[]]
        assert is_q3_reachable(goal, _SERIALIZED_EMPTY) is False

    def test_empty_goal_returns_false(self):
        assert is_q3_reachable([], _SERIALIZED_EMPTY) is False

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
