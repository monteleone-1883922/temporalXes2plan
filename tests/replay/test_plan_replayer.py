"""Tests for replay.plan_replayer.replay_plan (caso 4 of
docs/trace_replayer_analysis.md §2 — supersedes evaluation/plan_validator.py).

Builds a real PDDLDomain via build_domain_with_variant_map so variant_map is
exactly what a planner-facing domain would carry, then feeds hand-picked
variant names as a "plan" (no planner/PDDL solver involved).

One real Petri net transition firing is a SINGLE plan step (standard
semantics: only places are ever marked, never the transition itself — see
claude_plans/standard_petri_net_marking_plan.md) — there is no separate
"mark_places_from_*" step to include in plan_steps.
"""
from models import AttributeCatalogEntry, Guard
from encoding.domain_builder import build_domain_with_variant_map
from encoding.prepared_input import (
    GraphEdge, GraphNode, PreparedDomainInput, PreparedEffectGroup, PreparedTransition,
)
from replay.plan_replayer import replay_plan

_CATALOG = {
    "flag": AttributeCatalogEntry(attribute_type="categorical", possible_values={"x", "y"}, bin_boundaries=None),
    "risk": AttributeCatalogEntry(attribute_type="categorical", possible_values={"high", "low"}, bin_boundaries=None),
}


def _prepared_and_variant_map():
    group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
    group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
    trans_a = PreparedTransition(
        activity_name="A", input_places=["p_in"], preconditions=[],
        effect_groups=[group_a1, group_a2],
    )
    group_b = PreparedEffectGroup(
        assignments=[("risk", "high")], guard=[[Guard("flag", "y", False)]], probability=1.0,
    )
    trans_b = PreparedTransition(
        activity_name="B", input_places=["p_mid"], preconditions=[], effect_groups=[group_b],
    )
    nodes = [
        GraphNode(id="p_in", type="place", label="p_in"),
        GraphNode(id="p_mid", type="place", label="p_mid"),
        GraphNode(id="p_out", type="place", label="p_out"),
        GraphNode(id="t_a", type="transition", label="A"),
        GraphNode(id="t_b", type="transition", label="B"),
    ]
    edges = [
        GraphEdge(source="p_in", target="t_a"), GraphEdge(source="t_a", target="p_mid"),
        GraphEdge(source="p_mid", target="t_b"), GraphEdge(source="t_b", target="p_out"),
    ]
    prepared = PreparedDomainInput(
        transitions={"A": trans_a, "B": trans_b}, xor_branches={},
        attribute_catalog=_CATALOG, nodes=nodes, edges=edges,
    )
    _, variant_map = build_domain_with_variant_map(prepared)
    return prepared, variant_map


def _variant_named_for(variant_map, activity_name, assignments):
    return next(
        name for name, info in variant_map.items()
        if info.activity_name == activity_name
        and (info.effect_group.assignments if info.effect_group else []) == assignments
    )


class TestReplayPlan:
    def test_full_plan_with_correct_variant_choice_succeeds(self):
        prepared, variant_map = _prepared_and_variant_map()
        a_variant = _variant_named_for(variant_map, "A", [("flag", "y")])
        b_variant = _variant_named_for(variant_map, "B", [("risk", "high")])
        plan_steps = [a_variant, b_variant]

        outcome = replay_plan(
            plan_steps, init_marking={"p_in"}, init_attributes={},
            prepared=prepared, variant_map=variant_map,
        )

        assert outcome.is_replayable
        assert outcome.error_step is None
        assert outcome.final_attributes == {"flag": "y", "risk": "high"}
        assert outcome.final_marking == {"p_out"}
        assert [s.activity_name for s in outcome.steps] == ["A", "B"]

    def test_wrong_variant_choice_fails_at_next_precondition(self):
        prepared, variant_map = _prepared_and_variant_map()
        wrong_a = _variant_named_for(variant_map, "A", [("flag", "x")])
        b_variant = _variant_named_for(variant_map, "B", [("risk", "high")])
        plan_steps = [wrong_a, b_variant]

        outcome = replay_plan(
            plan_steps, init_marking={"p_in"}, init_attributes={},
            prepared=prepared, variant_map=variant_map,
        )

        assert not outcome.is_replayable
        assert outcome.error_step == 1
        assert "B" in outcome.error_reason
        assert outcome.final_attributes == {"flag": "x"}

    def test_missing_input_token_fails_token_flow_layer(self):
        prepared, variant_map = _prepared_and_variant_map()
        b_variant = _variant_named_for(variant_map, "B", [("risk", "high")])
        # Skip "A" entirely -- B's input place p_mid is never marked.
        plan_steps = [b_variant]

        outcome = replay_plan(
            plan_steps, init_marking={"p_in"}, init_attributes={},
            prepared=prepared, variant_map=variant_map,
        )

        assert not outcome.is_replayable
        assert outcome.error_step == 0
        assert "p_mid" in outcome.error_reason

    def test_firing_the_same_activity_twice_in_a_row_fails_the_second_time(self):
        # A's own action already unmarks p_in and marks p_mid within a
        # SINGLE step -- firing A again immediately (without anything
        # re-marking p_in first) must fail on the second attempt, since
        # p_in is no longer marked.
        prepared, variant_map = _prepared_and_variant_map()
        a_variant = _variant_named_for(variant_map, "A", [("flag", "y")])

        outcome = replay_plan(
            [a_variant, a_variant], init_marking={"p_in"}, init_attributes={},
            prepared=prepared, variant_map=variant_map,
        )

        assert not outcome.is_replayable
        assert outcome.error_step == 1
        assert "p_in" in outcome.error_reason
