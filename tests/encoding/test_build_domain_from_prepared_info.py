"""Tests for encoding.domain_builder.build_domain_from_prepared_info.

Builds PreparedDomainInput instances by hand (no JSON, no ParseResult) to
exercise the typed 3-axis Cartesian-product engine in isolation.
"""
import math

from encoding.domain_builder import build_domain_from_prepared_info
from encoding.pddl_model import PDDLAction, PDDLCondition, PDDLDurativeAction, PDDLEffect
from encoding.prepared_input import (
    GraphEdge,
    GraphNode,
    PreparedDomainInput,
    PreparedEffectGroup,
    PreparedTransition,
    PreparedXorBranch,
)
from models import ActionDurationStats, AttributeCatalogEntry, Guard


def _simple_prepared(**transition_kwargs) -> PreparedDomainInput:
    """One place -> one transition -> one place, plus place-marking wiring."""
    nodes = [
        GraphNode(id="p_start", type="place", label="p_start"),
        GraphNode(id="t_a", type="transition", label="activity_a"),
        GraphNode(id="p_end", type="place", label="p_end"),
    ]
    edges = [
        GraphEdge(source="p_start", target="t_a"),
        GraphEdge(source="t_a", target="p_end"),
    ]
    defaults = dict(
        activity_name="activity_a",
        input_places=["p_start"],
        preconditions=[],
        effect_groups=[],
        cost=0.0,
        duration=None,
    )
    defaults.update(transition_kwargs)
    transition = PreparedTransition(**defaults)
    return PreparedDomainInput(
        nodes=nodes,
        edges=edges,
        transitions={"activity_a": transition},
        xor_branches={},
        attribute_catalog={},
    )


def _sequence_prepared(attribute_catalog=None, **transition_a_kwargs) -> PreparedDomainInput:
    """p_start -> activity_a -> p_mid -> activity_b -> p_end, two transitions."""
    nodes = [
        GraphNode(id="p_start", type="place", label="p_start"),
        GraphNode(id="t_a", type="transition", label="activity_a"),
        GraphNode(id="p_mid", type="place", label="p_mid"),
        GraphNode(id="t_b", type="transition", label="activity_b"),
        GraphNode(id="p_end", type="place", label="p_end"),
    ]
    edges = [
        GraphEdge(source="p_start", target="t_a"),
        GraphEdge(source="t_a", target="p_mid"),
        GraphEdge(source="p_mid", target="t_b"),
        GraphEdge(source="t_b", target="p_end"),
    ]
    defaults = dict(
        activity_name="activity_a", input_places=["p_start"],
        preconditions=[], effect_groups=[], cost=0.0, duration=None,
    )
    defaults.update(transition_a_kwargs)
    transition_a = PreparedTransition(**defaults)
    transition_b = PreparedTransition(
        activity_name="activity_b", input_places=["p_mid"], preconditions=[], effect_groups=[],
    )
    return PreparedDomainInput(
        nodes=nodes,
        edges=edges,
        transitions={"activity_a": transition_a, "activity_b": transition_b},
        xor_branches={},
        attribute_catalog=attribute_catalog or {},
    )


def _tau_prepared(attribute_catalog=None) -> PreparedDomainInput:
    """p_start -> tau_0 (silent) -> p_mid -> activity_a -> p_end, no effects."""
    nodes = [
        GraphNode(id="p_start", type="place", label="p_start"),
        GraphNode(id="tau_0", type="silent", label="tau_0"),
        GraphNode(id="p_mid", type="place", label="p_mid"),
        GraphNode(id="t_a", type="transition", label="activity_a"),
        GraphNode(id="p_end", type="place", label="p_end"),
    ]
    edges = [
        GraphEdge(source="p_start", target="tau_0"),
        GraphEdge(source="tau_0", target="p_mid"),
        GraphEdge(source="p_mid", target="t_a"),
        GraphEdge(source="t_a", target="p_end"),
    ]
    transition = PreparedTransition(
        activity_name="activity_a", input_places=["p_mid"], preconditions=[], effect_groups=[],
    )
    return PreparedDomainInput(
        nodes=nodes,
        edges=edges,
        transitions={"activity_a": transition},
        xor_branches={},
        attribute_catalog=attribute_catalog or {},
    )


class TestSimpleTransition:
    def test_single_variant_named_without_suffix(self):
        domain = build_domain_from_prepared_info(_simple_prepared())
        names = {a.name for a in domain.actions}
        assert "execute_activity_a" in names

    def test_preconditions_mark_input_places(self):
        domain = build_domain_from_prepared_info(_simple_prepared())
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLCondition.marked("p_start") in action.preconditions

    def test_effects_unmark_input_and_mark_output(self):
        # Standard Petri net semantics: only places are ever marked, never
        # the transition itself -- one action does the whole firing (see
        # claude_plans/standard_petri_net_marking_plan.md).
        domain = build_domain_from_prepared_info(_simple_prepared())
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLEffect.unmarking("p_start") in action.effects
        assert PDDLEffect.marking("p_end") in action.effects
        assert PDDLEffect.marking("activity_a") not in action.effects

    def test_no_separate_place_marking_action(self):
        # mark_places_from_* no longer exists -- execute_activity_a is the
        # only action for this transition, and it marks p_end directly.
        domain = build_domain_from_prepared_info(_simple_prepared())
        names = {a.name for a in domain.actions}
        assert names == {"execute_activity_a"}

    def test_no_durative_actions_by_default(self):
        domain = build_domain_from_prepared_info(_simple_prepared())
        assert not any(isinstance(a, PDDLDurativeAction) for a in domain.actions)
        assert ":durative-actions" not in domain.requirements


def _and_split_prepared(**transition_kwargs) -> PreparedDomainInput:
    """p_start -> activity_a -> {p_out1, p_out2} (AND-split: one durative
    transition with two structural output places)."""
    nodes = [
        GraphNode(id="p_start", type="place", label="p_start"),
        GraphNode(id="t_a", type="transition", label="activity_a"),
        GraphNode(id="p_out1", type="place", label="p_out1"),
        GraphNode(id="p_out2", type="place", label="p_out2"),
    ]
    edges = [
        GraphEdge(source="p_start", target="t_a"),
        GraphEdge(source="t_a", target="p_out1"),
        GraphEdge(source="t_a", target="p_out2"),
    ]
    defaults = dict(
        activity_name="activity_a",
        input_places=["p_start"],
        preconditions=[],
        effect_groups=[],
        cost=0.0,
        duration=ActionDurationStats(effective_min=5.0, effective_max=10.0, source="external"),
    )
    defaults.update(transition_kwargs)
    transition = PreparedTransition(**defaults)
    return PreparedDomainInput(
        nodes=nodes,
        edges=edges,
        transitions={"activity_a": transition},
        xor_branches={},
        attribute_catalog={},
    )


class TestAndSplitTransition:
    """One real Petri net transition firing with multiple structural output
    places must remain a SINGLE action -- unmarking the input at start and
    marking every output place at end (standard Petri net semantics, see
    claude_plans/standard_petri_net_marking_plan.md)."""

    def test_single_action_marks_every_output_place(self):
        domain = build_domain_from_prepared_info(_and_split_prepared(), use_durative=True)
        names = {a.name for a in domain.actions}
        assert names == {"execute_activity_a"}

        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert isinstance(action, PDDLDurativeAction)
        assert PDDLEffect.unmarking("p_start") in action.effects_at_start
        assert PDDLEffect.marking("p_out1") in action.effects_at_end
        assert PDDLEffect.marking("p_out2") in action.effects_at_end
        assert PDDLEffect.marking("p_start") not in action.effects_at_start
        assert PDDLEffect.marking("p_start") not in action.effects_at_end

    def test_no_transition_marking_and_no_mark_places_action(self):
        domain = build_domain_from_prepared_info(_and_split_prepared(), use_durative=True)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLEffect.marking("activity_a") not in action.effects_at_start
        assert PDDLEffect.marking("activity_a") not in action.effects_at_end
        assert not any(a.name.startswith("mark_places_from_") for a in domain.actions)


class TestPreconditions:
    def test_structural_precondition_guard_applied(self):
        guard = Guard(attribute="risk", value="high", negated=False)
        prepared = _simple_prepared(preconditions=[[guard]])
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLCondition.attr_is("risk", "high") in action.preconditions

    def test_duplicate_and_clauses_do_not_produce_duplicate_variants(self):
        guard = Guard(attribute="risk", value="high", negated=False)
        prepared = _simple_prepared(preconditions=[[guard], [guard]])
        domain = build_domain_from_prepared_info(prepared)
        matching = [a for a in domain.actions if a.name.startswith("execute_activity_a")]
        assert len(matching) == 1
        assert matching[0].name == "execute_activity_a"

    def test_precondition_conflicting_with_effect_group_guard_produces_no_action(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu"},
            )
        }
        precondition_guard = Guard(attribute="risk", value="high", negated=False)
        group_guard = Guard(attribute="risk", value="low", negated=False)
        group = PreparedEffectGroup(
            assignments=[("diagnosis", "flu")], guard=[[group_guard]], probability=1.0,
        )
        prepared = _simple_prepared(
            preconditions=[[precondition_guard]], effect_groups=[group],
        )
        prepared.attribute_catalog = catalog
        domain = build_domain_from_prepared_info(prepared)
        matching = [a for a in domain.actions if a.name.startswith("execute_activity_a")]
        assert matching == []


class TestXorBranch:
    def test_guard_clauses_produce_one_variant_per_clause(self):
        left = Guard(attribute="risk", value="high", negated=False)
        right = Guard(attribute="risk", value="low", negated=False)
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor"] = [
            PreparedXorBranch(
                activity_name="activity_a", conditions=[[left], [right]], probability=0.5,
            )
        ]
        domain = build_domain_from_prepared_info(prepared)
        variant_names = {a.name for a in domain.actions if a.name.startswith("execute_activity_a")}
        assert variant_names == {"execute_activity_a_v0", "execute_activity_a_v1"}

        conds = {
            frozenset(a.preconditions)
            for a in domain.actions if a.name.startswith("execute_activity_a_v")
        }
        assert frozenset({PDDLCondition.marked("p_start"), PDDLCondition.attr_is("risk", "high")}) in conds
        assert frozenset({PDDLCondition.marked("p_start"), PDDLCondition.attr_is("risk", "low")}) in conds

    def test_no_conditions_adds_probability_cost(self):
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[], probability=0.25)
        ]
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.base_cost == -math.log(0.25)

    def test_full_probability_adds_no_cost(self):
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[], probability=1.0)
        ]
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.base_cost is None

    def test_two_branches_with_identical_conditions_do_not_duplicate(self):
        guard = Guard(attribute="risk", value="high", negated=False)
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor1"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[guard]], probability=0.5)
        ]
        prepared.xor_branches["p_xor2"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[guard]], probability=0.5)
        ]
        domain = build_domain_from_prepared_info(prepared)
        matching = [a for a in domain.actions if a.name.startswith("execute_activity_a")]
        assert len(matching) == 1
        assert PDDLCondition.attr_is("risk", "high") in matching[0].preconditions

    def test_two_branches_with_compatible_conditions_combine(self):
        risk_high = Guard(attribute="risk", value="high", negated=False)
        urgent = Guard(attribute="urgent", value=None, negated=False)
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor1"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[risk_high]], probability=0.5)
        ]
        prepared.xor_branches["p_xor2"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[urgent]], probability=0.5)
        ]
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLCondition.attr_is("risk", "high") in action.preconditions
        assert PDDLCondition.attr_true("urgent") in action.preconditions

    def test_two_branches_with_contradictory_conditions_produce_no_action(self):
        risk_high = Guard(attribute="risk", value="high", negated=False)
        risk_low = Guard(attribute="risk", value="low", negated=False)
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor1"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[risk_high]], probability=0.5)
        ]
        prepared.xor_branches["p_xor2"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[risk_low]], probability=0.5)
        ]
        domain = build_domain_from_prepared_info(prepared)
        matching = [a for a in domain.actions if a.name.startswith("execute_activity_a")]
        assert matching == []

    def test_two_unguarded_branches_multiply_cost(self):
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor1"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[], probability=0.5)
        ]
        prepared.xor_branches["p_xor2"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[], probability=0.25)
        ]
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.base_cost == -math.log(0.5 * 0.25)

    def test_one_guarded_one_unguarded_branch_combine(self):
        guard = Guard(attribute="risk", value="high", negated=False)
        prepared = _simple_prepared()
        prepared.xor_branches["p_xor1"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[[guard]], probability=1.0)
        ]
        prepared.xor_branches["p_xor2"] = [
            PreparedXorBranch(activity_name="activity_a", conditions=[], probability=0.4)
        ]
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLCondition.attr_is("risk", "high") in action.preconditions
        assert action.base_cost == -math.log(0.4)


class TestEffectGroups:
    def test_group_with_guard_produces_variant_per_clause(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        guard = Guard(attribute="urgent", value=None, negated=False)
        group = PreparedEffectGroup(
            assignments=[("diagnosis", "flu")], guard=[[guard]], probability=1.0,
        )
        prepared = _simple_prepared(effect_groups=[group])
        prepared.attribute_catalog = catalog
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLCondition.attr_true("urgent") in action.preconditions
        assert PDDLEffect.set_attr_is("diagnosis", "flu") in action.effects

    def test_group_without_guard_adds_probability_cost(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        group = PreparedEffectGroup(
            assignments=[("diagnosis", "flu")], guard=[], probability=0.6,
        )
        prepared = _simple_prepared(effect_groups=[group])
        prepared.attribute_catalog = catalog
        domain = build_domain_from_prepared_info(prepared)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.base_cost == -math.log(0.6)


class TestDurativeAction:
    def test_duration_produces_durative_action_when_enabled(self):
        duration = ActionDurationStats(effective_min=5.0, effective_max=10.0, source="external")
        prepared = _simple_prepared(duration=duration)
        domain = build_domain_from_prepared_info(prepared, use_durative=True)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert isinstance(action, PDDLDurativeAction)
        assert action.duration_min == 5.0
        assert action.duration_max == 10.0
        assert ":durative-actions" in domain.requirements
        assert ":duration-inequalities" in domain.requirements

    def test_duration_ignored_when_use_durative_false(self):
        duration = ActionDurationStats(effective_min=5.0, effective_max=10.0, source="external")
        prepared = _simple_prepared(duration=duration)
        domain = build_domain_from_prepared_info(prepared, use_durative=False)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert isinstance(action, PDDLAction)

    def test_has_deadline_requires_durative_action_present(self):
        prepared = _simple_prepared()
        domain = build_domain_from_prepared_info(prepared, use_durative=True, has_deadline=True)
        assert domain.has_deadline is False  # no duration data -> no durative action


class TestDurationRescale:
    """domain_builder._maybe_rescale_durations — see
    encoding/domain_builder.py::_FD_MAX_DURATION_VALUE for why this exists:
    Fast Downward silently corrupts its internal search state (does not
    raise a clean error) when a durative action's duration exceeds
    INT32_MAX/1000 ~= 2,147,483.647."""

    _FD_MAX = 2147483.647

    def test_no_rescale_when_under_limit(self):
        duration = ActionDurationStats(effective_min=5.0, effective_max=10.0, source="external")
        prepared = _simple_prepared(duration=duration)
        domain = build_domain_from_prepared_info(prepared, use_durative=True)
        assert domain.duration_scale_factor == 1.0
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.duration_min == 5.0
        assert action.duration_max == 10.0

    def test_rescale_applied_when_over_limit(self):
        duration = ActionDurationStats(
            effective_min=12990198.67, effective_max=60021504.9, source="external",
        )
        prepared = _simple_prepared(duration=duration)
        domain = build_domain_from_prepared_info(prepared, use_durative=True)
        assert domain.duration_scale_factor > 1.0
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.duration_max <= self._FD_MAX
        assert action.duration_min <= self._FD_MAX
        # Original ratio between min/max must be preserved (uniform scaling).
        assert action.duration_max == 60021504.9 / domain.duration_scale_factor
        assert action.duration_min == 12990198.67 / domain.duration_scale_factor

    def test_rescale_picks_finest_sufficient_unit(self):
        # 5,000,000s: /60 (minutes) -> 83333.3, still way under the limit but
        # /1 already fails, so the finest candidate that clears the bar is
        # minutes (60) -- not hours/days/weeks.
        duration = ActionDurationStats(effective_min=0.0, effective_max=5_000_000.0, source="external")
        prepared = _simple_prepared(duration=duration)
        domain = build_domain_from_prepared_info(prepared, use_durative=True)
        assert domain.duration_scale_factor == 60.0

    def test_rescale_never_leaves_value_over_limit(self):
        duration = ActionDurationStats(effective_min=0.0, effective_max=6e11, source="external")
        prepared = _simple_prepared(duration=duration)
        domain = build_domain_from_prepared_info(prepared, use_durative=True)
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert action.duration_max <= self._FD_MAX

    def test_rescale_applies_uniformly_to_every_durative_action(self):
        """A single over-limit action must trigger the same scale factor for
        every OTHER durative action too — different actions on different time
        axes within the same plan would be meaningless."""
        big_duration = ActionDurationStats(effective_min=0.0, effective_max=6_000_000.0, source="external")
        small_duration = ActionDurationStats(effective_min=1.0, effective_max=2.0, source="external")
        prepared = _sequence_prepared(duration=big_duration)
        prepared.transitions["activity_b"] = PreparedTransition(
            activity_name="activity_b", input_places=["p_mid"], preconditions=[], effect_groups=[],
            duration=small_duration,
        )
        domain = build_domain_from_prepared_info(prepared, use_durative=True)
        factor = domain.duration_scale_factor
        assert factor > 1.0
        action_a = next(a for a in domain.actions if a.name == "execute_activity_a")
        action_b = next(a for a in domain.actions if a.name == "execute_activity_b")
        assert action_a.duration_max == 6_000_000.0 / factor
        assert action_b.duration_max == 2.0 / factor
        assert action_b.duration_min == 1.0 / factor


class TestTauAction:
    def test_plain_tau_produces_execute_action(self):
        nodes = [
            GraphNode(id="p_start", type="place", label="p_start"),
            GraphNode(id="tau_0", type="silent", label="tau_0"),
            GraphNode(id="p_end", type="place", label="p_end"),
        ]
        edges = [
            GraphEdge(source="p_start", target="tau_0"),
            GraphEdge(source="tau_0", target="p_end"),
        ]
        prepared = PreparedDomainInput(
            nodes=nodes, edges=edges, transitions={}, xor_branches={}, attribute_catalog={},
        )
        domain = build_domain_from_prepared_info(prepared)
        tau_action = next(a for a in domain.actions if a.name == "execute_tau_0")
        assert PDDLCondition.marked("p_start") in tau_action.preconditions
        assert PDDLEffect.marking("p_end") in tau_action.effects
        assert PDDLEffect.marking("tau_0") not in tau_action.effects

    def test_tau_covered_by_prepared_transition_is_not_duplicated(self):
        """A silent GraphNode whose label is also a PreparedTransition key (XOR-branch
        tau) must be built once, with guards, not twice."""
        nodes = [
            GraphNode(id="p_start", type="place", label="p_start"),
            GraphNode(id="xor_tau_1", type="silent", label="xor_tau_1"),
        ]
        edges = [GraphEdge(source="p_start", target="xor_tau_1")]
        transition = PreparedTransition(
            activity_name="xor_tau_1",
            input_places=["p_start"],
            preconditions=[],
            effect_groups=[],
        )
        prepared = PreparedDomainInput(
            nodes=nodes, edges=edges, transitions={"xor_tau_1": transition},
            xor_branches={}, attribute_catalog={},
        )
        domain = build_domain_from_prepared_info(prepared)
        matching = [a for a in domain.actions if a.name == "execute_xor_tau_1"]
        assert len(matching) == 1


class TestDeduplication:
    def test_equivalent_variants_collapse_to_one(self):
        # Two effect groups producing identical preconditions/effects (same
        # assignment, both without a guard, same probability) must collapse.
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu"},
            )
        }
        group_a = PreparedEffectGroup(
            assignments=[("diagnosis", "flu")], guard=[], probability=1.0,
        )
        group_b = PreparedEffectGroup(
            assignments=[("diagnosis", "flu")], guard=[], probability=1.0,
        )
        prepared = _simple_prepared(effect_groups=[group_a, group_b])
        prepared.attribute_catalog = catalog
        domain = build_domain_from_prepared_info(prepared)
        matching = [a for a in domain.actions if a.name.startswith("execute_activity_a")]
        assert len(matching) == 1
        assert matching[0].name == "execute_activity_a"


class TestNegatedAttributes:
    def test_negated_precondition_generates_is_not_predicate(self):
        guard = Guard(attribute="risk", value="high", negated=True)
        prepared = _simple_prepared(preconditions=[[guard]])
        prepared.attribute_catalog = {
            "risk": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"high", "low"},
            )
        }
        domain = build_domain_from_prepared_info(prepared)
        predicate_names = {p.name for p in domain.predicates}
        assert "risk_is_not" in predicate_names
        action = next(a for a in domain.actions if a.name == "execute_activity_a")
        assert PDDLCondition.attr_is_not("risk", "high") in action.preconditions


class TestTypes:
    def test_always_has_petri_element_hierarchy(self):
        # Transitions are never PDDL objects (standard Petri net semantics
        # only ever marks places) -- "transition" is not a declared type.
        domain = build_domain_from_prepared_info(_sequence_prepared())
        type_map = {t.name: t.parent for t in domain.types}

        assert "petri_element" in type_map
        assert type_map["place"] == "petri_element"
        assert "transition" not in type_map

    def test_categorical_attribute_generates_value_type(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        type_names = {t.name for t in domain.types}
        assert "diagnosis_val" in type_names

    def test_numerical_attribute_generates_value_type(self):
        catalog = {
            "crp": AttributeCatalogEntry(
                attribute_type="numerical", possible_values={"lte_10_0", "gte_10_0"},
            )
        }
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        type_names = {t.name for t in domain.types}
        assert "crp_val" in type_names

    def test_boolean_attribute_no_value_type(self):
        catalog = {"urgent": AttributeCatalogEntry(attribute_type="boolean", possible_values=set())}
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        type_names = {t.name for t in domain.types}
        assert "urgent_val" not in type_names

    def test_empty_catalog_only_petri_types(self):
        domain = build_domain_from_prepared_info(_tau_prepared())
        type_names = {t.name for t in domain.types}
        assert type_names == {"petri_element", "place"}


class TestConstants:
    def test_places_as_constants(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        place_consts = {c.name for c in domain.constants if c.type_name == "place"}
        assert place_consts == {"p_start", "p_mid", "p_end"}

    def test_transitions_are_not_constants(self):
        # Standard Petri net semantics: transitions are never markable PDDL
        # objects, only places are -- see
        # claude_plans/standard_petri_net_marking_plan.md.
        domain = build_domain_from_prepared_info(_sequence_prepared())
        all_names = {c.name for c in domain.constants}
        assert "activity_a" not in all_names
        assert "activity_b" not in all_names
        assert not any(c.type_name == "transition" for c in domain.constants)

    def test_tau_transitions_are_not_constants(self):
        domain = build_domain_from_prepared_info(_tau_prepared())
        all_names = {c.name for c in domain.constants}
        assert "tau_0" not in all_names
        assert "activity_a" not in all_names

    def test_categorical_value_objects(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold", "covid"},
            )
        }
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        diag_vals = {c.name for c in domain.constants if c.type_name == "diagnosis_val"}
        # Constant names always carry the "{attr}_val_{value}" prefix (see
        # core_utils.sanitize_value) so a bare value like "flu" can never
        # collide with a differently-attributed constant sharing the same
        # raw value.
        assert diag_vals == {"diagnosis_val_flu", "diagnosis_val_cold", "diagnosis_val_covid"}

    def test_no_value_objects_for_boolean(self):
        catalog = {"urgent": AttributeCatalogEntry(attribute_type="boolean", possible_values=set())}
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        urgent_vals = {c.name for c in domain.constants if c.type_name == "urgent_val"}
        assert urgent_vals == set()


class TestPredicates:
    def test_marked_predicate_exists(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        marked = [p for p in domain.predicates if p.name == "marked"]
        assert len(marked) == 1
        assert marked[0].parameters == [("?x", "petri_element")]

    def test_categorical_positive_predicate_always_present(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        pred_names = {p.name for p in domain.predicates}
        assert "diagnosis_is" in pred_names

    def test_categorical_negative_predicate_absent_when_not_negated(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        pred_names = {p.name for p in domain.predicates}
        assert "diagnosis_is_not" not in pred_names

    def test_categorical_negative_predicate_present_when_negated(self):
        # A negated Guard anywhere (here: activity_a's precondition) is what
        # drives _prepared_negated_attributes to add the "_is_not" predicate.
        guard = Guard(attribute="diagnosis", value="flu", negated=True)
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        domain = build_domain_from_prepared_info(
            _sequence_prepared(catalog, preconditions=[[guard]])
        )
        pred_names = {p.name for p in domain.predicates}
        assert "diagnosis_is" in pred_names
        assert "diagnosis_is_not" in pred_names

    def test_categorical_predicate_has_typed_parameter(self):
        guard = Guard(attribute="diagnosis", value="flu", negated=True)
        catalog = {
            "diagnosis": AttributeCatalogEntry(
                attribute_type="categorical", possible_values={"flu", "cold"},
            )
        }
        domain = build_domain_from_prepared_info(
            _sequence_prepared(catalog, preconditions=[[guard]])
        )
        diag_is = next(p for p in domain.predicates if p.name == "diagnosis_is")
        diag_is_not = next(p for p in domain.predicates if p.name == "diagnosis_is_not")
        assert diag_is.parameters == [("?v", "diagnosis_val")]
        assert diag_is_not.parameters == [("?v", "diagnosis_val")]

    def test_boolean_false_predicate_absent_when_not_negated(self):
        catalog = {"urgent": AttributeCatalogEntry(attribute_type="boolean", possible_values=set())}
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        pred_names = {p.name for p in domain.predicates}
        assert "urgent_true" in pred_names
        assert "urgent_false" not in pred_names

    def test_boolean_false_predicate_present_when_negated(self):
        guard = Guard(attribute="urgent", value=None, negated=True)
        catalog = {"urgent": AttributeCatalogEntry(attribute_type="boolean", possible_values=set())}
        domain = build_domain_from_prepared_info(
            _sequence_prepared(catalog, preconditions=[[guard]])
        )
        urgent_true = next(p for p in domain.predicates if p.name == "urgent_true")
        urgent_false = next(p for p in domain.predicates if p.name == "urgent_false")
        assert urgent_true.parameters == []
        assert urgent_false.parameters == []

    def test_mixed_catalog_no_negated(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(attribute_type="categorical", possible_values={"flu"}),
            "crp": AttributeCatalogEntry(attribute_type="numerical", possible_values={"lte_10_0"}),
            "urgent": AttributeCatalogEntry(attribute_type="boolean", possible_values=set()),
        }
        domain = build_domain_from_prepared_info(_sequence_prepared(catalog))
        pred_names = {p.name for p in domain.predicates}
        assert "diagnosis_is" in pred_names
        assert "crp_is" in pred_names
        assert "urgent_true" in pred_names
        assert "diagnosis_is_not" not in pred_names
        assert "crp_is_not" not in pred_names
        assert "urgent_false" not in pred_names

    def test_mixed_catalog_all_negated(self):
        catalog = {
            "diagnosis": AttributeCatalogEntry(attribute_type="categorical", possible_values={"flu"}),
            "crp": AttributeCatalogEntry(attribute_type="numerical", possible_values={"lte_10_0"}),
            "urgent": AttributeCatalogEntry(attribute_type="boolean", possible_values=set()),
        }
        guards = [
            Guard(attribute="diagnosis", value="flu", negated=True),
            Guard(attribute="crp", value="lte_10_0", negated=True),
            Guard(attribute="urgent", value=None, negated=True),
        ]
        domain = build_domain_from_prepared_info(
            _sequence_prepared(catalog, preconditions=[[g] for g in guards])
        )
        pred_names = {p.name for p in domain.predicates}
        expected = {
            "marked",
            "diagnosis_is", "diagnosis_is_not",
            "crp_is", "crp_is_not",
            "urgent_true", "urgent_false",
        }
        assert expected.issubset(pred_names)


class TestBuildDomain:
    def test_domain_name(self):
        domain = build_domain_from_prepared_info(_sequence_prepared(), domain_name="my_process")
        assert domain.name == "my_process"

    def test_default_requirements(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        assert domain.requirements == [":strips", ":typing"]

    def test_actions_are_populated(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        assert len(domain.actions) > 0

    def test_action_count_sequence_net(self):
        # One action per transition (no more separate mark_places_from_*
        # actions) -- see claude_plans/standard_petri_net_marking_plan.md.
        domain = build_domain_from_prepared_info(_sequence_prepared())
        mark_actions = [a for a in domain.actions if a.name.startswith("mark_")]
        exec_actions = [a for a in domain.actions if a.name.startswith("execute_")]
        assert len(mark_actions) == 0
        assert len(exec_actions) == 2


class TestHasCosts:
    def test_has_costs_false_by_default(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        assert domain.has_costs is False

    def test_no_action_costs_requirement_when_no_cost(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        assert ":action-costs" not in domain.requirements
        assert ":numeric-fluents" not in domain.requirements

    def test_has_costs_true_when_requested(self):
        # has_costs on PDDLDomain simply echoes the use_costs flag — it does
        # not depend on whether any action actually ended up with a cost.
        domain = build_domain_from_prepared_info(_sequence_prepared(), use_costs=True)
        assert domain.has_costs is True

    def test_action_costs_requirements_added_when_has_costs(self):
        domain = build_domain_from_prepared_info(_sequence_prepared(), use_costs=True)
        assert ":action-costs" in domain.requirements
        assert ":numeric-fluents" in domain.requirements


class TestDeadline:
    def test_no_deadline_when_not_durative(self):
        # has_deadline=True is requested, but with no duration data and
        # use_durative=False no durative action can ever be produced.
        domain = build_domain_from_prepared_info(
            _sequence_prepared(), use_durative=False, has_deadline=True,
        )
        assert domain.has_deadline is False

    def test_no_deadline_when_durative_but_no_durations(self):
        # _sequence_prepared has no duration data — all actions stay instantaneous.
        domain = build_domain_from_prepared_info(
            _sequence_prepared(), use_durative=True, has_deadline=True,
        )
        assert domain.has_deadline is False

    def _durative_sequence_prepared(self) -> PreparedDomainInput:
        duration = ActionDurationStats(effective_min=5.0, effective_max=10.0, source="external")
        return _sequence_prepared(duration=duration)

    def test_has_deadline_when_durative_with_durations(self):
        # activity_a has duration data -> at least one PDDLDurativeAction produced.
        domain = build_domain_from_prepared_info(
            self._durative_sequence_prepared(), use_durative=True, has_deadline=True,
        )
        assert domain.has_deadline is True

    def test_deadline_predicate_in_rendered_predicates(self):
        domain = build_domain_from_prepared_info(
            self._durative_sequence_prepared(), use_durative=True, has_deadline=True,
        )
        text = str(domain)
        assert "(deadline_ok)" in text

    def test_durative_action_has_over_all_deadline(self):
        domain = build_domain_from_prepared_info(
            self._durative_sequence_prepared(), use_durative=True, has_deadline=True,
        )
        text = str(domain)
        assert "(over all (deadline_ok))" in text

    def test_instantaneous_action_in_durative_domain_has_deadline_precondition(self):
        # activity_b has no duration -> rendered as PDDLAction, not PDDLDurativeAction,
        # but still gets the deadline_ok precondition since has_deadline is True.
        domain = build_domain_from_prepared_info(
            self._durative_sequence_prepared(), use_durative=True, has_deadline=True,
        )
        assert domain.has_deadline is True

        durative_names = {a.name for a in domain.actions if isinstance(a, PDDLDurativeAction)}
        instant_names = {a.name for a in domain.actions if not isinstance(a, PDDLDurativeAction)}
        assert any("activity_b" in n for n in instant_names), "activity_b should be instantaneous"

        text = str(domain)
        action_blocks = text.split("(:action ")
        activity_b_block = next((b for b in action_blocks if b.startswith("execute_activity_b")), None)
        assert activity_b_block is not None
        assert "(deadline_ok)" in activity_b_block

    def test_deadline_predicate_absent_without_durative(self):
        domain = build_domain_from_prepared_info(
            _sequence_prepared(), use_durative=False, has_deadline=True,
        )
        text = str(domain)
        assert "deadline_ok" not in text
