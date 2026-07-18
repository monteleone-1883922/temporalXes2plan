"""Tests for encoding.pddl_model — PDDLDomain/PDDLAction/PDDLDurativeAction rendering."""
import pytest

from encoding.pddl_model import (
    PDDLAction, PDDLBaseAction, PDDLCondition, PDDLConstants, PDDLDomain, PDDLDurativeAction,
    PDDLEffect, PDDLObject, PDDLPredicate, PDDLPredicates, PDDLType, PDDLTypes,
)
from encoding.domain_builder import build_domain_from_prepared_info
from encoding.prepared_input import GraphEdge, GraphNode, PreparedDomainInput, PreparedTransition


@pytest.fixture
def minimal_domain():
    return PDDLDomain(
        name="test",
        requirements=[":strips", ":typing"],
        types=PDDLTypes(items=[
            PDDLType("petri_element"),
            PDDLType("place", parent="petri_element"),
            PDDLType("transition", parent="petri_element"),
        ]),
        constants=PDDLConstants(items=[
            PDDLObject("p1", "place"),
            PDDLObject("t1", "transition"),
        ]),
        predicates=PDDLPredicates(items=[
            PDDLPredicate("marked", [("?x", "petri_element")]),
        ]),
        actions=[
            PDDLAction(
                name="mark_p1_from_t1",
                preconditions={PDDLCondition.marked("t1")},
                effects=[PDDLEffect.marking("p1")],
            ),
        ],
    )


class TestDomainStr:

    def test_domain_name_in_output(self, minimal_domain):
        text = str(minimal_domain)
        assert "(define (domain test)" in text

    def test_requirements_in_output(self, minimal_domain):
        text = str(minimal_domain)
        assert ":strips" in text
        assert ":typing" in text

    def test_types_grouped_by_parent(self, minimal_domain):
        text = str(minimal_domain)
        assert "place transition - petri_element" in text or \
               "transition place - petri_element" in text

    def test_constants_grouped_by_type(self, minimal_domain):
        text = str(minimal_domain)
        assert "p1 - place" in text
        assert "t1 - transition" in text

    def test_predicate_with_parameters(self, minimal_domain):
        text = str(minimal_domain)
        assert "(marked ?x - petri_element)" in text

    def test_ground_predicate(self):
        domain = PDDLDomain(
            name="test", requirements=[":strips"],
            types=PDDLTypes(), constants=PDDLConstants(),
            predicates=PDDLPredicates(items=[PDDLPredicate("urgent_true")]),
            actions=[],
        )
        text = str(domain)
        assert "(urgent_true)" in text

    def test_closing_parenthesis(self, minimal_domain):
        text = str(minimal_domain)
        assert text.strip().endswith(")")


class TestPDDLTypes:

    def test_iterates_items(self):
        types = PDDLTypes(items=[PDDLType("a"), PDDLType("b")])
        assert {t.name for t in types} == {"a", "b"}

    def test_add(self):
        types = PDDLTypes()
        types.add(PDDLType("place"))
        assert list(types) == [PDDLType("place")]

    def test_empty_block(self):
        assert str(PDDLTypes()) == "  (:types\n  )"

    def test_no_parent_grouped_under_object(self):
        text = str(PDDLTypes(items=[PDDLType("standalone")]))
        assert "standalone - object" in text


class TestPDDLActionRendering:

    def test_single_precondition_no_and(self):
        action = PDDLAction(
            name="simple",
            preconditions={PDDLCondition.marked("p1")},
            effects=[PDDLEffect.marking("t1")],
        )
        text = action.to_pddl()
        assert ":precondition (marked p1)" in text
        assert "(and" not in text

    def test_multiple_preconditions_wrapped_in_and(self):
        action = PDDLAction(
            name="join",
            preconditions={PDDLCondition.marked("p1"), PDDLCondition.marked("p2")},
            effects=[PDDLEffect.marking("t1")],
        )
        text = action.to_pddl()
        assert ":precondition (and" in text
        assert "(marked p1)" in text
        assert "(marked p2)" in text

    def test_empty_precondition(self):
        action = PDDLAction(name="noop", preconditions=set(), effects=[PDDLEffect.marking("p1")])
        text = action.to_pddl()
        assert ":precondition ()" in text

    def test_single_effect_no_and(self):
        action = PDDLAction(
            name="simple",
            preconditions={PDDLCondition.marked("p1")},
            effects=[PDDLEffect.marking("t1")],
        )
        text = action.to_pddl()
        assert ":effect (marked t1)" in text

    def test_multiple_effects_wrapped_in_and(self):
        action = PDDLAction(
            name="multi",
            preconditions=set(),
            effects=[PDDLEffect.marking("t1"), PDDLEffect.set_attr_is("diagnosis", "flu")],
        )
        text = action.to_pddl()
        assert ":effect (and" in text

    def test_empty_parameters(self):
        action = PDDLAction(name="act", preconditions=set(), effects=[])
        text = action.to_pddl()
        assert ":parameters ()" in text

    def test_str_matches_to_pddl_defaults(self):
        action = PDDLAction(
            name="simple",
            preconditions={PDDLCondition.marked("p1")},
            effects=[PDDLEffect.marking("t1")],
        )
        assert str(action) == action.to_pddl()


class TestWriteToFile:

    def test_writes_file(self, tmp_path, minimal_domain):
        out = tmp_path / "domain.pddl"
        text = minimal_domain.write(out)

        assert out.exists()
        assert out.read_text(encoding="utf-8") == text

    def test_creates_parent_directories(self, tmp_path, minimal_domain):
        out = tmp_path / "sub" / "dir" / "domain.pddl"
        minimal_domain.write(out)

        assert out.exists()

    def test_returned_text_matches_str_domain(self, tmp_path, minimal_domain):
        out = tmp_path / "domain.pddl"
        text = minimal_domain.write(out)
        assert text == str(minimal_domain)


def _sequence_prepared() -> PreparedDomainInput:
    """p_start -> activity_a -> p_mid -> activity_b -> p_end, no effects."""
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
    transitions = {
        "activity_a": PreparedTransition(
            activity_name="activity_a", input_places=["p_start"], preconditions=[], effect_groups=[],
        ),
        "activity_b": PreparedTransition(
            activity_name="activity_b", input_places=["p_mid"], preconditions=[], effect_groups=[],
        ),
    }
    return PreparedDomainInput(
        nodes=nodes, edges=edges, transitions=transitions, xor_branches={}, attribute_catalog={},
    )


def _tau_prepared() -> PreparedDomainInput:
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
    transitions = {
        "activity_a": PreparedTransition(
            activity_name="activity_a", input_places=["p_mid"], preconditions=[], effect_groups=[],
        ),
    }
    return PreparedDomainInput(
        nodes=nodes, edges=edges, transitions=transitions, xor_branches={}, attribute_catalog={},
    )


class TestEndToEnd:

    def test_full_pipeline_produces_valid_structure(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        text = str(domain)

        assert "(define (domain test_process)" in text
        assert "(:requirements" in text
        assert "(:types" in text
        assert "(:constants" in text
        assert "(:predicates" in text
        assert "(:action" in text

    def test_full_pipeline_with_tau(self):
        domain = build_domain_from_prepared_info(_tau_prepared())
        text = str(domain)

        assert "execute_tau_0" in text
        assert "(marked tau_0)" in text

    def test_pddl_parentheses_balanced(self):
        domain = build_domain_from_prepared_info(_sequence_prepared())
        text = str(domain)

        assert text.count("(") == text.count(")")


class TestActionCosts:

    def _domain_with_costs(self, base_cost=1.5):
        return PDDLDomain(
            name="test",
            requirements=[":strips", ":typing", ":action-costs", ":numeric-fluents"],
            types=PDDLTypes(items=[
                PDDLType("petri_element"), PDDLType("place", parent="petri_element"),
                PDDLType("transition", parent="petri_element"),
            ]),
            constants=PDDLConstants(items=[PDDLObject("p1", "place"), PDDLObject("t1", "transition")]),
            predicates=PDDLPredicates(items=[PDDLPredicate("marked", [("?x", "petri_element")])]),
            actions=[
                PDDLAction(
                    name="execute_t1",
                    preconditions={PDDLCondition.marked("p1")},
                    effects=[PDDLEffect.marking("t1")],
                    base_cost=base_cost,
                ),
            ],
            has_costs=True,
        )

    def test_no_functions_section_when_no_costs(self, minimal_domain):
        text = str(minimal_domain)
        assert "(:functions" not in text
        assert "total-cost" not in text

    def test_functions_section_emitted_when_has_costs(self):
        domain = self._domain_with_costs()
        text = str(domain)
        assert "(:functions" in text
        assert "(total-cost)" in text

    def test_increase_effect_in_action_when_has_costs(self):
        domain = self._domain_with_costs(base_cost=1.5)
        text = str(domain)
        assert "(increase (total-cost)" in text
        assert "1.5000" in text

    def test_no_increase_effect_when_no_costs(self, minimal_domain):
        text = str(minimal_domain)
        assert "(increase" not in text

    def test_default_cost_one_when_action_cost_is_zero(self):
        domain = self._domain_with_costs(base_cost=0.0)
        domain.actions[0].additional_cost = None
        text = str(domain)
        assert "(increase (total-cost) 1.0000)" in text

    def test_cost_uses_sum_of_base_and_additional(self):
        domain = self._domain_with_costs(base_cost=1.0)
        domain.actions[0].additional_cost = 0.5
        text = str(domain)
        assert "1.5000" in text

    def test_action_costs_requirement_in_output(self):
        domain = self._domain_with_costs()
        text = str(domain)
        assert ":action-costs" in text
        assert ":numeric-fluents" in text

    def test_parentheses_balanced_with_costs(self):
        domain = self._domain_with_costs()
        text = str(domain)
        assert text.count("(") == text.count(")")

    def test_place_marking_actions_get_default_cost_one(self):
        domain = PDDLDomain(
            name="test",
            requirements=[":strips", ":typing", ":action-costs", ":numeric-fluents"],
            types=PDDLTypes(items=[
                PDDLType("petri_element"), PDDLType("place", parent="petri_element"),
                PDDLType("transition", parent="petri_element"),
            ]),
            constants=PDDLConstants(items=[PDDLObject("p1", "place"), PDDLObject("t1", "transition")]),
            predicates=PDDLPredicates(items=[PDDLPredicate("marked", [("?x", "petri_element")])]),
            actions=[
                PDDLAction(
                    name="execute_t1",
                    preconditions={PDDLCondition.marked("p1")},
                    effects=[PDDLEffect.marking("t1")],
                ),
            ],
            has_costs=True,
        )
        text = str(domain)
        assert "(increase (total-cost) 1.0000)" in text


# ---------------------------------------------------------------------------
# Deadline
#
# PDDLDomain.deadline_predicate defaults to "deadline_ok" (a positive-framing
# predicate: true while the deadline has not been exceeded). Both PDDLAction
# and PDDLDurativeAction render it as a bare atom, not negated — there is
# nothing to negate, since "ok" already means "not exceeded".
# ---------------------------------------------------------------------------

class TestDeadline:

    def _durative_domain(self, has_deadline: bool = False) -> PDDLDomain:
        action = PDDLDurativeAction(
            name="execute_register",
            conditions_at_start=[PDDLCondition.marked("p_start")],
            conditions_over_all=[],
            conditions_at_end=[],
            effects_at_start=[PDDLEffect(kind="marked", attribute="p_start", clear=True)],
            effects_at_end=[PDDLEffect.marking("register")],
            duration_min=10.0,
            duration_max=30.0,
        )
        return PDDLDomain(
            name="test",
            requirements=[":strips", ":typing", ":durative-actions"],
            types=PDDLTypes(items=[PDDLType("petri_element")]),
            constants=PDDLConstants(),
            predicates=PDDLPredicates(items=[PDDLPredicate("marked", [("?x", "petri_element")])]),
            actions=[action],
            has_deadline=has_deadline,
        )

    def test_no_deadline_predicate_by_default(self):
        text = str(self._durative_domain(has_deadline=False))
        assert "deadline_ok" not in text

    def test_deadline_predicate_added_when_has_deadline(self):
        text = str(self._durative_domain(has_deadline=True))
        assert "(deadline_ok)" in text

    def test_no_over_all_deadline_without_flag(self):
        text = str(self._durative_domain(has_deadline=False))
        assert "(over all (deadline_ok))" not in text

    def test_instantaneous_action_no_over_all(self):
        domain = PDDLDomain(
            name="test",
            requirements=[":strips", ":typing"],
            types=PDDLTypes(items=[PDDLType("petri_element")]),
            constants=PDDLConstants(),
            predicates=PDDLPredicates(items=[PDDLPredicate("marked", [("?x", "petri_element")])]),
            actions=[PDDLAction(
                name="exec",
                preconditions={PDDLCondition.marked("p1")},
                effects=[PDDLEffect.marking("t1")],
            )],
            has_deadline=True,
        )
        text = str(domain)
        assert "(over all" not in text

    def test_instantaneous_action_gets_deadline_precondition(self):
        domain = PDDLDomain(
            name="test",
            requirements=[":strips", ":typing"],
            types=PDDLTypes(items=[PDDLType("petri_element")]),
            constants=PDDLConstants(),
            predicates=PDDLPredicates(items=[PDDLPredicate("marked", [("?x", "petri_element")])]),
            actions=[PDDLAction(
                name="exec",
                preconditions={PDDLCondition.marked("p1")},
                effects=[PDDLEffect.marking("t1")],
            )],
            has_deadline=True,
        )
        text = str(domain)
        assert "(deadline_ok)" in text

    def test_parentheses_balanced_with_deadline(self):
        text = str(self._durative_domain(has_deadline=True))
        assert text.count("(") == text.count(")")


class TestPDDLDurativeActionModel:

    def test_is_subclass_of_base(self):
        action = PDDLDurativeAction(name="test", duration_min=5.0, duration_max=15.0)
        assert isinstance(action, PDDLBaseAction)
        assert not isinstance(action, PDDLAction)

    def test_pddl_action_is_subclass_of_base(self):
        action = PDDLAction(name="test")
        assert isinstance(action, PDDLBaseAction)
        assert not isinstance(action, PDDLDurativeAction)

    def test_default_fields(self):
        action = PDDLDurativeAction(name="d", duration_min=1.0, duration_max=2.0)
        assert action.parameters == []
        assert action.conditions_at_start == set()
        assert action.conditions_over_all == set()
        assert action.conditions_at_end == set()
        assert action.effects_at_start == []
        assert action.effects_at_end == []

    def test_all_fields_set(self):
        action = PDDLDurativeAction(
            name="act",
            duration_min=10.0,
            duration_max=30.0,
            conditions_at_start={PDDLCondition.marked("p1")},
            conditions_over_all={PDDLCondition.attr_true("some_inv")},
            conditions_at_end={PDDLCondition.attr_true("goal_cond")},
            effects_at_start=[PDDLEffect.marking("start_eff")],
            effects_at_end=[PDDLEffect.marking("t1"), PDDLEffect.set_attr_is("diagnosis", "flu")],
        )
        assert action.duration_min == 10.0
        assert action.duration_max == 30.0
        assert action.conditions_at_start == {PDDLCondition.marked("p1")}
        assert action.conditions_over_all == {PDDLCondition.attr_true("some_inv")}
        assert action.conditions_at_end == {PDDLCondition.attr_true("goal_cond")}
        assert action.effects_at_start == [PDDLEffect.marking("start_eff")]
        assert action.effects_at_end == [
            PDDLEffect.marking("t1"),
            PDDLEffect.set_attr_is("diagnosis", "flu"),
        ]


class TestDurativeActionRendering:
    """Rendering of a durative action alongside an instantaneous one, built
    directly (no domain-builder involved) to isolate pddl_model.py's own
    __str__ logic."""

    def _domain(self) -> PDDLDomain:
        durative = PDDLDurativeAction(
            name="execute_activity_a",
            duration_min=10.0,
            duration_max=30.0,
            conditions_at_start={PDDLCondition.marked("p_start")},
            effects_at_end=[PDDLEffect.marking("activity_a")],
        )
        instantaneous = PDDLAction(
            name="execute_activity_b",
            preconditions={PDDLCondition.marked("p_mid")},
            effects=[PDDLEffect.marking("activity_b")],
        )
        return PDDLDomain(
            name="test",
            requirements=[":strips", ":typing", ":durative-actions", ":duration-inequalities"],
            types=PDDLTypes(items=[PDDLType("petri_element")]),
            constants=PDDLConstants(),
            predicates=PDDLPredicates(items=[PDDLPredicate("marked", [("?x", "petri_element")])]),
            actions=[durative, instantaneous],
        )

    def test_durative_action_keyword_in_output(self):
        text = str(self._domain())
        assert "(:durative-action execute_activity_a" in text

    def test_duration_constraint_in_output(self):
        text = str(self._domain())
        assert ":duration" in text
        assert ">= ?duration 10.0" in text
        assert "<= ?duration 30.0" in text

    def test_at_start_condition_in_output(self):
        text = str(self._domain())
        assert "at start" in text
        assert "(marked p_start)" in text

    def test_at_end_effect_in_output(self):
        text = str(self._domain())
        assert "at end" in text
        assert "(marked activity_a)" in text

    def test_instantaneous_action_still_uses_action_keyword(self):
        text = str(self._domain())
        assert "(:action execute_activity_b" in text

    def test_durative_requirement_in_output(self):
        text = str(self._domain())
        assert ":durative-actions" in text

    def test_parentheses_balanced_with_durative(self):
        text = str(self._domain())
        assert text.count("(") == text.count(")")


class TestMultiWordAttributeNameRendering:
    """PDDLCondition/PDDLEffect.to_pddl() must sanitize the attribute name
    used to build the VALUE constant, not just the predicate name -- both
    to_pddl() must call sanitize_value(attr, ...) with the already-sanitized
    `attr`, not the raw self.attribute. A raw attribute name containing a
    space (e.g. "organization involved", straight from an XES column) would
    otherwise leak that space into the value constant (e.g.
    "organization involved_val_none" instead of
    "organization_involved_val_none"), producing a 2-token predicate call
    that doesn't match any :constants declaration."""

    def test_condition_value_constant_uses_sanitized_attribute(self):
        cond = PDDLCondition.attr_is("organization involved", "none")
        assert cond.to_pddl() == "(organization_involved_is organization_involved_val_none)"

    def test_condition_attr_is_not_value_constant_uses_sanitized_attribute(self):
        cond = PDDLCondition.attr_is_not("organization involved", "none")
        assert cond.to_pddl() == "(organization_involved_is_not organization_involved_val_none)"

    def test_effect_value_constant_uses_sanitized_attribute(self):
        eff = PDDLEffect.set_attr_is("organization involved", "none")
        assert eff.to_pddl() == "(organization_involved_is organization_involved_val_none)"

    def test_effect_clear_value_constant_uses_sanitized_attribute(self):
        eff = PDDLEffect.clear_attr_is("organization involved", "none")
        assert eff.to_pddl() == "(not (organization_involved_is organization_involved_val_none))"
