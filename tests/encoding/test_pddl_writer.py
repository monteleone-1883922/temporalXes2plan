"""Tests for encoding.pddl_writer.PDDLWriter."""
import pytest
from pathlib import Path

from encoding.pddl_model import (
    PDDLAction, PDDLCondition, PDDLDomain, PDDLDurativeAction, PDDLEffect, PDDLObject,
    PDDLPredicate, PDDLType,
)
from encoding.pddl_writer import PDDLWriter
from encoding.domain_builder import DomainBuilder


@pytest.fixture
def minimal_domain():
    return PDDLDomain(
        name="test",
        requirements=[":strips", ":typing"],
        types=[
            PDDLType("petri_element"),
            PDDLType("place", parent="petri_element"),
            PDDLType("transition", parent="petri_element"),
        ],
        constants=[
            PDDLObject("p1", "place"),
            PDDLObject("t1", "transition"),
        ],
        predicates=[
            PDDLPredicate("marked", [("?x", "petri_element")]),
        ],
        actions=[
            PDDLAction(
                name="mark_p1_from_t1",
                preconditions={PDDLCondition.marked("t1")},
                effects=[PDDLEffect.marking("p1")],
            ),
        ],
    )


class TestRenderDomain:

    def test_domain_name_in_output(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert "(define (domain test)" in text

    def test_requirements_in_output(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert ":strips" in text
        assert ":typing" in text

    def test_types_grouped_by_parent(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert "place transition - petri_element" in text or \
               "transition place - petri_element" in text

    def test_constants_grouped_by_type(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert "p1 - place" in text
        assert "t1 - transition" in text

    def test_predicate_with_parameters(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert "(marked ?x - petri_element)" in text

    def test_ground_predicate(self):
        domain = PDDLDomain(
            name="test", requirements=[":strips"],
            types=[], constants=[],
            predicates=[PDDLPredicate("urgent_true")],
            actions=[],
        )
        text = PDDLWriter()._render_domain(domain)
        assert "(urgent_true)" in text

    def test_closing_parenthesis(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert text.strip().endswith(")")


class TestRenderAction:

    def test_single_precondition_no_and(self):
        action = PDDLAction(
            name="simple",
            preconditions={PDDLCondition.marked("p1")},
            effects=[PDDLEffect.marking("t1")],
        )
        text = PDDLWriter()._render_action(action)
        assert ":precondition (marked p1)" in text
        assert "(and" not in text

    def test_multiple_preconditions_wrapped_in_and(self):
        action = PDDLAction(
            name="join",
            preconditions={PDDLCondition.marked("p1"), PDDLCondition.marked("p2")},
            effects=[PDDLEffect.marking("t1")],
        )
        text = PDDLWriter()._render_action(action)
        assert ":precondition (and" in text
        assert "(marked p1)" in text
        assert "(marked p2)" in text

    def test_empty_precondition(self):
        action = PDDLAction(name="noop", preconditions=set(), effects=[PDDLEffect.marking("p1")])
        text = PDDLWriter()._render_action(action)
        assert ":precondition ()" in text

    def test_single_effect_no_and(self):
        action = PDDLAction(
            name="simple",
            preconditions={PDDLCondition.marked("p1")},
            effects=[PDDLEffect.marking("t1")],
        )
        text = PDDLWriter()._render_action(action)
        assert ":effect (marked t1)" in text

    def test_multiple_effects_wrapped_in_and(self):
        action = PDDLAction(
            name="multi",
            preconditions=set(),
            effects=[PDDLEffect.marking("t1"), PDDLEffect.set_attr_is("diagnosis", "flu")],
        )
        text = PDDLWriter()._render_action(action)
        assert ":effect (and" in text

    def test_empty_parameters(self):
        action = PDDLAction(name="act", preconditions=set(), effects=[])
        text = PDDLWriter()._render_action(action)
        assert ":parameters ()" in text


class TestWriteToFile:

    def test_writes_file(self, tmp_path, minimal_domain):
        out = tmp_path / "domain.pddl"
        text = PDDLWriter().write_domain(minimal_domain, out)

        assert out.exists()
        assert out.read_text(encoding="utf-8") == text

    def test_creates_parent_directories(self, tmp_path, minimal_domain):
        out = tmp_path / "sub" / "dir" / "domain.pddl"
        PDDLWriter().write_domain(minimal_domain, out)

        assert out.exists()


class TestEndToEnd:

    def test_full_pipeline_produces_valid_structure(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        text = PDDLWriter()._render_domain(domain)

        assert "(define (domain process)" in text
        assert "(:requirements" in text
        assert "(:types" in text
        assert "(:constants" in text
        assert "(:predicates" in text
        assert "(:action" in text

    def test_full_pipeline_with_tau(self, tau_parse_result):
        domain = DomainBuilder().build(tau_parse_result)
        text = PDDLWriter()._render_domain(domain)

        assert "execute_tau_0" in text
        assert "(marked tau_0)" in text

    def test_pddl_parentheses_balanced(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        text = PDDLWriter()._render_domain(domain)

        assert text.count("(") == text.count(")")


class TestActionCosts:

    def _domain_with_costs(self, base_cost=1.5):
        return PDDLDomain(
            name="test",
            requirements=[":strips", ":typing", ":action-costs", ":numeric-fluents"],
            types=[PDDLType("petri_element"), PDDLType("place", parent="petri_element"),
                   PDDLType("transition", parent="petri_element")],
            constants=[PDDLObject("p1", "place"), PDDLObject("t1", "transition")],
            predicates=[PDDLPredicate("marked", [("?x", "petri_element")])],
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
        text = PDDLWriter()._render_domain(minimal_domain)
        assert "(:functions" not in text
        assert "total-cost" not in text

    def test_functions_section_emitted_when_has_costs(self):
        domain = self._domain_with_costs()
        text = PDDLWriter()._render_domain(domain)
        assert "(:functions" in text
        assert "(total-cost)" in text

    def test_increase_effect_in_action_when_has_costs(self):
        domain = self._domain_with_costs(base_cost=1.5)
        text = PDDLWriter()._render_domain(domain)
        assert "(increase (total-cost)" in text
        assert "1.5000" in text

    def test_no_increase_effect_when_no_costs(self, minimal_domain):
        text = PDDLWriter()._render_domain(minimal_domain)
        assert "(increase" not in text

    def test_default_cost_one_when_action_cost_is_zero(self):
        domain = self._domain_with_costs(base_cost=0.0)
        domain.actions[0].additional_cost = None
        text = PDDLWriter()._render_domain(domain)
        assert "(increase (total-cost) 1.0000)" in text

    def test_cost_uses_sum_of_base_and_additional(self):
        domain = self._domain_with_costs(base_cost=1.0)
        domain.actions[0].additional_cost = 0.5
        text = PDDLWriter()._render_domain(domain)
        assert "1.5000" in text

    def test_action_costs_requirement_in_output(self):
        domain = self._domain_with_costs()
        text = PDDLWriter()._render_domain(domain)
        assert ":action-costs" in text
        assert ":numeric-fluents" in text

    def test_parentheses_balanced_with_costs(self):
        domain = self._domain_with_costs()
        text = PDDLWriter()._render_domain(domain)
        assert text.count("(") == text.count(")")

    def test_place_marking_actions_get_default_cost_one(self):
        domain = PDDLDomain(
            name="test",
            requirements=[":strips", ":typing", ":action-costs", ":numeric-fluents"],
            types=[PDDLType("petri_element"), PDDLType("place", parent="petri_element"),
                   PDDLType("transition", parent="petri_element")],
            constants=[PDDLObject("p1", "place"), PDDLObject("t1", "transition")],
            predicates=[PDDLPredicate("marked", [("?x", "petri_element")])],
            actions=[
                PDDLAction(
                    name="execute_t1",
                    preconditions={PDDLCondition.marked("p1")},
                    effects=[PDDLEffect.marking("t1")],
                ),
            ],
            has_costs=True,
        )
        text = PDDLWriter()._render_domain(domain)
        assert "(increase (total-cost) 1.0000)" in text


# ---------------------------------------------------------------------------
# Deadline
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
            types=[PDDLType("petri_element")],
            constants=[],
            predicates=[PDDLPredicate("marked", [("?x", "petri_element")])],
            actions=[action],
            has_deadline=has_deadline,
        )

    def test_no_deadline_predicate_by_default(self):
        text = PDDLWriter()._render_domain(self._durative_domain(has_deadline=False))
        assert "deadline_exceeded" not in text

    def test_deadline_predicate_added_when_has_deadline(self):
        text = PDDLWriter()._render_domain(self._durative_domain(has_deadline=True))
        assert "(deadline_exceeded)" in text

    def test_durative_action_has_over_all_deadline_condition(self):
        text = PDDLWriter()._render_domain(self._durative_domain(has_deadline=True))
        assert "(over all (not (deadline_exceeded)))" in text

    def test_no_over_all_deadline_without_flag(self):
        text = PDDLWriter()._render_domain(self._durative_domain(has_deadline=False))
        assert "(over all (not (deadline_exceeded)))" not in text

    def test_instantaneous_action_no_over_all(self):
        domain = PDDLDomain(
            name="test",
            requirements=[":strips", ":typing"],
            types=[PDDLType("petri_element")],
            constants=[],
            predicates=[PDDLPredicate("marked", [("?x", "petri_element")])],
            actions=[PDDLAction(
                name="exec",
                preconditions={PDDLCondition.marked("p1")},
                effects=[PDDLEffect.marking("t1")],
            )],
            has_deadline=True,
        )
        text = PDDLWriter()._render_domain(domain)
        assert "(over all" not in text

    def test_parentheses_balanced_with_deadline(self):
        text = PDDLWriter()._render_domain(self._durative_domain(has_deadline=True))
        assert text.count("(") == text.count(")")
