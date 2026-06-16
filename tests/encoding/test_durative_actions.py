"""Tests for durative action generation and rendering."""
import pytest

from encoding.action_builder import ActionBuilder
from encoding.domain_builder import DomainBuilder
from encoding.pddl_model import PDDLAction, PDDLBaseAction, PDDLDurativeAction
from encoding.pddl_writer import PDDLWriter


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
            conditions_at_start={"(marked p1)"},
            conditions_over_all={"(some_inv)"},
            conditions_at_end={"(goal_cond)"},
            effects_at_start=["(start_eff)"],
            effects_at_end=["(marked t1)", "(diagnosis_is flu)"],
        )
        assert action.duration_min == 10.0
        assert action.duration_max == 30.0
        assert action.conditions_at_start == {"(marked p1)"}
        assert action.conditions_over_all == {"(some_inv)"}
        assert action.conditions_at_end == {"(goal_cond)"}
        assert action.effects_at_start == ["(start_eff)"]
        assert action.effects_at_end == ["(marked t1)", "(diagnosis_is flu)"]


class TestActionBuilderDurative:

    def test_use_durative_false_produces_only_instantaneous(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=False)
        actions = builder._build_transition_actions()

        assert all(isinstance(a, PDDLAction) for a in actions)
        assert not any(isinstance(a, PDDLDurativeAction) for a in actions)

    def test_use_durative_true_transition_with_duration_is_durative(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        actions = builder._build_transition_actions()
        durative = [a for a in actions if isinstance(a, PDDLDurativeAction)]

        assert len(durative) == 1
        assert durative[0].name == "execute_activity_a"

    def test_use_durative_true_transition_without_duration_stays_instantaneous(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        actions = builder._build_transition_actions()
        instantaneous = [a for a in actions if isinstance(a, PDDLAction)]

        assert any(a.name == "execute_activity_b" for a in instantaneous)

    def test_durative_action_duration_values(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if isinstance(a, PDDLDurativeAction))

        assert action.duration_min == 10.0
        assert action.duration_max == 30.0

    def test_durative_action_conditions_at_start_from_preconditions(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if isinstance(a, PDDLDurativeAction))

        assert "(marked p_start)" in action.conditions_at_start

    def test_durative_action_effects_at_end_includes_marked_self(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if isinstance(a, PDDLDurativeAction))

        assert "(marked activity_a)" in action.effects_at_end

    def test_durative_action_deterministic_effects_in_effects_at_end(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if isinstance(a, PDDLDurativeAction))

        assert "(diagnosis_is flu)" in action.effects_at_end

    def test_tau_actions_never_durative(self, tau_parse_result):
        tau_parse_result.transitions["activity_a"].duration = None
        builder = ActionBuilder(tau_parse_result, use_durative=True)
        tau_actions = builder._build_tau_actions()

        assert all(isinstance(a, PDDLAction) for a in tau_actions)
        assert not any(isinstance(a, PDDLDurativeAction) for a in tau_actions)

    def test_place_marking_actions_never_durative(self, durative_parse_result):
        builder = ActionBuilder(durative_parse_result, use_durative=True)
        mark_actions = builder._build_place_marking_actions()

        assert all(isinstance(a, PDDLAction) for a in mark_actions)


class TestDomainBuilderDurative:

    def test_use_durative_false_no_durative_requirement(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=False)
        assert ":durative-actions" not in domain.requirements

    def test_use_durative_true_adds_requirement(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        assert ":durative-actions" in domain.requirements

    def test_no_durative_requirement_when_no_duration_data(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result, use_durative=True)
        assert ":durative-actions" not in domain.requirements

    def test_domain_contains_durative_action(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        durative = [a for a in domain.actions if isinstance(a, PDDLDurativeAction)]
        assert len(durative) == 1

    def test_default_is_instantaneous(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result)
        assert not any(isinstance(a, PDDLDurativeAction) for a in domain.actions)


class TestPDDLWriterDurative:

    def test_durative_action_keyword_in_output(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert "(:durative-action execute_activity_a" in text

    def test_duration_constraint_in_output(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert ":duration" in text
        assert ">= ?duration 10.0" in text
        assert "<= ?duration 30.0" in text

    def test_at_start_condition_in_output(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert "at start" in text
        assert "(marked p_start)" in text

    def test_at_end_effect_in_output(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert "at end" in text
        assert "(marked activity_a)" in text

    def test_instantaneous_action_still_uses_action_keyword(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert "(:action execute_activity_b" in text

    def test_durative_requirement_in_output(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert ":durative-actions" in text

    def test_parentheses_balanced_with_durative(self, durative_parse_result):
        domain = DomainBuilder().build(durative_parse_result, use_durative=True)
        text = PDDLWriter()._render_domain(domain)
        assert text.count("(") == text.count(")")

    def test_render_durative_action_single_condition(self):
        action = PDDLDurativeAction(
            name="test_act",
            duration_min=5.0,
            duration_max=10.0,
            conditions_at_start={"(marked p1)"},
            effects_at_end=["(marked t1)"],
        )
        text = PDDLWriter()._render_durative_action(action)
        assert "(:durative-action test_act" in text
        assert "(at start (marked p1))" in text
        assert "(at end (marked t1))" in text

    def test_render_durative_action_multiple_conditions(self):
        action = PDDLDurativeAction(
            name="multi",
            duration_min=1.0,
            duration_max=2.0,
            conditions_at_start={"(marked p1)", "(marked p2)"},
            conditions_over_all={"(resource_free)"},
            effects_at_end=["(marked t1)", "(diagnosis_is flu)"],
        )
        text = PDDLWriter()._render_durative_action(action)
        assert "(at start" in text
        assert "(over all" in text
        assert "(and" in text

    def test_render_durative_action_empty_conditions(self):
        action = PDDLDurativeAction(
            name="empty",
            duration_min=1.0,
            duration_max=5.0,
        )
        text = PDDLWriter()._render_durative_action(action)
        assert ":condition ()" in text
        assert ":effect ()" in text
