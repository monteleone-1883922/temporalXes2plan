"""Tests for encoding.action_builder.ActionBuilder."""
import pytest

from models import (
    AttributeCatalogEntry, EffectInfo, EffectGuards, ParseResult,
    PetriNetModel, TransitionInfo,
)
from encoding.action_builder import ActionBuilder
from encoding.pddl_model import PDDLCondition, PDDLEffect


class TestPlaceMarkingActions:

    def test_one_action_per_predecessor(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_place_marking_actions()
        names = {a.name for a in actions}

        assert "mark_p_mid_from_activity_a" in names
        assert "mark_p_end_from_activity_b" in names
        assert len(actions) == 2

    def test_precondition_is_predecessor_marked(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_place_marking_actions()
        action = next(a for a in actions if a.name == "mark_p_mid_from_activity_a")

        assert action.preconditions == {PDDLCondition.marked("activity_a")}

    def test_effect_is_place_marked(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_place_marking_actions()
        action = next(a for a in actions if a.name == "mark_p_mid_from_activity_a")

        assert action.effects == [PDDLEffect.marking("p_mid")]

    def test_no_parameters(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_place_marking_actions()

        for a in actions:
            assert a.parameters == []

    def test_multiple_predecessors_generate_multiple_actions(self, xor_net):
        pr = ParseResult(
            petri_net_model=xor_net,
            place_predecessors={
                "p_xor": ["source"],
                "p_end_left": ["left_branch"],
                "p_end_right": ["right_branch"],
            },
            transition_predecessors={
                "source": ["p_start"],
                "left_branch": ["p_xor"],
                "right_branch": ["p_xor"],
            },
            transitions={
                "source": TransitionInfo("source", ["p_start"], 100, None, {}),
                "left_branch": TransitionInfo("left_branch", ["p_xor"], 60, None, {}),
                "right_branch": TransitionInfo("right_branch", ["p_xor"], 40, None, {}),
            },
            start_place="p_start",
            end_place="p_end_left",
            attribute_catalog={},
        )
        builder = ActionBuilder(pr)
        actions = builder._build_place_marking_actions()

        assert len(actions) == 3


class TestTransitionActions:

    def test_one_action_per_labeled_transition(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        names = {a.name for a in actions}

        assert names == {"execute_activity_a", "execute_activity_b"}

    def test_preconditions_are_predecessor_places(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert action.preconditions == {PDDLCondition.marked("p_start")}

    def test_multiple_predecessor_places_and_join(self):
        """AND-join: transition with 2 input places."""
        from pm4py.objects.petri_net.obj import PetriNet, Marking

        net = PetriNet("and_join")
        p1 = PetriNet.Place("p1")
        p2 = PetriNet.Place("p2")
        p_end = PetriNet.Place("p_end")
        t_join = PetriNet.Transition("t_join", label="Join")
        net.places.update([p1, p2, p_end])
        net.transitions.add(t_join)
        for src, tgt in [(p1, t_join), (p2, t_join), (t_join, p_end)]:
            net.arcs.add(PetriNet.Arc(src, tgt))

        model = PetriNetModel(
            petrinet=net, initial_marking=Marking(), final_marking=Marking({p_end: 1}),
            activities={"join"}, silent_transitions={},
            trans_inputs={t_join: {p1, p2}}, trans_outputs={t_join: {p_end}},
            xor_splits={}, place_inputs={p_end: [t_join]},
        )
        pr = ParseResult(
            petri_net_model=model,
            place_predecessors={"p_end": ["join"]},
            transition_predecessors={"join": ["p1", "p2"]},
            transitions={
                "join": TransitionInfo("join", ["p1", "p2"], 50, None, {}),
            },
            start_place="p1",
            end_place="p_end",
            attribute_catalog={},
        )
        builder = ActionBuilder(pr)
        actions = builder._build_transition_actions()
        action = actions[0]

        assert action.preconditions == {PDDLCondition.marked("p1"), PDDLCondition.marked("p2")}

    def test_effect_always_includes_marked_self(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()

        for action in actions:
            trans_name = action.name.replace("execute_", "")
            assert PDDLEffect.marking(trans_name) in action.effects


class TestDeterministicEffects:

    def test_categorical_deterministic_included(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.set_attr_is("diagnosis", "flu") in action.effects

    # --- categorical without negated_attributes ---

    def test_categorical_no_is_not_when_not_negated(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert not any(
            e.kind == "attr_is_not" and e.attribute == "diagnosis"
            for e in action.effects
        )

    def test_categorical_still_clears_positive_for_other_values(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.clear_attr_is("diagnosis", "cold") in action.effects
        assert PDDLEffect.clear_attr_is("diagnosis", "covid") in action.effects

    # --- categorical with negated_attributes ---

    def test_categorical_is_not_present_when_negated(self, negated_parse_result):
        builder = ActionBuilder(negated_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.set_attr_is_not("diagnosis", "cold") in action.effects
        assert PDDLEffect.set_attr_is_not("diagnosis", "covid") in action.effects

    def test_categorical_clears_is_not_for_active_value_when_negated(self, negated_parse_result):
        builder = ActionBuilder(negated_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.clear_attr_is_not("diagnosis", "flu") in action.effects

    def test_categorical_clears_positive_for_other_values_when_negated(self, negated_parse_result):
        builder = ActionBuilder(negated_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.clear_attr_is("diagnosis", "cold") in action.effects
        assert PDDLEffect.clear_attr_is("diagnosis", "covid") in action.effects

    # --- boolean without negated_attributes ---

    def test_boolean_true_no_clear_false_when_not_negated(self, simple_parse_result, boolean_effect):
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean", possible_values=set(),
        )
        simple_parse_result.transitions["activity_a"].effects["urgent"] = boolean_effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.set_attr_true("urgent") in action.effects
        assert PDDLEffect.clear_attr_false("urgent") not in action.effects

    def test_boolean_false_clears_true_when_not_negated(self, simple_parse_result):
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean", possible_values=set(),
        )
        effect = EffectInfo(
            attribute="urgent", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={False: 1.0},
            value_level=1, value_guards=None,
        )
        simple_parse_result.transitions["activity_a"].effects["urgent"] = effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.clear_attr_true("urgent") in action.effects
        assert PDDLEffect.set_attr_false("urgent") not in action.effects

    # --- boolean with negated_attributes ---

    def test_boolean_true_clears_false_when_negated(self, simple_parse_result, boolean_effect):
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean", possible_values=set(),
        )
        simple_parse_result.negated_attributes = {"urgent"}
        simple_parse_result.transitions["activity_a"].effects["urgent"] = boolean_effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.set_attr_true("urgent") in action.effects
        assert PDDLEffect.clear_attr_false("urgent") in action.effects

    def test_boolean_false_sets_false_when_negated(self, simple_parse_result):
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean", possible_values=set(),
        )
        simple_parse_result.negated_attributes = {"urgent"}
        effect = EffectInfo(
            attribute="urgent", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={False: 1.0},
            value_level=1, value_guards=None,
        )
        simple_parse_result.transitions["activity_a"].effects["urgent"] = effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert PDDLEffect.set_attr_false("urgent") in action.effects
        assert PDDLEffect.clear_attr_true("urgent") in action.effects


class TestNonDeterministicEffectsExcluded:

    def test_multi_value_effect_excluded(self, simple_parse_result, non_deterministic_effect):
        simple_parse_result.attribute_catalog["outcome"] = AttributeCatalogEntry(
            attribute_type="categorical", possible_values={"recovered", "deceased"},
        )
        simple_parse_result.transitions["activity_a"].effects["outcome"] = non_deterministic_effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert not any(e.attribute == "outcome" for e in action.effects)

    def test_effect_with_guards_excluded(self, simple_parse_result):
        """Even single-value effects with guards are not deterministic."""
        from models import Guard
        guarded_effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={"appears": [[Guard("risk", "high")]]},
                total_samples=50, dt_accuracy=0.85,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        simple_parse_result.transitions["activity_a"].effects["diagnosis"] = guarded_effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert not any(
            e.attribute == "diagnosis" and not e.clear for e in action.effects
        )

    def test_level3_effect_excluded(self, simple_parse_result):
        level3_effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.3,
            appearance_level=3, appearance_guards=None,
            value_probabilities={}, value_level=3, value_guards=None,
        )
        simple_parse_result.transitions["activity_a"].effects["diagnosis"] = level3_effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert not any(e.attribute == "diagnosis" for e in action.effects)

    def test_appearance_level2_excluded(self, simple_parse_result):
        """appearance_level=2 means not certainly deterministic."""
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.9,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        simple_parse_result.transitions["activity_a"].effects["diagnosis"] = effect

        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_transition_actions()
        action = next(a for a in actions if a.name == "execute_activity_a")

        assert not any(e.attribute == "diagnosis" for e in action.effects)


class TestTauActions:

    def test_tau_action_created(self, tau_parse_result):
        builder = ActionBuilder(tau_parse_result)
        actions = builder._build_tau_actions()

        assert len(actions) == 1
        assert actions[0].name == "execute_tau_0"

    def test_tau_precondition_is_input_place(self, tau_parse_result):
        builder = ActionBuilder(tau_parse_result)
        actions = builder._build_tau_actions()

        assert PDDLCondition.marked("p_start") in actions[0].preconditions

    def test_tau_effect_is_marked_self(self, tau_parse_result):
        builder = ActionBuilder(tau_parse_result)
        actions = builder._build_tau_actions()

        assert PDDLEffect.marking("tau_0") in actions[0].effects

    def test_no_tau_actions_when_none_exist(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder._build_tau_actions()

        assert actions == []


class TestBuildAll:

    def test_build_all_combines_all_action_types(self, tau_parse_result):
        builder = ActionBuilder(tau_parse_result)
        actions = builder.build_all()
        names = {a.name for a in actions}

        assert "mark_p_mid_from_tau_0" in names
        assert "mark_p_end_from_activity_a" in names
        assert "execute_activity_a" in names
        assert "execute_tau_0" in names

    def test_no_duplicate_action_names(self, simple_parse_result):
        builder = ActionBuilder(simple_parse_result)
        actions = builder.build_all()
        names = [a.name for a in actions]

        assert len(names) == len(set(names))
