"""Tests for encoding.action_registry.ActionRegistry."""
import pytest

from encoding.action_registry import ActionRegistry
from encoding.pddl_model import PDDLAction, PDDLDurativeAction


@pytest.fixture
def three_actions():
    return [
        PDDLAction(name="mark_p1_from_t1", preconditions=["(marked t1)"], effects=["(marked p1)"]),
        PDDLAction(name="execute_a", preconditions=["(marked p0)"], effects=["(marked a)"]),
        PDDLAction(name="execute_b", preconditions=["(marked p1)"], effects=["(marked b)"]),
    ]


class TestFromBaseActions:

    def test_each_action_has_one_variant(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        for action in three_actions:
            assert len(registry.get(action.name)) == 1

    def test_variant_is_the_original(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        assert registry.get("execute_a")[0] is three_actions[1]

    def test_empty_list(self):
        registry = ActionRegistry.from_base_actions([])
        assert registry.all_actions() == []


class TestAllActions:

    def test_returns_all_single_variants(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        all_act = registry.all_actions()
        assert len(all_act) == 3

    def test_order_matches_insertion(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        names = [a.name for a in registry.all_actions()]
        assert names == ["mark_p1_from_t1", "execute_a", "execute_b"]

    def test_expanded_variants_flattened(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        v0 = PDDLAction(name="execute_a_v0", preconditions=["(marked p0)", "(risk_is high)"])
        v1 = PDDLAction(name="execute_a_v1", preconditions=["(marked p0)", "(risk_is low)"])
        registry.replace("execute_a", [v0, v1])

        all_act = registry.all_actions()
        names = [a.name for a in all_act]
        assert names == ["mark_p1_from_t1", "execute_a_v0", "execute_a_v1", "execute_b"]


class TestReplace:

    def test_replace_updates_variants(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        new_v = PDDLAction(name="execute_a_v0", preconditions=["(cond)"])
        registry.replace("execute_a", [new_v])
        assert registry.get("execute_a") == [new_v]

    def test_replace_unknown_key_raises(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        with pytest.raises(KeyError):
            registry.replace("nonexistent", [])


class TestContains:

    def test_known_key_present(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        assert "execute_a" in registry

    def test_unknown_key_absent(self, three_actions):
        registry = ActionRegistry.from_base_actions(three_actions)
        assert "execute_z" not in registry
