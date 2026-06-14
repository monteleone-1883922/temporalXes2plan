"""Tests for encoding.xor_branch_processor.XorBranchProcessor."""
import math

import pytest

from encoding.action_builder import ActionBuilder
from encoding.action_registry import ActionRegistry
from encoding.pddl_model import PDDLAction, PDDLDurativeAction
from encoding.xor_branch_processor import XorBranchProcessor


def _make_registry(parse_result, use_durative=False):
    actions = ActionBuilder(parse_result, use_durative=use_durative).build_all()
    return ActionRegistry.from_base_actions(actions)


class TestSingleClauseGuard:

    def test_no_duplication_with_single_clause(self, xor_parse_result_single_clause):
        registry = _make_registry(xor_parse_result_single_clause)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        left_variants = registry.get("execute_left_branch")
        assert len(left_variants) == 1

    def test_name_unchanged_with_single_clause(self, xor_parse_result_single_clause):
        registry = _make_registry(xor_parse_result_single_clause)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.name == "execute_left_branch"

    def test_guard_condition_added_to_preconditions(self, xor_parse_result_single_clause):
        registry = _make_registry(xor_parse_result_single_clause)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert "(risk_is high)" in variant.preconditions

    def test_negated_guard_added_to_other_branch(self, xor_parse_result_single_clause):
        registry = _make_registry(xor_parse_result_single_clause)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        variant = registry.get("execute_right_branch")[0]
        assert "(risk_is_not high)" in variant.preconditions

    def test_base_preconditions_preserved(self, xor_parse_result_single_clause):
        registry = _make_registry(xor_parse_result_single_clause)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert "(marked p_xor)" in variant.preconditions

    def test_no_cost_assigned_for_level1(self, xor_parse_result_single_clause):
        registry = _make_registry(xor_parse_result_single_clause)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.additional_cost is None


class TestMultiClauseGuard:

    def test_two_or_clauses_produce_two_variants(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        left_variants = registry.get("execute_left_branch")
        assert len(left_variants) == 2

    def test_variants_named_with_suffix(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        names = {v.name for v in registry.get("execute_left_branch")}
        assert names == {"execute_left_branch_v0", "execute_left_branch_v1"}

    def test_first_variant_has_first_clause_conditions(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        variants = {v.name: v for v in registry.get("execute_left_branch")}
        v0 = variants["execute_left_branch_v0"]
        assert "(risk_is high)" in v0.preconditions

    def test_second_variant_has_second_clause_conditions(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        variants = {v.name: v for v in registry.get("execute_left_branch")}
        v1 = variants["execute_left_branch_v1"]
        assert "(urgent_true)" in v1.preconditions
        assert "(crp_is lte_10_0)" in v1.preconditions

    def test_base_preconditions_in_all_variants(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        for variant in registry.get("execute_left_branch"):
            assert "(marked p_xor)" in variant.preconditions

    def test_no_duplication_for_actions_without_guards(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        assert len(registry.get("execute_source")) == 1

    def test_all_actions_flattened_correctly(self, xor_parse_result_multi_clause):
        registry = _make_registry(xor_parse_result_multi_clause)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        all_names = [a.name for a in registry.all_actions()]
        assert "execute_left_branch_v0" in all_names
        assert "execute_left_branch_v1" in all_names
        assert "execute_left_branch" not in all_names


class TestCascadeLevel2:

    def test_no_duplication(self, xor_parse_result_no_guards):
        registry = _make_registry(xor_parse_result_no_guards)
        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        assert len(registry.get("execute_left_branch")) == 1
        assert len(registry.get("execute_right_branch")) == 1

    def test_no_conditions_added(self, xor_parse_result_no_guards):
        registry = _make_registry(xor_parse_result_no_guards)
        original_preconds = list(registry.get("execute_left_branch")[0].preconditions)

        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.preconditions == original_preconds

    def test_additional_cost_assigned(self, xor_parse_result_no_guards):
        registry = _make_registry(xor_parse_result_no_guards)
        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.additional_cost is not None
        assert variant.additional_cost == pytest.approx(-math.log(0.6))

    def test_additional_cost_matches_probability(self, xor_parse_result_no_guards):
        registry = _make_registry(xor_parse_result_no_guards)
        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        left = registry.get("execute_left_branch")[0]
        right = registry.get("execute_right_branch")[0]
        assert left.additional_cost == pytest.approx(-math.log(0.6))
        assert right.additional_cost == pytest.approx(-math.log(0.4))

    def test_base_cost_untouched(self, xor_parse_result_no_guards):
        registry = _make_registry(xor_parse_result_no_guards)
        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.base_cost is None

    def test_name_unchanged(self, xor_parse_result_no_guards):
        registry = _make_registry(xor_parse_result_no_guards)
        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.name == "execute_left_branch"


class TestCascadeLevel3:

    def test_no_duplication(self, xor_parse_result_level3):
        registry = _make_registry(xor_parse_result_level3)
        XorBranchProcessor(xor_parse_result_level3).apply(registry)

        assert len(registry.get("execute_left_branch")) == 1
        assert len(registry.get("execute_right_branch")) == 1

    def test_no_conditions_added(self, xor_parse_result_level3):
        registry = _make_registry(xor_parse_result_level3)
        original_preconds = list(registry.get("execute_left_branch")[0].preconditions)

        XorBranchProcessor(xor_parse_result_level3).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert variant.preconditions == original_preconds

    def test_no_cost_assigned(self, xor_parse_result_level3):
        registry = _make_registry(xor_parse_result_level3)
        XorBranchProcessor(xor_parse_result_level3).apply(registry)

        left = registry.get("execute_left_branch")[0]
        right = registry.get("execute_right_branch")[0]
        assert left.additional_cost is None
        assert right.additional_cost is None

    def test_name_unchanged(self, xor_parse_result_level3):
        registry = _make_registry(xor_parse_result_level3)
        XorBranchProcessor(xor_parse_result_level3).apply(registry)

        assert registry.get("execute_left_branch")[0].name == "execute_left_branch"


class TestDurativeActionsWithGuards:

    def test_guard_added_to_conditions_at_start(
        self, xor_parse_result_single_clause, duration_stats
    ):
        xor_parse_result_single_clause.transitions["left_branch"].duration = duration_stats
        registry = _make_registry(xor_parse_result_single_clause, use_durative=True)
        XorBranchProcessor(xor_parse_result_single_clause).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert isinstance(variant, PDDLDurativeAction)
        assert "(risk_is high)" in variant.conditions_at_start

    def test_multi_clause_durative_renamed(
        self, xor_parse_result_multi_clause, duration_stats
    ):
        xor_parse_result_multi_clause.transitions["left_branch"].duration = duration_stats
        registry = _make_registry(xor_parse_result_multi_clause, use_durative=True)
        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        names = {v.name for v in registry.get("execute_left_branch")}
        assert names == {"execute_left_branch_v0", "execute_left_branch_v1"}

    def test_level2_cost_on_durative_action(
        self, xor_parse_result_no_guards, duration_stats
    ):
        xor_parse_result_no_guards.transitions["left_branch"].duration = duration_stats
        registry = _make_registry(xor_parse_result_no_guards, use_durative=True)
        XorBranchProcessor(xor_parse_result_no_guards).apply(registry)

        variant = registry.get("execute_left_branch")[0]
        assert isinstance(variant, PDDLDurativeAction)
        assert variant.additional_cost == pytest.approx(-math.log(0.6))


class TestXorTauStructure:
    """Verify encoding behavior when a virtual xor_tau_ has been injected."""

    def test_tau_action_present_in_registry(self, xor_parse_result_with_tau):
        registry = _make_registry(xor_parse_result_with_tau)
        assert "execute_xor_tau_left_branch" in registry

    def test_tau_action_has_cost(self, xor_parse_result_with_tau):
        registry = _make_registry(xor_parse_result_with_tau)
        XorBranchProcessor(xor_parse_result_with_tau).apply(registry)

        tau = registry.get("execute_xor_tau_left_branch")[0]
        assert tau.additional_cost == pytest.approx(-math.log(0.6))

    def test_real_action_has_no_cost(self, xor_parse_result_with_tau):
        registry = _make_registry(xor_parse_result_with_tau)
        XorBranchProcessor(xor_parse_result_with_tau).apply(registry)

        real = registry.get("execute_left_branch")[0]
        assert real.additional_cost is None

    def test_real_action_precondition_is_tau_marking(self, xor_parse_result_with_tau):
        registry = _make_registry(xor_parse_result_with_tau)
        real = registry.get("execute_left_branch")[0]
        assert "(marked xor_tau_left_branch)" in real.preconditions

    def test_tau_action_precondition_is_xor_place(self, xor_parse_result_with_tau):
        registry = _make_registry(xor_parse_result_with_tau)
        tau = registry.get("execute_xor_tau_left_branch")[0]
        assert "(marked p_xor)" in tau.preconditions

    def test_no_duplication_of_tau(self, xor_parse_result_with_tau):
        registry = _make_registry(xor_parse_result_with_tau)
        XorBranchProcessor(xor_parse_result_with_tau).apply(registry)

        assert len(registry.get("execute_xor_tau_left_branch")) == 1

    def test_right_branch_also_gets_cost(self, xor_parse_result_with_tau):
        """right_branch has no appearance_level=2 effect → no tau, cost on action."""
        registry = _make_registry(xor_parse_result_with_tau)
        XorBranchProcessor(xor_parse_result_with_tau).apply(registry)

        right = registry.get("execute_right_branch")[0]
        assert right.additional_cost == pytest.approx(-math.log(0.4))


class TestCartesianProduct:

    def test_existing_two_variants_times_two_clauses(self, xor_parse_result_multi_clause):
        """If the registry already has 2 variants, and there are 2 OR-clauses,
        the result must be 4 variants."""
        registry = _make_registry(xor_parse_result_multi_clause)

        v0 = PDDLAction(name="execute_left_branch_pre0", preconditions=["(marked p_xor)"])
        v1 = PDDLAction(name="execute_left_branch_pre1", preconditions=["(marked p_xor)"])
        registry.replace("execute_left_branch", [v0, v1])

        XorBranchProcessor(xor_parse_result_multi_clause).apply(registry)

        result = registry.get("execute_left_branch")
        assert len(result) == 4
