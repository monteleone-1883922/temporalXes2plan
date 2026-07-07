"""Tests for encoding.guard_encoder."""
import pytest

from models import Guard
from encoding.guard_encoder import and_clause_to_conditions, guard_to_condition
from encoding.pddl_model import PDDLCondition


class TestGuardToCondition:

    def test_categorical_positive(self):
        g = Guard(attribute="diagnosis", value="flu", negated=False)
        assert guard_to_condition(g) == PDDLCondition.attr_is("diagnosis", "flu")

    def test_categorical_negated(self):
        g = Guard(attribute="diagnosis", value="flu", negated=True)
        assert guard_to_condition(g) == PDDLCondition.attr_is_not("diagnosis", "flu")

    def test_boolean_positive(self):
        g = Guard(attribute="urgent", value=None, negated=False)
        assert guard_to_condition(g) == PDDLCondition.attr_true("urgent")

    def test_boolean_negated(self):
        g = Guard(attribute="urgent", value=None, negated=True)
        assert guard_to_condition(g) == PDDLCondition.attr_false("urgent")

    def test_numerical_discretized_positive(self):
        g = Guard(attribute="crp", value="lte_10_0", negated=False)
        assert guard_to_condition(g) == PDDLCondition.attr_is("crp", "lte_10_0")

    def test_numerical_discretized_negated(self):
        g = Guard(attribute="crp", value="lte_10_0", negated=True)
        assert guard_to_condition(g) == PDDLCondition.attr_is_not("crp", "lte_10_0")

    def test_pddl_output_categorical_positive(self):
        g = Guard(attribute="diagnosis", value="flu", negated=False)
        assert guard_to_condition(g).to_pddl() == "(diagnosis_is diagnosis_val_flu)"

    def test_pddl_output_boolean_negative(self):
        g = Guard(attribute="urgent", value=None, negated=True)
        assert guard_to_condition(g).to_pddl() == "(urgent_false)"


class TestAndClauseToConditions:

    def test_empty_clause(self):
        assert and_clause_to_conditions([]) == []

    def test_single_guard(self):
        clause = [Guard("diagnosis", "flu", negated=False)]
        result = and_clause_to_conditions(clause)
        assert result == [PDDLCondition.attr_is("diagnosis", "flu")]

    def test_multiple_guards(self):
        clause = [
            Guard("diagnosis", "flu", negated=False),
            Guard("urgent", None, negated=True),
            Guard("crp", "lte_10_0", negated=False),
        ]
        result = and_clause_to_conditions(clause)
        assert result == [
            PDDLCondition.attr_is("diagnosis", "flu"),
            PDDLCondition.attr_false("urgent"),
            PDDLCondition.attr_is("crp", "lte_10_0"),
        ]

    def test_mixed_negation(self):
        clause = [
            Guard("risk", "high", negated=False),
            Guard("admitted", None, negated=True),
        ]
        result = and_clause_to_conditions(clause)
        assert PDDLCondition.attr_is("risk", "high") in result
        assert PDDLCondition.attr_false("admitted") in result

    def test_pddl_output_matches_old_format(self):
        clause = [
            Guard("diagnosis", "flu", negated=False),
            Guard("urgent", None, negated=True),
        ]
        result = [c.to_pddl() for c in and_clause_to_conditions(clause)]
        assert "(diagnosis_is diagnosis_val_flu)" in result
        assert "(urgent_false)" in result
