"""Tests for encoding.guard_encoder."""
import pytest

from models import Guard
from encoding.guard_encoder import and_clause_to_pddl, guard_to_pddl


class TestGuardToPddl:

    def test_categorical_positive(self):
        g = Guard(attribute="diagnosis", value="flu", negated=False)
        assert guard_to_pddl(g) == "(diagnosis_is flu)"

    def test_categorical_negated(self):
        g = Guard(attribute="diagnosis", value="flu", negated=True)
        assert guard_to_pddl(g) == "(diagnosis_is_not flu)"

    def test_boolean_positive(self):
        g = Guard(attribute="urgent", value=None, negated=False)
        assert guard_to_pddl(g) == "(urgent_true)"

    def test_boolean_negated(self):
        g = Guard(attribute="urgent", value=None, negated=True)
        assert guard_to_pddl(g) == "(urgent_false)"

    def test_numerical_discretized_positive(self):
        g = Guard(attribute="crp", value="lte_10_0", negated=False)
        assert guard_to_pddl(g) == "(crp_is lte_10_0)"

    def test_numerical_discretized_negated(self):
        g = Guard(attribute="crp", value="lte_10_0", negated=True)
        assert guard_to_pddl(g) == "(crp_is_not lte_10_0)"


class TestAndClauseToPddl:

    def test_empty_clause(self):
        assert and_clause_to_pddl([]) == []

    def test_single_guard(self):
        clause = [Guard("diagnosis", "flu", negated=False)]
        assert and_clause_to_pddl(clause) == ["(diagnosis_is flu)"]

    def test_multiple_guards(self):
        clause = [
            Guard("diagnosis", "flu", negated=False),
            Guard("urgent", None, negated=True),
            Guard("crp", "lte_10_0", negated=False),
        ]
        result = and_clause_to_pddl(clause)
        assert result == [
            "(diagnosis_is flu)",
            "(urgent_false)",
            "(crp_is lte_10_0)",
        ]

    def test_mixed_negation(self):
        clause = [
            Guard("risk", "high", negated=False),
            Guard("admitted", None, negated=True),
        ]
        result = and_clause_to_pddl(clause)
        assert "(risk_is high)" in result
        assert "(admitted_false)" in result
