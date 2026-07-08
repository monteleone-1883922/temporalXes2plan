"""Tests for models.Guard.conflicts_with / Guard.first_conflict."""
from models import Guard


class TestConflictsWith:
    def test_different_attributes_never_conflict(self):
        a = Guard(attribute="risk", value="high", negated=False)
        b = Guard(attribute="urgent", value=None, negated=False)
        assert not a.conflicts_with(b)

    def test_same_value_opposite_negation_conflicts(self):
        a = Guard(attribute="risk", value="high", negated=False)
        b = Guard(attribute="risk", value="high", negated=True)
        assert a.conflicts_with(b)
        assert b.conflicts_with(a)

    def test_boolean_true_false_conflicts(self):
        a = Guard(attribute="urgent", value=None, negated=False)
        b = Guard(attribute="urgent", value=None, negated=True)
        assert a.conflicts_with(b)

    def test_different_non_negated_values_conflict(self):
        a = Guard(attribute="risk", value="high", negated=False)
        b = Guard(attribute="risk", value="low", negated=False)
        assert a.conflicts_with(b)

    def test_same_value_same_negation_does_not_conflict(self):
        a = Guard(attribute="risk", value="high", negated=False)
        b = Guard(attribute="risk", value="high", negated=False)
        assert not a.conflicts_with(b)

    def test_different_negated_values_do_not_conflict(self):
        # "not high" and "not low" can both hold (e.g. value is "medium")
        a = Guard(attribute="risk", value="high", negated=True)
        b = Guard(attribute="risk", value="low", negated=True)
        assert not a.conflicts_with(b)


class TestFirstConflict:
    def test_no_conflict_returns_none(self):
        guards = [
            Guard(attribute="risk", value="high", negated=False),
            Guard(attribute="urgent", value=None, negated=False),
        ]
        assert Guard.first_conflict(guards) is None

    def test_finds_conflicting_pair(self):
        g1 = Guard(attribute="risk", value="high", negated=False)
        g2 = Guard(attribute="risk", value="low", negated=False)
        result = Guard.first_conflict([g1, g2])
        assert result == (g1, g2)

    def test_empty_collection_returns_none(self):
        assert Guard.first_conflict([]) is None

    def test_accepts_a_generator(self):
        g1 = Guard(attribute="risk", value="high", negated=False)
        g2 = Guard(attribute="risk", value="low", negated=False)
        result = Guard.first_conflict(g for g in [g1, g2])
        assert result == (g1, g2)
