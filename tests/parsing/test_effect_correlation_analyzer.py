"""Unit tests for parsing.effect_correlation_analyzer.EffectCorrelationAnalyzer."""
import pytest

from models import PreprocessedLog, TransitionFiringData
from parsing.effect_correlation_analyzer import EffectCorrelationAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _firing(changed: dict) -> TransitionFiringData:
    return TransitionFiringData(
        activity_name="t",
        pre_state={},
        changed_attrs=changed,
        from_places=frozenset(),
    )


def _log(transition_firings: dict) -> PreprocessedLog:
    return PreprocessedLog(
        transition_firings=transition_firings,
        xor_firings={},
    )


def _compute(transition_firings: dict) -> dict:
    return EffectCorrelationAnalyzer().compute(_log(transition_firings))


# ---------------------------------------------------------------------------
# Always co-present
# ---------------------------------------------------------------------------

class TestAlwaysCoPresent:

    def test_joint_probability_is_one(self):
        firings = [_firing({"a": 1, "b": 2}) for _ in range(10)]
        result = _compute({"t": firings})
        assert result["t"]["a"]["b"] == pytest.approx(1.0)

    def test_symmetric(self):
        firings = [_firing({"a": 1, "b": 2}) for _ in range(10)]
        result = _compute({"t": firings})
        assert result["t"]["a"]["b"] == result["t"]["b"]["a"]


# ---------------------------------------------------------------------------
# Never co-present (attributes appear in different firings)
# ---------------------------------------------------------------------------

class TestNeverCoPresent:

    def test_pair_absent_when_no_joint_occurrence(self):
        firings = [_firing({"a": 1}), _firing({"b": 2})]
        result = _compute({"t": firings})
        # either the transition is absent or the pair is absent
        joint = result.get("t", {})
        assert "b" not in joint.get("a", {})
        assert "a" not in joint.get("b", {})


# ---------------------------------------------------------------------------
# Partial co-occurrence
# ---------------------------------------------------------------------------

class TestPartialCoOccurrence:

    def test_half_joint(self):
        firings = [
            _firing({"a": 1, "b": 2}),
            _firing({"a": 1, "b": 2}),
            _firing({"a": 1}),
            _firing({"a": 1}),
        ]
        result = _compute({"t": firings})
        assert result["t"]["a"]["b"] == pytest.approx(0.5)

    def test_symmetric_at_partial(self):
        firings = [
            _firing({"a": 1, "b": 2}),
            _firing({"a": 1}),
            _firing({"b": 2}),
        ]
        result = _compute({"t": firings})
        ab = result["t"]["a"]["b"]
        ba = result["t"]["b"]["a"]
        assert ab == ba


# ---------------------------------------------------------------------------
# Three attributes
# ---------------------------------------------------------------------------

class TestThreeAttributes:

    def test_all_three_pairs_present(self):
        firings = [_firing({"a": 1, "b": 2, "c": 3}) for _ in range(4)]
        result = _compute({"t": firings})
        joint = result["t"]
        assert "b" in joint["a"]
        assert "c" in joint["a"]
        assert "c" in joint["b"]

    def test_partial_three_way(self):
        # a+b always, a+c half the time, b+c half the time
        firings = [
            _firing({"a": 1, "b": 2, "c": 3}),
            _firing({"a": 1, "b": 2, "c": 3}),
            _firing({"a": 1, "b": 2}),
            _firing({"a": 1, "b": 2}),
        ]
        result = _compute({"t": firings})
        assert result["t"]["a"]["b"] == pytest.approx(1.0)
        assert result["t"]["a"]["c"] == pytest.approx(0.5)
        assert result["t"]["b"]["c"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_attribute_per_firing_produces_no_entry(self):
        firings = [_firing({"a": 1}), _firing({"b": 2}), _firing({"a": 1})]
        result = _compute({"t": firings})
        assert "t" not in result

    def test_empty_firings_produces_no_entry(self):
        result = _compute({"t": []})
        assert "t" not in result

    def test_empty_changed_attrs_ignored(self):
        firings = [_firing({}), _firing({"a": 1})]
        result = _compute({"t": firings})
        assert "t" not in result

    def test_multiple_transitions_independent(self):
        firings_t1 = [_firing({"a": 1, "b": 2}) for _ in range(4)]
        firings_t2 = [_firing({"x": 1, "y": 2}) for _ in range(2)]
        result = _compute({"t1": firings_t1, "t2": firings_t2})

        assert "a" in result["t1"]
        assert "x" in result["t2"]
        assert "x" not in result.get("t1", {})
        assert "a" not in result.get("t2", {})

    def test_transition_not_in_result_if_no_pairs(self):
        """A transition where every firing changes only one attribute is absent."""
        firings = [_firing({"a": i}) for i in range(5)]
        result = _compute({"solo": firings})
        assert "solo" not in result


# ---------------------------------------------------------------------------
# Symmetry property
# ---------------------------------------------------------------------------

class TestSymmetry:

    def test_symmetry_holds_for_all_pairs(self):
        firings = [
            _firing({"a": 1, "b": 2, "c": 3}),
            _firing({"a": 1, "b": 2}),
            _firing({"b": 2, "c": 3}),
            _firing({"a": 1}),
        ]
        result = _compute({"t": firings})
        joint = result["t"]
        for attr_a, others in joint.items():
            for attr_b, prob in others.items():
                assert joint[attr_b][attr_a] == pytest.approx(prob), (
                    f"Symmetry violated for ({attr_a}, {attr_b})"
                )
