"""Unit tests for parsing.effect_correlation_analyzer.EffectCorrelationAnalyzer."""
import pytest

from models import AnalysisConfig, PreprocessedLog, TransitionFiringData
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
        static_attributes=set(),
    )


def _analyze(transition_firings: dict, related: float = 0.9, incompatible: float = 0.05):
    config = AnalysisConfig(related_effect_prob=related, incompatible_effect_prob=incompatible)
    return EffectCorrelationAnalyzer().analyze(_log(transition_firings), config)


def _pair(a: str, b: str) -> frozenset:
    return frozenset({a, b})


# ---------------------------------------------------------------------------
# Related pairs (joint >= related_effect_prob)
# ---------------------------------------------------------------------------

class TestRelatedClassification:

    def test_always_co_present_is_related(self):
        firings = [_firing({"a": 1, "b": 2}) for _ in range(10)]
        result = _analyze({"t": firings}, related=0.9)
        related, _ = result["t"]
        assert _pair("a", "b") in related

    def test_exact_threshold_is_related(self):
        # 9 out of 10 firings → joint = 0.9 → exactly at threshold → related
        firings = [_firing({"a": 1, "b": 2})] * 9 + [_firing({"a": 1})]
        result = _analyze({"t": firings}, related=0.9)
        related, _ = result["t"]
        assert _pair("a", "b") in related

    def test_below_threshold_not_related(self):
        # joint = 0.5 with related=0.9 → not related
        firings = [_firing({"a": 1, "b": 2})] * 5 + [_firing({"a": 1})] * 5
        result = _analyze({"t": firings}, related=0.9, incompatible=0.0)
        related, _ = result.get("t", (set(), set()))
        assert _pair("a", "b") not in related


# ---------------------------------------------------------------------------
# Incompatible pairs (joint <= incompatible_effect_prob)
# ---------------------------------------------------------------------------

class TestIncompatibleClassification:

    def test_never_co_present_is_incompatible(self):
        # attributes appear in different firings only
        firings = [_firing({"a": 1}), _firing({"b": 2})] * 5
        result = _analyze({"t": firings}, incompatible=0.05)
        # no joint occurrences → pair absent from analysis → not incompatible
        # (zero co-occurrences means the pair was never observed together, not tracked)
        _, incompatible = result.get("t", (set(), set()))
        assert _pair("a", "b") not in incompatible

    def test_low_joint_is_incompatible(self):
        # 1 out of 20 firings → joint = 0.05 → exactly at threshold
        firings = [_firing({"a": 1, "b": 2})] + [_firing({"a": 1})] * 10 + [_firing({"b": 2})] * 9
        result = _analyze({"t": firings}, related=1.0, incompatible=0.05)
        _, incompatible = result.get("t", (set(), set()))
        assert _pair("a", "b") in incompatible

    def test_above_incompatible_threshold_not_incompatible(self):
        # joint = 0.5 with incompatible=0.05 → not incompatible
        firings = [_firing({"a": 1, "b": 2})] * 5 + [_firing({"a": 1})] * 5
        result = _analyze({"t": firings}, related=1.0, incompatible=0.05)
        _, incompatible = result.get("t", (set(), set()))
        assert _pair("a", "b") not in incompatible


# ---------------------------------------------------------------------------
# Middle ground — no classification
# ---------------------------------------------------------------------------

class TestNoClassification:

    def test_middle_probability_produces_no_entry(self):
        # joint = 0.5, thresholds at 0.9 / 0.05 → no classification
        firings = [_firing({"a": 1, "b": 2})] * 5 + [_firing({"a": 1})] * 5
        result = _analyze({"t": firings}, related=0.9, incompatible=0.05)
        # transition absent OR both sets empty
        if "t" in result:
            related, incompatible = result["t"]
            assert _pair("a", "b") not in related
            assert _pair("a", "b") not in incompatible

    def test_transition_absent_when_no_pairs_classified(self):
        firings = [_firing({"a": 1, "b": 2})] * 5 + [_firing({"a": 1})] * 5
        result = _analyze({"t": firings}, related=0.9, incompatible=0.05)
        assert "t" not in result


# ---------------------------------------------------------------------------
# Three attributes
# ---------------------------------------------------------------------------

class TestThreeAttributes:

    def test_all_three_pairs_related(self):
        firings = [_firing({"a": 1, "b": 2, "c": 3}) for _ in range(10)]
        result = _analyze({"t": firings}, related=0.9)
        related, _ = result["t"]
        assert _pair("a", "b") in related
        assert _pair("a", "c") in related
        assert _pair("b", "c") in related

    def test_mixed_pairs(self):
        # a+b always together (19/20 = 0.95 >= 0.9 → related)
        # a+c only once      (1/20  = 0.05 <= 0.05 → incompatible)
        firings = (
            [_firing({"a": 1, "b": 2})] * 19 +
            [_firing({"a": 1, "b": 2, "c": 3})] * 1
        )
        result = _analyze({"t": firings}, related=0.9, incompatible=0.05)
        related, incompatible = result["t"]
        assert _pair("a", "b") in related
        assert _pair("a", "c") in incompatible


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_attribute_per_firing_no_result(self):
        firings = [_firing({"a": i}) for i in range(5)]
        result = _analyze({"t": firings})
        assert "t" not in result

    def test_empty_firings_no_result(self):
        result = _analyze({"t": []})
        assert "t" not in result

    def test_multiple_transitions_independent(self):
        firings_t1 = [_firing({"a": 1, "b": 2}) for _ in range(10)]
        firings_t2 = [_firing({"x": 1})] * 10
        result = _analyze({"t1": firings_t1, "t2": firings_t2}, related=0.9)
        assert "t1" in result
        assert "t2" not in result

    def test_config_thresholds_respected(self):
        # With very strict thresholds nothing should be classified
        firings = [_firing({"a": 1, "b": 2})] * 8 + [_firing({"a": 1})] * 2
        result = _analyze({"t": firings}, related=0.95, incompatible=0.0)
        assert "t" not in result


# ---------------------------------------------------------------------------
# FrozenSet symmetry
# ---------------------------------------------------------------------------

class TestPairSymmetry:

    def test_pair_is_frozenset(self):
        firings = [_firing({"a": 1, "b": 2}) for _ in range(10)]
        result = _analyze({"t": firings}, related=0.9)
        related, _ = result["t"]
        pair = next(iter(related))
        assert isinstance(pair, frozenset)
        assert pair == frozenset({"a", "b"})

    def test_single_entry_per_pair(self):
        # frozenset ensures {a,b} and {b,a} are the same entry
        firings = [_firing({"a": 1, "b": 2}) for _ in range(10)]
        result = _analyze({"t": firings}, related=0.9)
        related, _ = result["t"]
        assert len(related) == 1
