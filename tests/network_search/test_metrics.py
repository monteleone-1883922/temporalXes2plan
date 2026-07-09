"""Unit tests for network_search.metrics — synthetic screening/transition
fixtures, no dependency on a real log (docs/network_improvement_loop_plan.md
§9 phase 2)."""
import pytest

from encoding.prepared_input import PreparedEffectGroup, PreparedTransition
from models import (
    EffectAttrScreening,
    EffectValueScreening,
    TransitionScreening,
    XorBranchScreening,
    XorSplitScreening,
)
from network_search.metrics import (
    compute_coverage,
    compute_duplication_score,
    compute_xor_score,
    compute_effect_score,
    compute_reproducibility,
)
from network_search.scoring import ScoreWeights
from replay.trace_replayer import ReplayOutcome

WEIGHTS = ScoreWeights(w_det_xor=1.0, w_det_eff=1.0, w_fb_xor=0.5, w_fb_eff=0.5, w_prune_xor=1.0, w_xor=1.0, w_eff=1.0, w_dup=0.2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xor_branch(status: str, activity_name: str = "act", probability: float = 0.5) -> XorBranchScreening:
    return XorBranchScreening(activity_name=activity_name, probability=probability, status=status)


def _effect_attr(appearance_action: str, value_action: str) -> EffectAttrScreening:
    return EffectAttrScreening(
        presence_probability=0.5,
        appearance_action=appearance_action,
        appearance_samples=50,
        value_action=value_action,
        value_samples=50,
        values={"v": EffectValueScreening(probability=0.5, status="active")},
    )


def _prepared_transition(n_effect_groups: int) -> PreparedTransition:
    return PreparedTransition(
        activity_name="act",
        input_places=["p1"],
        preconditions=[],
        effect_groups=[
            PreparedEffectGroup(assignments=[("attr", "val")], guard=[], probability=1.0)
            for _ in range(n_effect_groups)
        ],
    )


# ---------------------------------------------------------------------------
# compute_xor_score — Asse A (XOR)
# ---------------------------------------------------------------------------

class TestComputeXorScore:
    def test_empty_inputs_return_neutral(self):
        assert compute_xor_score({}, WEIGHTS) == 0.5

    def test_certain_branch_is_rewarded(self):
        xor_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="deterministic",
                branches={"a": _xor_branch("certain", "a")},
            )
        }
        expected = 0.5 + 0.5 * WEIGHTS.w_det_xor
        assert compute_xor_score(xor_screening, WEIGHTS) == pytest.approx(expected)

    def test_pruned_branch_is_penalized_more_than_fallback(self):
        pruned_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="dt",
                branches={"a": _xor_branch("pruned", "a")},
            )
        }
        fallback_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="fallback",
                branches={"a": _xor_branch("fallback", "a")},
            )
        }
        pruned_score = compute_xor_score(pruned_screening, WEIGHTS)
        fallback_score = compute_xor_score(fallback_screening, WEIGHTS)
        assert pruned_score < fallback_score < 0.5

    def test_active_branch_under_dt_action_is_neutral(self):
        xor_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="dt",
                branches={"a": _xor_branch("active", "a")},
            )
        }
        assert compute_xor_score(xor_screening, WEIGHTS) == 0.5

    def test_active_branch_under_fallback_action_is_penalized(self):
        xor_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="fallback",
                branches={"a": _xor_branch("active", "a")},
            )
        }
        expected = 0.5 - 0.5 * WEIGHTS.w_fb_xor
        assert compute_xor_score(xor_screening, WEIGHTS) == pytest.approx(expected)

    def test_branch_level_fallback_status_penalized_regardless_of_place_action(self):
        # Branch-level "fallback" status (insufficient prob/samples for DT
        # eligibility) is penalized the same way whether the place-level
        # action ended up "dt" or "fallback" — same weight either way
        # (docs/network_improvement_loop.md §3.2: level 2/3 fallback weigh
        # the same).
        under_dt = {
            "p1": XorSplitScreening(
                total_samples=100, action="dt",
                branches={"a": _xor_branch("fallback", "a")},
            )
        }
        under_fallback = {
            "p1": XorSplitScreening(
                total_samples=100, action="fallback",
                branches={"a": _xor_branch("fallback", "a")},
            )
        }
        assert compute_xor_score(under_dt, WEIGHTS) == compute_xor_score(under_fallback, WEIGHTS)

    def test_unexpected_status_raises(self):
        xor_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="dt",
                branches={"a": _xor_branch("bogus_status", "a")},
            )
        }
        with pytest.raises(ValueError):
            compute_xor_score(xor_screening, WEIGHTS)

    def test_multiple_branches_average_correctly(self):
        # One rewarded (certain), one penalized (pruned) -> average of the
        # two bounded [0,1] values.
        xor_screening = {
            "p1": XorSplitScreening(
                total_samples=100, action="deterministic",
                branches={
                    "a": _xor_branch("certain", "a"),
                    "b": _xor_branch("pruned", "b"),
                },
            )
        }
        certain_value = 0.5 + 0.5 * WEIGHTS.w_det_xor
        pruned_value = 0.5 - 0.5 * WEIGHTS.w_prune_xor
        expected = (certain_value + pruned_value) / 2
        assert compute_xor_score(xor_screening, WEIGHTS) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_effect_score — Asse B (effetti)
# ---------------------------------------------------------------------------

class TestComputeEffectScore:
    def test_never_action_excluded_from_score(self):
        effect_screening = {
            "act": TransitionScreening(
                transition_name="act", total_firings=10,
                attributes={"attr": _effect_attr("never", "never")},
            )
        }
        assert compute_effect_score(effect_screening, WEIGHTS) == 0.5

    def test_deterministic_appearance_and_value_both_rewarded(self):
        effect_screening = {
            "act": TransitionScreening(
                transition_name="act", total_firings=10,
                attributes={"attr": _effect_attr("deterministic", "deterministic")},
            )
        }
        expected = 0.5 + 0.5 * WEIGHTS.w_det_eff
        assert compute_effect_score(effect_screening, WEIGHTS) == pytest.approx(expected)

    def test_dt_action_is_neutral(self):
        effect_screening = {
            "act": TransitionScreening(
                transition_name="act", total_firings=10,
                attributes={"attr": _effect_attr("dt", "dt")},
            )
        }
        assert compute_effect_score(effect_screening, WEIGHTS) == 0.5

    def test_fallback_action_penalized(self):
        effect_screening = {
            "act": TransitionScreening(
                transition_name="act", total_firings=10,
                attributes={"attr": _effect_attr("fallback", "fallback")},
            )
        }
        expected = 0.5 - 0.5 * WEIGHTS.w_fb_eff
        assert compute_effect_score(effect_screening, WEIGHTS) == pytest.approx(expected)

    def test_mixed_appearance_and_value_averaged_over_two_decisions(self):
        # appearance=deterministic (reward), value=fallback (penalty) -> mean of the two.
        effect_screening = {
            "act": TransitionScreening(
                transition_name="act", total_firings=10,
                attributes={"attr": _effect_attr("deterministic", "fallback")},
            )
        }
        deterministic_value = 0.5 + 0.5 * WEIGHTS.w_det_eff
        fallback_value = 0.5 - 0.5 * WEIGHTS.w_fb_eff
        expected = (deterministic_value + fallback_value) / 2
        assert compute_effect_score(effect_screening, WEIGHTS) == pytest.approx(expected)

    def test_unexpected_action_raises(self):
        effect_screening = {
            "act": TransitionScreening(
                transition_name="act", total_firings=10,
                attributes={"attr": _effect_attr("bogus", "dt")},
            )
        }
        with pytest.raises(ValueError):
            compute_effect_score(effect_screening, WEIGHTS)


# ---------------------------------------------------------------------------
# compute_duplication_score
# ---------------------------------------------------------------------------

class TestComputeDuplicationScore:
    def test_empty_transitions_returns_zero(self):
        assert compute_duplication_score({}) == 0.0

    def test_single_effect_group_is_max_reward(self):
        # excess = 0 -> max(1, 0) = 1 -> len(transitions) / 1
        transitions = {"act": _prepared_transition(1)}
        assert compute_duplication_score(transitions) == pytest.approx(1.0)

    def test_no_effect_groups_is_max_reward(self):
        # excess = max(0, 0 - 1) = 0 -> same as the single-group case
        transitions = {"act": _prepared_transition(0)}
        assert compute_duplication_score(transitions) == pytest.approx(1.0)

    def test_multiple_effect_groups_reduce_the_reward(self):
        # total_groups = 3, reward = len(transitions) / total_groups = 1 / 3
        transitions = {"act": _prepared_transition(3)}
        assert compute_duplication_score(transitions) == pytest.approx(1 / 3)

    def test_reward_grows_with_transition_count_at_fixed_total_groups(self):
        # total_groups = 3 + 1 = 4, reward = len(transitions) / total_groups = 2 / 4
        transitions = {"a": _prepared_transition(3), "b": _prepared_transition(1)}
        assert compute_duplication_score(transitions) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_reproducibility
# ---------------------------------------------------------------------------

def _outcome(is_replayable: bool) -> ReplayOutcome:
    return ReplayOutcome(
        is_replayable=is_replayable, error_step=None, error_reason=None,
        steps=[], final_marking=set(), final_attributes={}, warnings=[],
    )


class TestComputeReproducibility:
    def test_empty_list_returns_zero(self):
        assert compute_reproducibility([]) == 0.0

    def test_all_replayable_returns_one(self):
        assert compute_reproducibility([_outcome(True), _outcome(True)]) == 1.0

    def test_none_replayable_returns_zero(self):
        assert compute_reproducibility([_outcome(False), _outcome(False)]) == 0.0

    def test_mixed_returns_fraction(self):
        outcomes = [_outcome(True), _outcome(True), _outcome(False), _outcome(False)]
        assert compute_reproducibility(outcomes) == 0.5


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------

class _FakeParseResult:
    """Duck-typed stand-in for ParseResult — only .transitions is read."""
    def __init__(self, transition_names):
        self.transitions = {name: None for name in transition_names}


class TestComputeCoverage:
    def test_empty_original_activities_returns_one(self):
        assert compute_coverage(_FakeParseResult([]), set()) == 1.0

    def test_all_activities_present_returns_one(self):
        pr = _FakeParseResult(["a", "b"])
        assert compute_coverage(pr, {"a", "b"}) == 1.0

    def test_partial_coverage_returns_fraction(self):
        pr = _FakeParseResult(["a"])
        assert compute_coverage(pr, {"a", "b"}) == 0.5

    def test_no_activities_present_returns_zero(self):
        pr = _FakeParseResult([])
        assert compute_coverage(pr, {"a", "b"}) == 0.0
