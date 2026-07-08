"""Unit tests for network_search.scoring — combine() and the exact
early-stopping bound max_reachable_score() (docs/network_improvement_loop.md
§5.4.2, docs/network_improvement_loop_plan.md §9 phase 2)."""
import random

import pytest

from network_search.scoring import ScoreWeights, TrialMetrics, combine, max_reachable_score

WEIGHTS = ScoreWeights(w_det=1.0, w_fb_xor=1.0, w_fb_eff=1.0, w_prune_xor=3.0, w_dup=0.2, w_repro=5.0)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

class TestCombine:
    def test_matches_manual_formula(self):
        metrics = TrialMetrics(
            fallback_score=0.4, duplication_penalty=2.0,
            reproducibility_score=0.8, n_test_replayed=10, coverage=1.0,
        )
        expected = WEIGHTS.w_repro * 0.8 + 0.4 - WEIGHTS.w_dup * 2.0
        assert combine(metrics, WEIGHTS) == pytest.approx(expected)

    def test_none_reproducibility_treated_as_zero(self):
        metrics = TrialMetrics(
            fallback_score=0.4, duplication_penalty=0.0,
            reproducibility_score=None, n_test_replayed=0, coverage=1.0,
        )
        assert combine(metrics, WEIGHTS) == pytest.approx(0.4)

    def test_monotone_increasing_in_reproducibility(self):
        base = TrialMetrics(
            fallback_score=0.0, duplication_penalty=0.0,
            reproducibility_score=0.3, n_test_replayed=3, coverage=1.0,
        )
        better = TrialMetrics(
            fallback_score=0.0, duplication_penalty=0.0,
            reproducibility_score=0.9, n_test_replayed=9, coverage=1.0,
        )
        assert combine(better, WEIGHTS) > combine(base, WEIGHTS)

    def test_higher_duplication_penalty_lowers_score(self):
        low_dup = TrialMetrics(0.0, 1.0, 0.5, 10, 1.0)
        high_dup = TrialMetrics(0.0, 5.0, 0.5, 10, 1.0)
        assert combine(high_dup, WEIGHTS) < combine(low_dup, WEIGHTS)


# ---------------------------------------------------------------------------
# max_reachable_score
# ---------------------------------------------------------------------------

class TestMaxReachableScore:
    def test_equals_combine_when_fully_processed(self):
        # n_processed == n_test_total: no remaining traces, the bound must
        # equal the actual achievable score for that exact outcome.
        n_test_total = 10
        n_success = 7
        bound = max_reachable_score(n_success, n_test_total, n_test_total, 0.1, 0.5, WEIGHTS)
        actual = combine(
            TrialMetrics(0.1, 0.5, n_success / n_test_total, n_test_total, 1.0), WEIGHTS,
        )
        assert bound == pytest.approx(actual)

    def test_decreases_as_failures_accumulate(self):
        n_test_total = 10
        # After 1 failure among 5 processed vs. after 3 failures among 5 processed:
        # more accumulated failures can only lower (or keep equal) the best-case bound.
        bound_1_failure = max_reachable_score(4, 5, n_test_total, 0.0, 0.0, WEIGHTS)
        bound_3_failures = max_reachable_score(2, 5, n_test_total, 0.0, 0.0, WEIGHTS)
        assert bound_3_failures < bound_1_failure

    def test_never_below_the_true_final_score_along_a_real_continuation(self):
        """The bound must never reject a trial that would go on to win —
        i.e. at every prefix of a real replay, the bound computed so far
        must be >= the actual final score eventually reached."""
        rng = random.Random(0)
        n_test_total = 30
        outcomes = [rng.random() < 0.6 for _ in range(n_test_total)]
        fallback_score, duplication_penalty = 0.2, 1.0

        final_success = sum(outcomes)
        final_score = combine(
            TrialMetrics(fallback_score, duplication_penalty,
                         final_success / n_test_total, n_test_total, 1.0),
            WEIGHTS,
        )

        n_success = 0
        for i, replayable in enumerate(outcomes):
            n_success += int(replayable)
            bound = max_reachable_score(
                n_success, i + 1, n_test_total, fallback_score, duplication_penalty, WEIGHTS,
            )
            assert bound >= final_score - 1e-9

    def test_bound_tightens_to_exact_reproducibility_score_component(self):
        # With fallback_score/duplication_penalty held at 0, the bound is
        # exactly w_repro * best_possible_reproducibility.
        bound = max_reachable_score(3, 5, 10, 0.0, 0.0, WEIGHTS)
        best_possible_repro = (3 + (10 - 5)) / 10  # 3 successes + all 5 remaining succeeding
        assert bound == pytest.approx(WEIGHTS.w_repro * best_possible_repro)
