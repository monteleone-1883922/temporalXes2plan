"""Unit tests for the exact early-stopping bound inside rung 2
(docs/network_improvement_loop.md §5.4.2, docs/network_improvement_loop_plan.md
§9 phase 6) — network_search.runner._evaluate_rung2.

No real log/Parser/DomainBuilder involved: TraceReplayer.replay_evaluation_split
is monkeypatched to return a canned sequence of outcomes and count calls,
so these tests run fast and target only the stopping logic itself.
"""
import types

import optuna
import pytest

from network_search import runner as runner_module
from network_search.runner import _Rung1Result, _evaluate_rung2
from network_search.scoring import ScoreWeights, TrialMetrics, combine, max_reachable_score
from replay.trace_replayer import EvaluationReplayOutcome


def _outcome(reached_end: bool, matches_expected_final_state: bool) -> EvaluationReplayOutcome:
    return EvaluationReplayOutcome(
        reached_end=reached_end,
        matches_expected_final_state=matches_expected_final_state,
        error_step=None if reached_end else 0,
        error_reason=None,
        steps=[],
        split_marking=set(),
        split_attributes={},
        final_marking=set(),
        final_attributes={},
        expected_final_attributes={},
        accumulated_duration_seconds=None,
        within_deadline=None,
        max_modeled_duration_seconds=0.0,
        warnings=[],
    )


class _FakeReplayer:
    """Stands in for TraceReplayer: replays a fixed outcome pattern in order
    and counts how many traces were actually asked for — the thing early
    stopping is supposed to cut down."""

    def __init__(self, outcomes, full_outcomes=None):
        # full_outcomes defaults to outcomes itself (full success == loose
        # success) when not given, matching every existing pre-full-replay
        # test scenario exactly.
        full_outcomes = full_outcomes if full_outcomes is not None else outcomes
        self._outcomes = iter(
            _outcome(o, f) for o, f in zip(outcomes, full_outcomes)
        )
        self.calls = 0

    def replay_evaluation_split(self, trace, n_prefix):
        assert n_prefix == 0
        self.calls += 1
        return next(self._outcomes)


def _rung1(xor_score: float = 0.0, effect_score: float = 0.0, duplication_score: float = 0.0) -> _Rung1Result:
    return _Rung1Result(
        trial=optuna.trial.FixedTrial({}),
        config=None,
        parse_result=types.SimpleNamespace(petri_net_model=None),
        prepared=None,
        xor_score=xor_score,
        effect_score=effect_score,
        duplication_score=duplication_score,
        coverage=1.0,
        partial_score=0.0,
    )


@pytest.fixture
def weights() -> ScoreWeights:
    # xor_score=effect_score=duplication_score=0.0, w_full_replay=0.0 --
    # combine() reduces to reproducibility_score itself (implicit
    # coefficient 1, no weight of its own), easy to reason about by hand.
    # w_full_replay is neutralized here so these hand-derived thresholds
    # (written before full_replayability_score existed) stay valid; see
    # test_full_replayability_affects_early_stopping_when_weighted below
    # for a test that specifically exercises a nonzero w_full_replay.
    return ScoreWeights(w_full_replay=0.0)


def test_stops_as_soon_as_the_bound_can_no_longer_beat_the_best_score(monkeypatch, weights):
    outcomes = [True, True, False, False, False, False, False, False, False, False]
    fake = _FakeReplayer(outcomes)
    monkeypatch.setattr(runner_module, "TraceReplayer", lambda *a, **kw: fake)

    rung1 = _rung1()
    test_traces = list(range(len(outcomes)))
    # Chosen so the bound drops to exactly best_score_so_far after the 4th
    # trace (2 successes + 2 failures): (2+6)/10 == 0.8 -- see the by-hand
    # derivation in the module's design doc §5.4.2.
    record = _evaluate_rung2(rung1, test_traces, weights, best_score_so_far=0.8)

    assert fake.calls == 4
    assert record.stopped_early is True
    assert record.metrics.n_test_replayed == 4


def test_replays_every_trace_when_the_bound_never_drops_low_enough(monkeypatch, weights):
    outcomes = [True] * 10
    fake = _FakeReplayer(outcomes)
    monkeypatch.setattr(runner_module, "TraceReplayer", lambda *a, **kw: fake)

    rung1 = _rung1()
    test_traces = list(range(len(outcomes)))
    record = _evaluate_rung2(rung1, test_traces, weights, best_score_so_far=float("-inf"))

    assert fake.calls == 10
    assert record.stopped_early is False
    assert record.metrics.n_test_replayed == 10
    assert record.metrics.reproducibility_score == 1.0
    assert record.metrics.full_replayability_score == 1.0


def test_full_replayability_affects_early_stopping_when_weighted(monkeypatch):
    """A trace that reaches the end but whose final attributes don't match
    the log (reached_end=True, matches_expected_final_state=False) still
    counts toward reproducibility_score but NOT toward
    full_replayability_score -- when w_full_replay > 0, this divergence
    must change the early-stopping outcome relative to the
    w_full_replay=0.0 case.

    Both scores are additive/non-negative, so including full_replayability_score
    can only ever RAISE the bound relative to reproducibility_score alone --
    it delays or prevents early stopping, never triggers it sooner, for a
    fixed threshold. This test picks best_score_so_far in the gap between
    the two weightings' respective ceilings to demonstrate that.
    """
    # Every trace reaches the end (loose success always true, so
    # reproducibility_score's bound is a constant 1.0 throughout), but only
    # the first 2 of 10 also match the expected final attribute state (so
    # full_replayability_score's own bound decays from 1.0 down to 0.2 as
    # the remaining-trace budget shrinks without any further matches).
    outcomes = [True] * 10
    full_outcomes = [True, True] + [False] * 8
    # threshold chosen strictly between reproducibility_score's constant
    # ceiling (1.0, with w_full_replay=0.0) and the combined bound's lowest
    # point (1.0 + 0.2 = 1.2, with w_full_replay=1.0) -- see the by-hand
    # trajectory: bound(k) = 1.0 + (2 + (10-k))/10, bottoming out at 1.2
    # when k=10.
    threshold = 1.1

    weights_no_full_replay = ScoreWeights(w_full_replay=0.0)
    fake_a = _FakeReplayer(outcomes, full_outcomes)
    monkeypatch.setattr(runner_module, "TraceReplayer", lambda *a, **kw: fake_a)
    record_a = _evaluate_rung2(_rung1(), list(range(10)), weights_no_full_replay, best_score_so_far=threshold)

    weights_with_full_replay = ScoreWeights(w_full_replay=1.0)
    fake_b = _FakeReplayer(outcomes, full_outcomes)
    monkeypatch.setattr(runner_module, "TraceReplayer", lambda *a, **kw: fake_b)
    record_b = _evaluate_rung2(_rung1(), list(range(10)), weights_with_full_replay, best_score_so_far=threshold)

    # Without w_full_replay, the bound is pinned at 1.0 <= threshold from
    # the very first trace, so it stops immediately. With w_full_replay=1.0,
    # the extra bonus term keeps the bound above threshold for the entire
    # replay, so it never stops early.
    assert record_a.stopped_early is True
    assert record_b.stopped_early is False
    assert fake_a.calls < fake_b.calls


def test_never_stops_early_a_trial_that_would_go_on_to_win(monkeypatch, weights):
    """The bound must be exact, not optimistic: replaying this trial to
    completion without early stopping must always score <= what
    max_reachable_score() promised at every intermediate step, for many
    random outcome patterns and best_score_so_far thresholds."""
    import random

    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 20)
        outcomes = [rng.random() < 0.5 for _ in range(n)]
        # full_success is a strict subset of outcomes (matches_expected_final_state
        # can only be true when reached_end is also true).
        full_outcomes = [o and rng.random() < 0.5 for o in outcomes]
        xor_score = rng.uniform(-1.0, 1.0)
        effect_score = rng.uniform(-1.0, 1.0)
        duplication_score = rng.uniform(0.0, 3.0)
        # w_full_replay must stay >= 0 for the bound's exactness guarantee
        # (see scoring.max_reachable_score's docstring).
        w_full_replay = rng.uniform(0.0, 3.0)
        weights_i = ScoreWeights(w_full_replay=w_full_replay)
        best_score_so_far = rng.uniform(-5.0, 10.0)

        n_success = 0
        n_full_success = 0
        for k, (success, full_success) in enumerate(zip(outcomes, full_outcomes), start=1):
            n_success += success
            n_full_success += full_success
            bound = max_reachable_score(
                n_success, n_full_success, k, n, xor_score, effect_score, duplication_score, weights_i,
            )
            # The bound at any point must be >= the score this trial would
            # get if every remaining trace also replayed successfully.
            best_case_final_repro = (n_success + (n - k)) / n
            best_case_final_full_replay = (n_full_success + (n - k)) / n
            best_case_final_score = combine(
                TrialMetrics(xor_score, effect_score, duplication_score, best_case_final_repro, n, 0.0,
                             full_replayability_score=best_case_final_full_replay),
                weights_i,
            )
            assert bound >= best_case_final_score - 1e-9

            if bound <= best_score_so_far:
                # Correctness property: stopping here must not discard a
                # trial that, replayed to the end, would have beaten
                # best_score_so_far -- i.e. every possible completion from
                # here scores <= best_score_so_far too.
                actual_final_repro = (n_success + (n - k)) / n
                actual_final_full_replay = (n_full_success + (n - k)) / n
                actual_final_score = combine(
                    TrialMetrics(xor_score, effect_score, duplication_score, actual_final_repro, n, 0.0,
                                 full_replayability_score=actual_final_full_replay),
                    weights_i,
                )
                assert actual_final_score <= best_score_so_far + 1e-9
