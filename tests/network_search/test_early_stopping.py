"""Unit tests for the exact early-stopping bound inside rung 2
(docs/network_improvement_loop.md §5.4.2, docs/network_improvement_loop_plan.md
§9 phase 6) — network_search.runner._evaluate_rung2.

No real log/Parser/DomainBuilder involved: TraceReplayer.replay_full_trace
is monkeypatched to return a canned sequence of outcomes and count calls,
so these tests run fast and target only the stopping logic itself.
"""
import types

import optuna
import pytest

from network_search import runner as runner_module
from network_search.runner import _Rung1Result, _evaluate_rung2
from network_search.scoring import ScoreWeights, TrialMetrics, combine, max_reachable_score
from replay.trace_replayer import ReplayOutcome


def _outcome(is_replayable: bool) -> ReplayOutcome:
    return ReplayOutcome(
        is_replayable=is_replayable,
        error_step=None if is_replayable else 0,
        error_reason=None,
        steps=[],
        final_marking=set(),
        final_attributes={},
        warnings=[],
    )


class _FakeReplayer:
    """Stands in for TraceReplayer: replays a fixed outcome pattern in order
    and counts how many traces were actually asked for — the thing early
    stopping is supposed to cut down."""

    def __init__(self, outcomes):
        self._outcomes = iter(_outcome(o) for o in outcomes)
        self.calls = 0

    def replay_full_trace(self, trace):
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
    # xor_score=effect_score=duplication_score=0.0 -- combine() reduces to
    # reproducibility_score itself (implicit coefficient 1, no weight of its
    # own), easy to reason about by hand.
    return ScoreWeights()


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
        xor_score = rng.uniform(-1.0, 1.0)
        effect_score = rng.uniform(-1.0, 1.0)
        duplication_score = rng.uniform(0.0, 3.0)
        best_score_so_far = rng.uniform(-5.0, 10.0)

        n_success = 0
        for k, success in enumerate(outcomes, start=1):
            n_success += success
            bound = max_reachable_score(
                n_success, k, n, xor_score, effect_score, duplication_score, weights,
            )
            # The bound at any point must be >= the score this trial would
            # get if every remaining trace also replayed successfully.
            best_case_final_repro = (n_success + (n - k)) / n
            best_case_final_score = combine(
                TrialMetrics(xor_score, effect_score, duplication_score, best_case_final_repro, n, 0.0),
                weights,
            )
            assert bound >= best_case_final_score - 1e-9

            if bound <= best_score_so_far:
                # Correctness property: stopping here must not discard a
                # trial that, replayed to the end, would have beaten
                # best_score_so_far -- i.e. every possible completion from
                # here scores <= best_score_so_far too.
                actual_final_repro = (n_success + (n - k)) / n
                actual_final_score = combine(
                    TrialMetrics(xor_score, effect_score, duplication_score, actual_final_repro, n, 0.0),
                    weights,
                )
                assert actual_final_score <= best_score_so_far + 1e-9
