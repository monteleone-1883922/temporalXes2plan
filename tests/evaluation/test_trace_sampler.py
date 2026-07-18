"""Unit tests for evaluation.trace_sampler.

sample_prefix() delegates all replay logic to a replay.trace_replayer.TraceReplayer
(replay_evaluation_split()) — these tests mock that boundary with a real
EvaluationReplayOutcome instance (rather than a loose MagicMock) so a typo'd
field name fails loudly. Replay internals themselves (Fase 1/Fase 2, guard
resolution, etc.) belong to tests/replay/, not here — this file only checks
what trace_sampler itself does with the outcome it gets back.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pm4py
import pytest

from evaluation.trace_sampler import (
    PrefixSample,
    _duration_seconds,
    _event_attributes,
    sample_prefix,
)
from replay.trace_replayer import EvaluationReplayOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(name: str, ts: Optional[datetime] = None, **extra) -> pm4py.objects.log.obj.Event:
    e = pm4py.objects.log.obj.Event()
    e["concept:name"] = name
    if ts is not None:
        e["time:timestamp"] = ts
    for k, v in extra.items():
        e[k] = v
    return e


def _make_trace(n_events: int, with_timestamps: bool = False, **last_attrs) -> pm4py.objects.log.obj.Trace:
    trace = pm4py.objects.log.obj.Trace()
    trace.attributes["concept:name"] = "case_1"
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n_events):
        ts = base + timedelta(hours=i) if with_timestamps else None
        extra = last_attrs if i == n_events - 1 else {}
        trace.append(_make_event(f"act_{i}", ts=ts, **extra))
    return trace


def _make_outcome(
    split_marking=None,
    split_attributes=None,
    expected_final_attributes=None,
    reached_end=True,
    matches_expected_final_state=True,
    warnings=None,
    max_modeled_duration_seconds=0.0,
) -> EvaluationReplayOutcome:
    return EvaluationReplayOutcome(
        reached_end=reached_end,
        matches_expected_final_state=matches_expected_final_state,
        error_step=None,
        error_reason=None,
        steps=[],
        split_marking=split_marking if split_marking is not None else {"p_1"},
        split_attributes=split_attributes or {},
        final_marking=set(),
        final_attributes={},
        expected_final_attributes=expected_final_attributes or {},
        accumulated_duration_seconds=None,
        within_deadline=None,
        max_modeled_duration_seconds=max_modeled_duration_seconds,
        warnings=warnings or [],
    )


def _make_replayer(outcome: Optional[EvaluationReplayOutcome] = None, raises: Optional[Exception] = None):
    replayer = MagicMock()
    if raises is not None:
        replayer.replay_evaluation_split.side_effect = raises
    else:
        replayer.replay_evaluation_split.return_value = outcome or _make_outcome()
    return replayer


# ---------------------------------------------------------------------------
# sample_prefix — prefix ratio bounds
# ---------------------------------------------------------------------------

class TestPrefixRatioBounds:
    def test_prefix_ratio_within_bounds(self):
        trace = _make_trace(20)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, min_prefix_pct=0.2, max_prefix_pct=0.8, seed=0)
        assert result is not None
        assert 0.2 <= result.prefix_ratio <= 0.8

    def test_prefix_ratio_multiple_seeds(self):
        trace = _make_trace(50)
        replayer = _make_replayer()
        for seed in range(10):
            result = sample_prefix(trace, replayer, min_prefix_pct=0.3, max_prefix_pct=0.7, seed=seed)
            assert result is not None
            assert 0.3 <= result.prefix_ratio <= 0.7

    def test_prefix_ratio_computed_from_actual_event_count(self):
        trace = _make_trace(10)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, min_prefix_pct=0.4, max_prefix_pct=0.6, seed=7)
        assert result is not None
        expected = len(result.prefix_events) / 10
        assert abs(result.prefix_ratio - expected) < 1e-9

    def test_replay_evaluation_split_called_with_n_prefix(self):
        trace = _make_trace(10)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, min_prefix_pct=0.4, max_prefix_pct=0.6, seed=7)
        assert result is not None
        replayer.replay_evaluation_split.assert_called_once_with(trace, len(result.prefix_events))


# ---------------------------------------------------------------------------
# sample_prefix — prefix + suffix = full trace
# ---------------------------------------------------------------------------

class TestPrefixSuffix:
    def test_suffix_is_complement(self):
        trace = _make_trace(20)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        combined = result.prefix_events + result.suffix_events
        assert len(combined) == 20

    def test_prefix_plus_suffix_equals_full_trace_events(self):
        trace = _make_trace(15)
        events = list(trace)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, seed=42)
        assert result is not None
        assert result.prefix_events + result.suffix_events == events

    def test_prefix_not_empty(self):
        trace = _make_trace(10)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert len(result.prefix_events) >= 1


# ---------------------------------------------------------------------------
# sample_prefix — init_places / init_effects derived from the outcome
# ---------------------------------------------------------------------------

class TestInitStateFromOutcome:
    def test_init_places_is_sorted_split_marking(self):
        trace = _make_trace(10)
        replayer = _make_replayer(_make_outcome(split_marking={"p_3", "p_1", "p_2"}))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.init_places == ["p_1", "p_2", "p_3"]

    def test_init_effects_converted_from_split_attributes(self):
        trace = _make_trace(10)
        replayer = _make_replayer(_make_outcome(split_attributes={"status": "open", "amount": "high"}))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert {"attribute": "status", "value": "open"} in result.init_effects
        assert {"attribute": "amount", "value": "high"} in result.init_effects
        assert len(result.init_effects) == 2

    def test_init_effects_empty_when_no_split_attributes(self):
        trace = _make_trace(10)
        replayer = _make_replayer(_make_outcome(split_attributes={}))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.init_effects == []


# ---------------------------------------------------------------------------
# sample_prefix — is_replayable combines reached_end AND matches_expected_final_state
# ---------------------------------------------------------------------------

class TestIsReplayable:
    @pytest.mark.parametrize(
        "reached_end,matches,expected",
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_is_replayable_is_conjunction(self, reached_end, matches, expected):
        trace = _make_trace(10)
        replayer = _make_replayer(_make_outcome(
            reached_end=reached_end, matches_expected_final_state=matches,
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.is_replayable is expected

    def test_warnings_passed_through(self):
        trace = _make_trace(10)
        replayer = _make_replayer(_make_outcome(warnings=["some warning"]))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.warnings == ["some warning"]


# ---------------------------------------------------------------------------
# sample_prefix — reached_within_time: reached_end AND
# max_modeled_duration_s fits within the trace's real remaining duration
# (full_duration_s - prefix_duration_s).
# ---------------------------------------------------------------------------

class TestReachedWithinTime:
    def _actual_budget(self, trace, seed) -> float:
        """Discover the real budget sample_prefix computes for this
        trace/seed (same trick as TestPrefixRatioBounds -- derive the
        expectation from a real call instead of hand-deriving the seeded
        RNG's prefix count)."""
        replayer = _make_replayer(_make_outcome(max_modeled_duration_seconds=0.0))
        result = sample_prefix(trace, replayer, seed=seed)
        assert result is not None
        assert result.full_duration_s is not None and result.prefix_duration_s is not None
        return result.full_duration_s - result.prefix_duration_s

    def test_true_when_duration_well_within_budget(self):
        trace = _make_trace(10, with_timestamps=True)
        replayer = _make_replayer(_make_outcome(
            reached_end=True, max_modeled_duration_seconds=0.0,
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.reached_within_time is True

    def test_true_at_exact_boundary(self):
        trace = _make_trace(10, with_timestamps=True)
        budget = self._actual_budget(trace, seed=0)
        replayer = _make_replayer(_make_outcome(
            reached_end=True, max_modeled_duration_seconds=budget,
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.reached_within_time is True

    def test_false_when_duration_exceeds_budget(self):
        trace = _make_trace(10, with_timestamps=True)
        budget = self._actual_budget(trace, seed=0)
        replayer = _make_replayer(_make_outcome(
            reached_end=True, max_modeled_duration_seconds=budget + 1.0,
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.reached_within_time is False

    def test_false_when_not_reached_end_even_if_duration_fits(self):
        trace = _make_trace(10, with_timestamps=True)
        replayer = _make_replayer(_make_outcome(
            reached_end=False, max_modeled_duration_seconds=0.0,
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.reached_within_time is False

    def test_false_when_no_timestamps(self):
        trace = _make_trace(10, with_timestamps=False)
        replayer = _make_replayer(_make_outcome(
            reached_end=True, max_modeled_duration_seconds=0.0,
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.reached_within_time is False

    def test_max_modeled_duration_s_passthrough_from_outcome(self):
        trace = _make_trace(10, with_timestamps=True)
        replayer = _make_replayer(_make_outcome(max_modeled_duration_seconds=42.5))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.max_modeled_duration_s == 42.5


# ---------------------------------------------------------------------------
# sample_prefix — final event attributes are a straight passthrough of
# outcome.expected_final_attributes (no filtering/derivation happens in
# trace_sampler itself — that logic lives in TraceReplayer now).
# ---------------------------------------------------------------------------

class TestFinalEventAttributes:
    def test_final_event_attributes_passthrough_from_outcome(self):
        trace = _make_trace(10)
        replayer = _make_replayer(_make_outcome(
            expected_final_attributes={"status": "discharged", "score": 42},
        ))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.final_event_attributes == {"status": "discharged", "score": 42}

    def test_final_event_attributes_empty_by_default(self):
        trace = _make_trace(5)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.final_event_attributes == {}


# ---------------------------------------------------------------------------
# sample_prefix — None cases
# ---------------------------------------------------------------------------

class TestReturnsNone:
    def test_returns_none_if_trace_too_short(self):
        trace = _make_trace(1)  # only 1 event
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, min_prefix_pct=0.5, max_prefix_pct=0.9, min_prefix_events=2)
        assert result is None

    def test_returns_none_if_replay_raises(self):
        trace = _make_trace(10)
        replayer = _make_replayer(raises=Exception("replay error"))
        result = sample_prefix(trace, replayer, seed=0)
        assert result is None


# ---------------------------------------------------------------------------
# _duration_seconds
# ---------------------------------------------------------------------------

class TestDurationSeconds:
    def test_duration_none_when_no_timestamps(self):
        events = [_make_event(f"act_{i}") for i in range(5)]
        assert _duration_seconds(events) is None

    def test_duration_none_when_single_event(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        events = [_make_event("act_0", ts=ts)]
        assert _duration_seconds(events) is None

    def test_duration_computed_in_seconds(self):
        base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event("a", ts=base),
            _make_event("b", ts=base + timedelta(hours=2)),
        ]
        assert _duration_seconds(events) == pytest.approx(7200.0)

    def test_duration_across_multiple_events(self):
        base = datetime(2024, 3, 1, tzinfo=timezone.utc)
        events = [
            _make_event("a", ts=base),
            _make_event("b", ts=base + timedelta(minutes=30)),
            _make_event("c", ts=base + timedelta(hours=1)),
        ]
        assert _duration_seconds(events) == pytest.approx(3600.0)

    def test_duration_none_when_only_one_has_timestamp(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        events = [_make_event("a", ts=base), _make_event("b")]
        assert _duration_seconds(events) is None

    def test_duration_in_prefix_sample(self):
        trace = _make_trace(10, with_timestamps=True)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, min_prefix_pct=0.9, max_prefix_pct=1.0, seed=0)
        assert result is not None
        assert result.full_duration_s is not None
        assert result.full_duration_s > 0

    def test_duration_none_in_prefix_sample_without_timestamps(self):
        trace = _make_trace(10, with_timestamps=False)
        replayer = _make_replayer()
        result = sample_prefix(trace, replayer, seed=0)
        assert result is not None
        assert result.prefix_duration_s is None
        assert result.full_duration_s is None


# ---------------------------------------------------------------------------
# _event_attributes
# ---------------------------------------------------------------------------

class TestEventAttributes:
    def test_excludes_metadata_keys(self):
        e = _make_event("act", ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                        status="done", score=5)
        attrs = _event_attributes(e)
        assert "concept:name" not in attrs
        assert "time:timestamp" not in attrs

    def test_includes_domain_attributes(self):
        e = _make_event("act", status="done", score=5)
        attrs = _event_attributes(e)
        assert attrs["status"] == "done"
        assert attrs["score"] == 5

    def test_empty_when_only_metadata(self):
        e = _make_event("act", ts=datetime(2024, 1, 1, tzinfo=timezone.utc))
        attrs = _event_attributes(e)
        assert attrs == {}
