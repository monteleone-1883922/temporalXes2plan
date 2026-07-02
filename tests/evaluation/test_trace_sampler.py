"""Unit tests for evaluation.trace_sampler."""
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


def _make_replay_result(
    init_places=None,
    init_effects=None,
    replayed=None,
    warnings=None,
):
    r = MagicMock()
    r.init_places = init_places or ["p_1"]
    r.init_effects = init_effects or []
    r.replayed_activities = replayed or []
    r.warnings = warnings or []
    return r


def _make_api(replay_result=None, replay_raises=None):
    api = MagicMock()
    if replay_raises:
        api.replay_trace.side_effect = replay_raises
    else:
        api.replay_trace.return_value = replay_result or _make_replay_result()
    return api


SERIALIZED: Dict[str, Any] = {
    "graph": {"nodes": [], "edges": []},
    "transitions": {},
    "attribute_catalog": {},
    "metadata": {"start_place": "p_start", "end_place": "p_end"},
}


# ---------------------------------------------------------------------------
# sample_prefix — prefix ratio bounds
# ---------------------------------------------------------------------------

class TestPrefixRatioBounds:
    def test_prefix_ratio_within_bounds(self):
        trace = _make_trace(20)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api,
                               min_prefix_pct=0.2, max_prefix_pct=0.8, seed=0)
        assert result is not None
        assert 0.2 <= result.prefix_ratio <= 0.8

    def test_prefix_ratio_multiple_seeds(self):
        trace = _make_trace(50)
        api = _make_api()
        for seed in range(10):
            result = sample_prefix(trace, SERIALIZED, api,
                                   min_prefix_pct=0.3, max_prefix_pct=0.7, seed=seed)
            assert result is not None
            assert 0.3 <= result.prefix_ratio <= 0.7

    def test_prefix_ratio_computed_from_actual_event_count(self):
        trace = _make_trace(10)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api,
                               min_prefix_pct=0.4, max_prefix_pct=0.6, seed=7)
        assert result is not None
        expected = len(result.prefix_events) / 10
        assert abs(result.prefix_ratio - expected) < 1e-9


# ---------------------------------------------------------------------------
# sample_prefix — prefix + suffix = full trace
# ---------------------------------------------------------------------------

class TestPrefixSuffix:
    def test_suffix_is_complement(self):
        trace = _make_trace(20)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api, seed=0)
        assert result is not None
        combined = result.prefix_events + result.suffix_events
        assert len(combined) == 20

    def test_prefix_plus_suffix_equals_full_trace_events(self):
        trace = _make_trace(15)
        events = list(trace)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api, seed=42)
        assert result is not None
        assert result.prefix_events + result.suffix_events == events

    def test_prefix_not_empty(self):
        trace = _make_trace(10)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api, seed=0)
        assert result is not None
        assert len(result.prefix_events) >= 1


# ---------------------------------------------------------------------------
# sample_prefix — final event attributes
# ---------------------------------------------------------------------------

class TestFinalEventAttributes:
    def test_final_event_attributes_from_last_prefix_event(self):
        trace = _make_trace(10, status="discharged", score=42)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api,
                               min_prefix_pct=0.9, max_prefix_pct=1.0, seed=0)
        assert result is not None
        # Last event has the custom attrs
        assert result.final_event_attributes.get("status") == "discharged"
        assert result.final_event_attributes.get("score") == 42

    def test_final_event_attributes_excludes_concept_name(self):
        trace = _make_trace(5)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api, seed=0)
        assert result is not None
        assert "concept:name" not in result.final_event_attributes

    def test_final_event_attributes_excludes_timestamp(self):
        trace = _make_trace(5, with_timestamps=True)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api, seed=0)
        assert result is not None
        assert "time:timestamp" not in result.final_event_attributes


# ---------------------------------------------------------------------------
# sample_prefix — None cases
# ---------------------------------------------------------------------------

class TestReturnsNone:
    def test_returns_none_if_trace_too_short(self):
        trace = _make_trace(1)  # only 1 event
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api,
                               min_prefix_pct=0.5, max_prefix_pct=0.9,
                               min_prefix_events=2)
        assert result is None

    def test_returns_none_if_replay_raises(self):
        trace = _make_trace(10)
        api = _make_api(replay_raises=Exception("replay error"))
        result = sample_prefix(trace, SERIALIZED, api, seed=0)
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
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api,
                               min_prefix_pct=0.9, max_prefix_pct=1.0, seed=0)
        assert result is not None
        assert result.full_duration_s is not None
        assert result.full_duration_s > 0

    def test_duration_none_in_prefix_sample_without_timestamps(self):
        trace = _make_trace(10, with_timestamps=False)
        api = _make_api()
        result = sample_prefix(trace, SERIALIZED, api, seed=0)
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
