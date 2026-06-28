"""Tests for the unified log_validator module."""

from datetime import datetime, timezone

import pytest
from pm4py.objects.log.obj import Event, EventLog, Trace

from parsing.log_validator import (
    TIMESTAMP_MIN_COVERAGE,
    ValidationResult,
    validate_event_log,
    validate_partial_trace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


def _make_log(traces: list) -> EventLog:
    """Build a pm4py EventLog from a list of dicts.

    Each dict has:
        case_id : str | None  (stored as trace attribute "concept:name")
        events  : list of dicts with event attribute key→value pairs
    """
    log = EventLog()
    for t_data in traces:
        trace = Trace()
        cid = t_data.get("case_id", "case1")
        if cid is not None:
            trace.attributes["concept:name"] = cid
        for e_data in t_data.get("events", []):
            event = Event()
            for k, v in e_data.items():
                event[k] = v
            trace.append(event)
        log.append(trace)
    return log


def _valid_event(activity: str = "A") -> dict:
    return {"concept:name": activity, "time:timestamp": _TS, "lifecycle:transition": "complete"}


# ---------------------------------------------------------------------------
# validate_event_log — case:concept:name
# ---------------------------------------------------------------------------

def test_valid_log_no_issues():
    log = _make_log([{"case_id": "c1", "events": [_valid_event()]}])
    r = validate_event_log(log)
    assert r.errors == []
    assert r.warnings == []
    assert not r.missing_timestamp


def test_missing_case_id_all():
    log = _make_log([{"case_id": None, "events": [_valid_event()]}])
    r = validate_event_log(log)
    assert any("case:concept:name" in e for e in r.errors)


def test_missing_case_id_partial():
    log = _make_log([
        {"case_id": "c1", "events": [_valid_event()]},
        {"case_id": None, "events": [_valid_event()]},
    ])
    r = validate_event_log(log)
    assert any("case:concept:name" in e for e in r.errors)


def test_float_case_id_all():
    log = _make_log([{"case_id": 1.0, "events": [_valid_event()]}])
    r = validate_event_log(log)
    assert any("float" in e for e in r.errors)


def test_boolean_case_id_all():
    log = _make_log([{"case_id": True, "events": [_valid_event()]}])
    r = validate_event_log(log)
    assert any("boolean" in e for e in r.errors)


# ---------------------------------------------------------------------------
# validate_event_log — concept:name
# ---------------------------------------------------------------------------

def test_missing_concept_name_all():
    log = _make_log([{"case_id": "c1", "events": [{"time:timestamp": _TS}]}])
    r = validate_event_log(log)
    assert any("concept:name" in e for e in r.errors)


def test_missing_concept_name_partial():
    log = _make_log([{"case_id": "c1", "events": [
        {"concept:name": "A", "time:timestamp": _TS},
        {"concept:name": None, "time:timestamp": _TS},
    ]}])
    r = validate_event_log(log)
    assert any("concept:name" in e for e in r.errors)


def test_numeric_concept_name():
    log = _make_log([{"case_id": "c1", "events": [
        {"concept:name": 42, "time:timestamp": _TS},
    ]}])
    r = validate_event_log(log)
    assert any("non-string" in e for e in r.errors)


def test_boolean_concept_name():
    log = _make_log([{"case_id": "c1", "events": [
        {"concept:name": True, "time:timestamp": _TS},
    ]}])
    r = validate_event_log(log)
    assert any("non-string" in e for e in r.errors)


# ---------------------------------------------------------------------------
# validate_event_log — time:timestamp
# ---------------------------------------------------------------------------

def test_missing_timestamp_all():
    log = _make_log([{"case_id": "c1", "events": [
        {"concept:name": "A", "lifecycle:transition": "complete"},
    ]}])
    r = validate_event_log(log)
    assert r.missing_timestamp is True
    assert r.warnings != []


def test_missing_timestamp_partial_below_threshold():
    n_total = 10
    n_with_ts = int(n_total * TIMESTAMP_MIN_COVERAGE) - 1  # just below threshold
    events = [{"concept:name": "A", "time:timestamp": _TS}] * n_with_ts
    events += [{"concept:name": "A"}] * (n_total - n_with_ts)
    log = _make_log([{"case_id": "c1", "events": events}])
    r = validate_event_log(log)
    assert r.missing_timestamp is True


def test_timestamp_above_threshold_no_warning():
    n_total = 10
    n_with_ts = int(n_total * TIMESTAMP_MIN_COVERAGE) + 1
    events = [{"concept:name": "A", "time:timestamp": _TS}] * n_with_ts
    events += [{"concept:name": "A"}] * (n_total - n_with_ts)
    log = _make_log([{"case_id": "c1", "events": events}])
    r = validate_event_log(log)
    assert r.missing_timestamp is False


# ---------------------------------------------------------------------------
# validate_event_log — lifecycle:transition
# ---------------------------------------------------------------------------

def test_missing_lifecycle_is_info():
    log = _make_log([{"case_id": "c1", "events": [
        {"concept:name": "A", "time:timestamp": _TS},
    ]}])
    r = validate_event_log(log)
    assert r.missing_lifecycle is True
    assert r.infos != []
    assert r.errors == []
    assert r.warnings == []


def test_nonstandard_lifecycle_values():
    log = _make_log([{"case_id": "c1", "events": [
        {"concept:name": "A", "time:timestamp": _TS, "lifecycle:transition": "done"},
    ]}])
    r = validate_event_log(log)
    assert any("non-standard" in i or "done" in i for i in r.infos)
    assert r.errors == []


def test_standard_lifecycle_no_issues():
    for val in ("start", "complete", "assign"):
        log = _make_log([{"case_id": "c1", "events": [
            {"concept:name": "A", "time:timestamp": _TS, "lifecycle:transition": val},
        ]}])
        r = validate_event_log(log)
        assert r.missing_lifecycle is False
        assert r.errors == []


# ---------------------------------------------------------------------------
# validate_partial_trace
# ---------------------------------------------------------------------------

def test_validate_partial_trace_ok():
    trace = Trace()
    trace.attributes["concept:name"] = "case1"
    e = Event()
    e["concept:name"] = "Register"
    e["time:timestamp"] = _TS
    trace.append(e)
    assert validate_partial_trace(trace) == []


def test_validate_partial_trace_no_case_id():
    trace = Trace()  # no concept:name attribute
    e = Event()
    e["concept:name"] = "Register"
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert any("case identifier" in err for err in errors)


def test_validate_partial_trace_missing_activity():
    trace = Trace()
    trace.attributes["concept:name"] = "case1"
    e = Event()
    e["time:timestamp"] = _TS  # no concept:name
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert any("concept:name" in err or "activity" in err for err in errors)


def test_validate_partial_trace_empty_activity():
    trace = Trace()
    trace.attributes["concept:name"] = "case1"
    e = Event()
    e["concept:name"] = ""
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert errors != []


def test_validate_partial_trace_boolean_activity():
    trace = Trace()
    trace.attributes["concept:name"] = "case1"
    e = Event()
    e["concept:name"] = True
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert any("boolean" in err for err in errors)


def test_validate_partial_trace_numeric_activity():
    trace = Trace()
    trace.attributes["concept:name"] = "case1"
    e = Event()
    e["concept:name"] = 99
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert any("non-string" in err or "int" in err for err in errors)


def test_validate_partial_trace_float_case_id():
    trace = Trace()
    trace.attributes["concept:name"] = 1.0
    e = Event()
    e["concept:name"] = "Register"
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert any("float" in err for err in errors)


def test_validate_partial_trace_boolean_case_id():
    trace = Trace()
    trace.attributes["concept:name"] = True
    e = Event()
    e["concept:name"] = "Register"
    trace.append(e)
    errors = validate_partial_trace(trace)
    assert any("boolean" in err for err in errors)
