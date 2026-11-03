"""Tests for parsing.attribute_filter.find_noise_attributes (filter 1 of the
attribute filters) and for LogProcessor's handling of the ignored list.

The filter is pure, so the logs here are plain lists of dicts: a trace is a
list of events, an event a dict of attributes.
"""
from collections import Counter
from typing import Any, Dict, List

import pytest

from models import AnalysisConfig
from parsing.attribute_filter import find_noise_attributes
from parsing.log_processor import LogProcessor


def _counts(log: List[List[Dict[str, Any]]]) -> Dict[str, Dict[Any, int]]:
    """Per-attribute {value: n events}, as LogProcessor.attribute_value_counts."""
    counts: Dict[str, Counter] = {}
    for trace in log:
        for event in trace:
            for attr, value in event.items():
                counts.setdefault(attr, Counter())[value] += 1
    return {a: dict(c) for a, c in counts.items()}


def _run(log, categories, **config_kwargs):
    config = AnalysisConfig(**config_kwargs)
    return find_noise_attributes(log, categories, _counts(log), config)


def _traces(n_traces: int, n_events: int, make_event) -> List[List[Dict[str, Any]]]:
    return [[make_event(t, e) for e in range(n_events)] for t in range(n_traces)]


class TestConstantRule:
    def test_single_value_attribute_is_noise(self):
        log = _traces(5, 4, lambda t, e: {"flag": "always"})
        result = _run(log, {"flag": "categorical"})
        assert list(result) == ["flag"]
        assert result["flag"].startswith("N1 constant")
        assert "'always'" in result["flag"]

    def test_constant_is_flagged_even_when_few_events_carry_it(self):
        log = _traces(5, 4, lambda t, e: {"rare": 0.0} if (t, e) == (0, 0) else {"other": e})
        assert "rare" in _run(log, {"rare": "numerical", "other": "numerical"})

    def test_attribute_with_two_values_is_not_constant(self):
        log = _traces(5, 4, lambda t, e: {"flag": "a" if t % 2 else "b"})
        assert _run(log, {"flag": "categorical"}) == {}


class TestLongTextRule:
    @staticmethod
    def _ids(n_distinct: int):
        # event ids that repeat every n_distinct events, so N3 cannot apply
        # (the value never equals the previous one but 'carry' is irrelevant here:
        # we only look at N2, with a high change rate of 1.0 and lots of values)
        return _traces(60, 1, lambda t, e: {"code": f"id{t % n_distinct}"})

    def test_text_with_more_values_than_the_limit_is_noise(self):
        result = _run(self._ids(60), {"code": "categorical"})
        assert result["code"].startswith("N2 long text")
        assert "60 distinct" in result["code"]

    def test_text_at_the_limit_is_kept(self):
        # one event per trace: no pairs, so N3 cannot fire either
        assert _run(self._ids(50), {"code": "categorical"}) == {}

    def test_the_limit_is_a_config_parameter(self):
        result = _run(self._ids(10), {"code": "categorical"}, noise_max_distinct_text=5)
        assert "code" in result

    def test_numerical_and_boolean_attributes_are_not_judged_by_n2(self):
        log = _traces(60, 1, lambda t, e: {"measure": float(t), "flag": bool(t % 2)})
        # 60 distinct numeric values: discretised later, so not "long text"
        assert _run(log, {"measure": "numerical", "flag": "boolean"}) == {}


class TestCounterRule:
    @staticmethod
    def _counter_log(change_every: int = 1, carry_all: bool = True):
        def make(t, e):
            event = {"step": f"s{e // change_every}"}
            if not carry_all and e % 2:
                event.pop("step")
                event["x"] = 1
            return event
        return _traces(6, 12, make)

    def test_value_that_changes_at_every_event_is_a_counter(self):
        result = _run(self._counter_log(), {"step": "categorical"})
        assert result["step"].startswith("N3 counter-like")
        assert "100%" in result["step"]

    def test_value_that_changes_rarely_is_not_a_counter(self):
        # changes every 3rd event: change share 1/3 < 0.9
        assert _run(self._counter_log(change_every=3), {"step": "categorical"}) == {}

    def test_attribute_carried_by_few_events_is_not_a_counter(self):
        log = self._counter_log(carry_all=False)
        assert "step" not in _run(log, {"step": "categorical", "x": "numerical"})

    def test_too_few_distinct_values_is_not_a_counter(self):
        # alternates between two values forever: changes at every event but only 2 values
        log = _traces(6, 12, lambda t, e: {"toggle": e % 2})
        assert _run(log, {"toggle": "numerical"}) == {}

    def test_thresholds_are_config_parameters(self):
        # 4 distinct values, changing in 3 of 11 pairs (27%): not a counter by default...
        log = self._counter_log(change_every=3)
        assert _run(log, {"step": "categorical"}) == {}
        # ...but it is once both thresholds are relaxed
        relaxed = _run(log, {"step": "categorical"}, noise_counter_min_change=0.25, noise_counter_min_distinct=3)
        assert "step" in relaxed


class TestProtections:
    def test_keep_attributes_are_never_reported(self):
        log = _traces(5, 4, lambda t, e: {"flag": "always"})
        assert _run(log, {"flag": "categorical"}, keep_attributes=("flag",)) == {}

    def test_ignored_attributes_are_not_reported_again(self):
        log = _traces(5, 4, lambda t, e: {"flag": "always"})
        config_ignored = {"flag"} | set(AnalysisConfig().ignored_attributes)
        assert _run(log, {"flag": "categorical"}, ignored_attributes=config_ignored) == {}

    def test_trace_level_attributes_are_skipped(self):
        log = _traces(5, 4, lambda t, e: {"case:id": "same"})
        assert _run(log, {"case:id": "categorical"}) == {}

    def test_attributes_without_counts_are_skipped(self):
        log = _traces(3, 3, lambda t, e: {"a": e})
        assert find_noise_attributes(log, {"ghost": "categorical"}, {}, AnalysisConfig()) == {}


class TestOutput:
    def test_result_is_sorted_by_attribute_name(self):
        log = _traces(5, 4, lambda t, e: {"zeta": "z", "alpha": "a"})
        assert list(_run(log, {"zeta": "categorical", "alpha": "categorical"})) == ["alpha", "zeta"]

    def test_empty_log_gives_no_noise(self):
        assert find_noise_attributes([], {}, {}, AnalysisConfig()) == {}


class TestLogProcessorIgnoredList:
    """LogProcessor must agree with AnalysisConfig on what is ignored, and keep
    the per-value counts the noise filter reuses."""

    @staticmethod
    def _log():
        import pm4py  # noqa: F401  (get_event_attributes needs it)
        from pm4py.objects.log.obj import Event, EventLog, Trace
        log = EventLog()
        for i in range(4):
            trace = Trace()
            trace.attributes["concept:name"] = f"c{i}"
            for name in ("A", "B"):
                event = Event()
                event["concept:name"] = name
                event["keepme"] = "x" if i % 2 else "y"
                event["junk"] = f"j{i}"
                trace.append(event)
            log.append(trace)
        return log

    def test_default_ignored_list_is_unchanged(self):
        attributes, categories = LogProcessor("unused.xes").initialize_attributes(self._log())
        assert {"keepme", "junk"} <= attributes
        assert "concept:name" not in attributes

    def test_explicit_ignored_list_removes_the_attribute_everywhere(self):
        processor = LogProcessor("unused.xes", ignored_attributes={"junk", "concept:name"})
        attributes, categories = processor.initialize_attributes(self._log())
        assert "junk" not in attributes and "junk" not in categories
        assert "junk" not in processor.attribute_value_counts
        assert "keepme" in categories

    def test_value_counts_are_kept_for_the_noise_filter(self):
        processor = LogProcessor("unused.xes")
        processor.initialize_attributes(self._log())
        assert processor.attribute_value_counts["keepme"] == {"x": 4, "y": 4}
