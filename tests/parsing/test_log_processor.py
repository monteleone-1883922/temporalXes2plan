"""LogProcessor.load_and_filter_log: lifecycle handling.

Regression test for a bug where the filter meant to keep only 'complete' events
used pm4py's default level="case" and so removed no event: start events stayed in
the log the Petri net is discovered from.
"""
import pm4py
import pytest
from pm4py.objects.log.obj import Event, EventLog, Trace

from parsing.log_processor import LogProcessor
from datetime import datetime, timedelta, timezone


def _event(name, lifecycle, second):
    e = Event()
    e["concept:name"] = name
    if lifecycle is not None:
        e["lifecycle:transition"] = lifecycle
    e["time:timestamp"] = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=second)
    return e


def _write(tmp_path, with_start: bool, lifecycle: bool = True) -> str:
    log = EventLog()
    for i in range(5):
        trace = Trace()
        trace.attributes["concept:name"] = f"c{i}"
        t = 0
        for name in ("A", "B", "C"):
            if with_start:
                trace.append(_event(name, "start", t))
                t += 1
            trace.append(_event(name, "complete" if lifecycle else None, t))
            t += 1
        log.append(trace)
    path = str(tmp_path / "log.xes")
    pm4py.write_xes(log, path)
    return path


def _lifecycles(log):
    return {e.get("lifecycle:transition") for trace in log for e in trace}


class TestLifecycleFilter:
    def test_start_events_are_removed_from_the_discovery_and_full_logs(self, tmp_path):
        log, full_log, _ = LogProcessor(_write(tmp_path, with_start=True)).load_and_filter_log(1.0)
        assert _lifecycles(log) == {"complete"}
        assert _lifecycles(full_log) == {"complete"}
        assert all(len(trace) == 3 for trace in log)

    def test_the_lifecycle_log_keeps_both_start_and_complete_for_durations(self, tmp_path):
        _, _, lifecycle_log = LogProcessor(_write(tmp_path, with_start=True)).load_and_filter_log(1.0)
        assert _lifecycles(lifecycle_log) == {"start", "complete"}
        assert all(len(trace) == 6 for trace in lifecycle_log)

    def test_a_log_with_only_complete_events_is_unchanged(self, tmp_path):
        log, full_log, _ = LogProcessor(_write(tmp_path, with_start=False)).load_and_filter_log(1.0)
        assert all(len(trace) == 3 for trace in full_log)
        assert len(full_log) == 5

    def test_a_log_without_a_lifecycle_attribute_is_left_alone(self, tmp_path):
        log, full_log, _ = LogProcessor(_write(tmp_path, with_start=False, lifecycle=False)).load_and_filter_log(1.0)
        assert len(full_log) == 5 and all(len(trace) == 3 for trace in full_log)

    def test_discovery_log_has_one_event_per_activity_so_the_net_has_no_spurious_loops(self, tmp_path):
        log, _, _ = LogProcessor(_write(tmp_path, with_start=True)).load_and_filter_log(1.0)
        net, _, _ = pm4py.discover_petri_net_inductive(log)
        assert sum(1 for t in net.transitions if t.label) == 3
        assert sum(1 for t in net.transitions if not t.label) == 0  # a pure sequence A -> B -> C
