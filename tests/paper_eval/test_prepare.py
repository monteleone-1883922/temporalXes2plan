"""Tests for paper_eval.prepare (shared preprocessing of the paper evaluation)."""

import gzip
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import make_event, make_log, make_trace

from paper_eval import prepare as prep
from paper_eval.config import LogSpec


def _names(trace):
    return [e["concept:name"] for e in trace]


class TestPreprocessTrace:
    def test_events_sorted_by_timestamp(self):
        trace = make_trace(make_event("B", offset=20), make_event("A", offset=10))
        assert _names(prep.preprocess_trace(trace)) == ["A", "B"]

    def test_sort_is_stable_on_equal_timestamps(self):
        trace = make_trace(
            make_event("X", offset=5), make_event("Y", offset=5), make_event("Z", offset=5),
        )
        assert _names(prep.preprocess_trace(trace)) == ["X", "Y", "Z"]

    def test_naive_timestamps_become_utc(self):
        trace = make_trace(make_event("A", offset=0, tz_aware=False))
        out = prep.preprocess_trace(trace)
        assert out[0]["time:timestamp"].tzinfo == timezone.utc
        assert out[0]["time:timestamp"].hour == 8

    def test_aware_timestamps_converted_to_utc(self):
        cet = timezone(timedelta(hours=1))
        event = make_event("A")
        event["time:timestamp"] = datetime(2024, 1, 1, 9, 0, tzinfo=cet)
        out = prep.preprocess_trace(make_trace(event))
        assert out[0]["time:timestamp"] == datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        assert out[0]["time:timestamp"].utcoffset() == timedelta(0)

    def test_lifecycle_lowercased_and_only_start_complete_kept(self):
        trace = make_trace(
            make_event("A", lifecycle="SCHEDULE", offset=0),
            make_event("A", lifecycle="START", offset=1),
            make_event("A", lifecycle="suspend", offset=2),
            make_event("A", lifecycle="COMPLETE", offset=3),
        )
        out = prep.preprocess_trace(trace)
        assert [e["lifecycle:transition"] for e in out] == ["start", "complete"]

    def test_events_without_lifecycle_are_kept(self):
        out = prep.preprocess_trace(make_trace(make_event("A"), make_event("B", offset=1)))
        assert _names(out) == ["A", "B"]

    def test_input_trace_not_mutated(self):
        trace = make_trace(make_event("A", lifecycle="COMPLETE"))
        prep.preprocess_trace(trace)
        assert trace[0]["lifecycle:transition"] == "COMPLETE"

    def test_trace_attributes_preserved(self):
        trace = make_trace(make_event("A"), case_id="case_42")
        assert prep.preprocess_trace(trace).attributes["concept:name"] == "case_42"


class TestPreprocessLog:
    def test_short_traces_dropped_counting_only_complete_events(self):
        # 3 events but only 2 complete -> dropped with min_trace_len=3.
        short = make_trace(
            make_event("A", lifecycle="start", offset=0),
            make_event("A", lifecycle="complete", offset=1),
            make_event("B", lifecycle="complete", offset=2),
            case_id="short",
        )
        long = make_trace(
            make_event("A", lifecycle="complete", offset=0),
            make_event("B", lifecycle="complete", offset=1),
            make_event("C", lifecycle="complete", offset=2),
            case_id="long",
        )
        out = prep.preprocess(make_log(short, long), min_trace_len=3)
        assert [t.attributes["concept:name"] for t in out] == ["long"]

    def test_start_events_kept_in_output(self):
        trace = make_trace(
            make_event("A", lifecycle="start", offset=0),
            make_event("A", lifecycle="complete", offset=1),
            make_event("B", lifecycle="complete", offset=2),
            make_event("C", lifecycle="complete", offset=3),
        )
        out = prep.preprocess(make_log(trace), min_trace_len=3)
        assert [e["lifecycle:transition"] for e in out[0]] == ["start", "complete", "complete", "complete"]

    def test_traces_with_missing_timestamps_dropped(self):
        broken = make_trace(make_event("A"), make_event("B", offset=1), make_event("C", offset=2), case_id="broken")
        del broken[1]["time:timestamp"]
        ok = make_trace(make_event("A"), make_event("B", offset=1), make_event("C", offset=2), case_id="ok")
        out = prep.preprocess(make_log(broken, ok), min_trace_len=3)
        assert [t.attributes["concept:name"] for t in out] == ["ok"]


class TestLogStats:
    def test_counts_on_control_flow_view(self):
        trace = make_trace(
            make_event("A", lifecycle="start", offset=0),
            make_event("A", lifecycle="complete", offset=60),
            make_event("B", lifecycle="complete", offset=86400),
        )
        stats = prep.log_stats(prep.preprocess(make_log(trace), min_trace_len=2))
        assert stats["n_traces"] == 1
        assert stats["n_events"] == 3
        assert stats["n_control_flow_events"] == 2
        assert stats["n_activities"] == 2
        assert stats["case_duration_days"]["max"] == pytest.approx(1.0)


class TestStartOfTraceStats:
    """Per event attribute: in how many traces the first control-flow event sets it and it
    never changes (a descriptive statistic of the whole log, raw values)."""

    @staticmethod
    def _trace(first, later):
        return make_trace(make_event("A", offset=0, **first), make_event("B", offset=60, **later))

    @staticmethod
    def _by_attribute(log):
        return {r["attribute"]: r for r in prep.start_of_trace_stats(log)}

    def test_set_by_the_first_event_and_never_changed(self):
        log = make_log(*[self._trace({"color": "red"}, {"color": "red"}) for _ in range(4)])
        row = self._by_attribute(log)["color"]
        assert (row["n_traces_with_attribute"], row["n_set_by_first_event"], row["n_unchanged"]) == (4, 4, 4)
        assert row["share"] == 1.0

    def test_a_trace_where_a_later_event_changes_it_does_not_count_as_unchanged(self):
        log = make_log(
            self._trace({"color": "red"}, {"color": "blue"}),
            *[self._trace({"color": "red"}, {}) for _ in range(3)],
        )
        row = self._by_attribute(log)["color"]
        assert (row["n_set_by_first_event"], row["n_unchanged"]) == (4, 3)
        assert row["share"] == pytest.approx(0.75)

    def test_set_only_by_a_later_event(self):
        log = make_log(*[self._trace({}, {"risk": "high"}) for _ in range(3)])
        row = self._by_attribute(log)["risk"]
        assert (row["n_traces_with_attribute"], row["n_set_by_first_event"], row["n_unchanged"]) == (3, 0, 0)
        assert row["share"] == 0.0

    def test_start_events_are_not_looked_at(self):
        # Only control-flow (complete) events count: an attribute carried by a start event
        # alone is not reported, and the first complete event is the trace's first event.
        trace = make_trace(
            make_event("A", lifecycle="start", offset=0, piece="p1"),
            make_event("A", lifecycle="complete", offset=5, color="red"),
            make_event("B", lifecycle="complete", offset=60, color="red"),
        )
        by_attribute = self._by_attribute(make_log(trace))
        assert "piece" not in by_attribute
        assert by_attribute["color"]["n_unchanged"] == 1

    def test_a_nan_repeated_in_every_event_is_unchanged(self):
        nan = float("nan")
        log = make_log(self._trace({"cost": nan}, {"cost": nan}))
        assert self._by_attribute(log)["cost"]["n_unchanged"] == 1

    def test_the_event_keys_are_not_attributes(self):
        log = make_log(self._trace({}, {}))
        names = set(self._by_attribute(log))
        assert not names & {"concept:name", "time:timestamp", "lifecycle:transition"}

    def test_log_stats_includes_it(self):
        log = prep.preprocess(make_log(self._trace({"color": "red"}, {})), min_trace_len=2)
        assert any(r["attribute"] == "color" for r in prep.log_stats(log)["start_of_trace_attributes"])


class TestPrepareLogAddsTheStatisticToOldStats:
    """A log prepared before the statistic existed gets it without being downloaded again."""

    def test_the_stats_file_is_completed_from_the_prepared_log(self, tmp_path):
        import json
        spec = LogSpec("toy", "toy.xes", "x", False)
        (tmp_path / "toy.xes").write_text("<log/>")
        (tmp_path / "toy.stats.json").write_text(json.dumps({"n_traces": 4, "n_traces_raw": 5}))
        log = make_log(*[make_trace(make_event("A", offset=0, color="red"), make_event("B", offset=60))
                         for _ in range(4)])
        with patch.object(prep, "_load_log", return_value=log), patch.object(prep, "download_log") as download:
            out = prep.prepare_log(spec, tmp_path, min_trace_len=2)
        download.assert_not_called()
        assert out == tmp_path / "toy.xes"
        saved = json.loads((tmp_path / "toy.stats.json").read_text())
        assert saved["n_traces_raw"] == 5  # what was there is kept
        assert saved["start_of_trace_attributes"][0]["attribute"] == "color"

    def test_a_stats_file_that_has_it_is_left_alone(self, tmp_path):
        import json
        spec = LogSpec("toy", "toy.xes", "x", False)
        (tmp_path / "toy.xes").write_text("<log/>")
        text = json.dumps({"n_traces": 4, "start_of_trace_attributes": []})
        (tmp_path / "toy.stats.json").write_text(text)
        with patch.object(prep, "_load_log") as load:
            prep.prepare_log(spec, tmp_path, min_trace_len=2)
        load.assert_not_called()
        assert (tmp_path / "toy.stats.json").read_text() == text


class TestDownloadLog:
    def _spec(self, sha):
        return LogSpec("toy", "toy.xes", sha, False)

    def _fake_response(self, payload: bytes):
        response = MagicMock()
        response.iter_content.return_value = [payload]
        response.raise_for_status.return_value = None
        return response

    def test_checksum_mismatch_raises(self, tmp_path):
        payload = gzip.compress(b"<log/>")
        with patch.object(prep.requests, "get", return_value=self._fake_response(payload)):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                prep.download_log(self._spec("0" * 64), tmp_path)
        assert not (tmp_path / "raw" / "toy.xes").exists()

    def test_valid_download_decompressed(self, tmp_path):
        payload = gzip.compress(b"<log/>")
        import hashlib
        sha = hashlib.sha256(payload).hexdigest()
        with patch.object(prep.requests, "get", return_value=self._fake_response(payload)):
            path = prep.download_log(self._spec(sha), tmp_path)
        assert path.read_bytes() == b"<log/>"

    def _local_spec(self, sha):
        return LogSpec("toy", "toy.xes", sha, False, local=True)

    def test_local_log_is_checked_and_never_downloaded(self, tmp_path):
        import hashlib
        (tmp_path / "raw").mkdir()
        target = tmp_path / "raw" / "toy.xes"
        target.write_bytes(b"<log/>")
        sha = hashlib.sha256(b"<log/>").hexdigest()
        with patch.object(prep.requests, "get") as get:
            assert prep.download_log(self._local_spec(sha), tmp_path) == target
            # force does not replace a log provided by hand
            assert prep.download_log(self._local_spec(sha), tmp_path, force=True) == target
        get.assert_not_called()
        assert target.read_bytes() == b"<log/>"

    def test_local_log_missing_says_where_to_copy_it(self, tmp_path):
        with patch.object(prep.requests, "get") as get:
            with pytest.raises(FileNotFoundError, match="copy the log to"):
                prep.download_log(self._local_spec("0" * 64), tmp_path)
        get.assert_not_called()

    def test_local_log_with_wrong_checksum_raises(self, tmp_path):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "toy.xes").write_bytes(b"<log/>")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            prep.download_log(self._local_spec("0" * 64), tmp_path)

    def test_existing_file_not_downloaded_again(self, tmp_path):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "toy.xes").write_text("cached")
        with patch.object(prep.requests, "get") as get:
            prep.download_log(self._spec("x"), tmp_path)
        get.assert_not_called()


class TestMergeStatusIntoActivity:
    """BPIC 2013 incidents: lifecycle:transition is a sub-status, not start/complete."""

    @staticmethod
    def _status_trace():
        return make_trace(
            make_event("Accepted", lifecycle="In Progress", offset=10),
            make_event("Queued", lifecycle="Awaiting Assignment", offset=20),
            make_event("Completed", lifecycle="Closed", offset=30),
            case_id="inc1",
        )

    def test_the_activity_is_status_and_sub_status(self):
        out = prep.merge_status_into_activity(make_log(self._status_trace()))
        assert _names(out[0]) == ["Accepted-In Progress", "Queued-Awaiting Assignment", "Completed-Closed"]

    def test_every_event_becomes_a_complete(self):
        out = prep.merge_status_into_activity(make_log(self._status_trace()))
        assert {e["lifecycle:transition"] for e in out[0]} == {"complete"}

    def test_nothing_is_lost_by_the_preprocessing_afterwards(self):
        # without the merge, no event would be a start or a complete and the log would be empty
        log = make_log(self._status_trace())
        assert len(prep.preprocess(log, 3)) == 0
        merged = prep.preprocess(prep.merge_status_into_activity(log), 3)
        assert len(merged) == 1 and _names(merged[0])[0] == "Accepted-In Progress"
        assert prep.log_stats(merged)["n_activities"] == 3

    def test_an_event_without_lifecycle_is_left_alone(self):
        out = prep.merge_status_into_activity(make_log(make_trace(make_event("A", offset=1), case_id="c")))
        assert _names(out[0]) == ["A"]

    def test_the_input_log_is_not_modified_and_trace_attributes_are_kept(self):
        log = make_log(self._status_trace())
        out = prep.merge_status_into_activity(log)
        assert _names(log[0]) == ["Accepted", "Queued", "Completed"]
        assert out[0].attributes["concept:name"] == "inc1"

    def test_timestamps_and_order_are_unchanged(self):
        log = make_log(self._status_trace())
        out = prep.merge_status_into_activity(log)
        assert [e["time:timestamp"] for e in out[0]] == [e["time:timestamp"] for e in log[0]]
