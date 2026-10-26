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

    def test_existing_file_not_downloaded_again(self, tmp_path):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "toy.xes").write_text("cached")
        with patch.object(prep.requests, "get") as get:
            prep.download_log(self._spec("x"), tmp_path)
        get.assert_not_called()
