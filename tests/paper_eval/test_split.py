"""Tests for paper_eval.split (temporal split, validation set, T_q)."""

import pytest

from tests.helpers import make_event, make_log, make_trace

from paper_eval import split as sp

DAY = 86400


def _trace(case_id, start_day, duration_days=1.0, lifecycle=None):
    start = int(start_day * DAY)
    end = start + int(duration_days * DAY)
    return make_trace(
        make_event("A", lifecycle=lifecycle, offset=start),
        make_event("B", lifecycle=lifecycle, offset=(start + end) // 2),
        make_event("C", lifecycle=lifecycle, offset=end),
        case_id=case_id,
    )


@pytest.fixture
def ten_traces():
    # Case ids deliberately not in chronological order.
    order = [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]
    return [_trace(f"c{d}", start_day=d, duration_days=d + 1) for d in order]


class TestTemporalSplit:
    def test_parts_are_disjoint_and_complete(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, test_pct=0.3, val_pct=0.2)
        all_ids = ids["train_fit"] + ids["val"] + ids["test"]
        assert sorted(all_ids) == sorted(f"c{d}" for d in range(10))
        assert len(set(all_ids)) == 10

    def test_sizes(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, test_pct=0.3, val_pct=0.2)
        assert (len(ids["train_fit"]), len(ids["val"]), len(ids["test"])) == (6, 1, 3)

    def test_test_cases_start_after_every_train_case(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, test_pct=0.3, val_pct=0.2)
        assert ids["test"] == ["c7", "c8", "c9"]
        assert ids["val"] == ["c6"]
        assert ids["train_fit"] == ["c0", "c1", "c2", "c3", "c4", "c5"]

    def test_ties_broken_by_case_id(self):
        traces = [_trace("b", 0), _trace("a", 0), _trace("c", 1)]
        assert sp.chronological_case_ids(traces) == ["a", "b", "c"]

    def test_start_time_uses_control_flow_view(self):
        # 'late' has a start event early on, but its first complete is later
        # than 'early''s first complete.
        late = make_trace(
            make_event("A", lifecycle="start", offset=0),
            make_event("A", lifecycle="complete", offset=10 * DAY),
            case_id="late",
        )
        early = make_trace(make_event("A", lifecycle="complete", offset=DAY), case_id="early")
        assert sp.chronological_case_ids([late, early]) == ["early", "late"]


class TestDeadlineThresholds:
    def test_quantiles_over_given_traces_only(self):
        traces = [_trace(f"c{i}", start_day=i, duration_days=d) for i, d in enumerate([1, 2, 3, 4, 5])]
        th = sp.deadline_thresholds(traces, (0.5, 0.75))
        assert th == {"q50": pytest.approx(3 * DAY), "q75": pytest.approx(4 * DAY)}

    def test_make_split_thresholds_ignore_test_cases(self, ten_traces, tmp_path):
        log_path = tmp_path / "log.xes"
        log_path.write_text("x")
        split = sp.make_split(make_log(*ten_traces), log_path, 0.3, 0.2, (0.5,))
        # TRAIN = c0..c6 with durations 1..7 days -> median 4 days.
        assert split.deadline_thresholds_s["q50"] == pytest.approx(4 * DAY)


class TestSplitHashAndIO:
    def test_hash_stable_and_param_sensitive(self):
        params = {"strategy": "temporal", "test_pct": 0.3}
        assert sp.compute_split_hash("abc", params) == sp.compute_split_hash("abc", dict(params))
        assert sp.compute_split_hash("abc", params) != sp.compute_split_hash("abc", {**params, "test_pct": 0.2})
        assert sp.compute_split_hash("abc", params) != sp.compute_split_hash("abd", params)

    def test_roundtrip(self, ten_traces, tmp_path):
        log_path = tmp_path / "log.xes"
        log_path.write_text("x")
        split = sp.make_split(make_log(*ten_traces), log_path, 0.3, 0.2, (0.5, 0.9))
        sp.write_split(split, tmp_path / "out" / "split.json")
        loaded = sp.read_split(tmp_path / "out" / "split.json")
        assert loaded == split
        assert loaded.train_ids == split.train_fit_ids + split.val_ids
