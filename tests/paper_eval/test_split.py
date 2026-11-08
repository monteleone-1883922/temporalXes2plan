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


class TestVariantMeanDurations:
    """Mean case duration of each variant (distinct activity sequence)."""

    @staticmethod
    def _variant_trace(case_id, acts, hours):
        return make_trace(
            *[make_event(a, offset=int(h * 3600)) for a, h in zip(acts, hours)],
            case_id=case_id,
        )

    def test_mean_over_the_cases_of_each_variant(self):
        traces = [
            self._variant_trace("c1", "ABC", [0, 1, 2]),  # variant ABC, 2 h
            self._variant_trace("c2", "ABC", [0, 2, 4]),  # variant ABC, 4 h
            self._variant_trace("c3", "ABD", [0, 1, 10]),  # variant ABD, 10 h
        ]
        means = sp.variant_mean_durations(traces)
        assert means == {("A", "B", "C"): pytest.approx(3 * 3600), ("A", "B", "D"): pytest.approx(10 * 3600)}

    def test_only_the_control_flow_view_counts(self):
        # start events are not activity executions: the variant is A, B and the
        # duration runs from the first to the last COMPLETE event
        trace = make_trace(
            make_event("A", lifecycle="start", offset=0), make_event("A", lifecycle="complete", offset=3600),
            make_event("B", lifecycle="start", offset=3600), make_event("B", lifecycle="complete", offset=7200),
            case_id="c1",
        )
        assert sp.variant_mean_durations([trace]) == {("A", "B"): pytest.approx(3600)}

    def test_no_traces_gives_no_variants(self):
        assert sp.variant_mean_durations([]) == {}


class TestSpanningCases:
    """Cases still running when the test period starts are taken out of TRAIN.

    In ten_traces, case cK starts on day K and lasts K + 1 days, so it ends on day 2K + 1.
    The first TEST case is c7: the split point is day 7."""

    def test_train_cases_ending_after_the_split_point_are_removed(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, 0.3, 0.2, drop_spanning=True)
        assert ids["removed_spanning"] == ["c4", "c5", "c6"]  # end on days 9, 11, 13
        assert ids["test"] == ["c7", "c8", "c9"]  # the test cases are never touched

    def test_a_case_ending_exactly_at_the_split_point_is_kept(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, 0.3, 0.2, drop_spanning=True)
        assert "c3" in ids["train_fit"] + ids["val"]  # c3 ends on day 7

    def test_fit_and_val_are_divided_on_the_cases_that_remain(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, 0.3, 0.2, drop_spanning=True)
        # 4 cases left: c0..c3, 80% fit (round(3.2) = 3), the newest one is VAL
        assert ids["train_fit"] == ["c0", "c1", "c2"]
        assert ids["val"] == ["c3"]

    def test_every_case_is_in_exactly_one_part(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, 0.3, 0.2, drop_spanning=True)
        everything = ids["train_fit"] + ids["val"] + ids["test"] + ids["removed_spanning"]
        assert sorted(everything) == sorted(f"c{d}" for d in range(10))

    def test_without_the_option_nothing_is_removed(self, ten_traces):
        ids = sp.temporal_split_ids(ten_traces, 0.3, 0.2)
        assert ids["removed_spanning"] == []
        assert (len(ids["train_fit"]), len(ids["val"]), len(ids["test"])) == (6, 1, 3)

    def test_no_case_spanning_the_split_point_changes_nothing(self):
        traces = [_trace(f"d{i}", start_day=i * 3, duration_days=1) for i in range(10)]
        plain = sp.temporal_split_ids(traces, 0.3, 0.2)
        dropped = sp.temporal_split_ids(traces, 0.3, 0.2, drop_spanning=True)
        assert dropped == plain

    def test_make_split_records_the_removed_cases_and_the_flag_changes_the_hash(self, ten_traces, tmp_path):
        log_path = tmp_path / "log.xes"
        log_path.write_text("x")
        kept = sp.make_split(make_log(*ten_traces), log_path, 0.3, 0.2, (0.5,))
        dropped = sp.make_split(make_log(*ten_traces), log_path, 0.3, 0.2, (0.5,), drop_spanning=True)
        assert kept.removed_spanning_ids == [] and dropped.removed_spanning_ids == ["c4", "c5", "c6"]
        assert dropped.params["drop_spanning_train"] is True and kept.params["drop_spanning_train"] is False
        assert dropped.split_hash != kept.split_hash
        assert dropped.train_ids == ["c0", "c1", "c2", "c3"]  # the removed cases are not in TRAIN

    def test_thresholds_are_computed_on_the_remaining_train(self, ten_traces, tmp_path):
        log_path = tmp_path / "log.xes"
        log_path.write_text("x")
        split = sp.make_split(make_log(*ten_traces), log_path, 0.3, 0.2, (0.5,), drop_spanning=True)
        # TRAIN = c0..c3 with durations 1, 2, 3, 4 days -> median 2.5 days
        assert split.deadline_thresholds_s["q50"] == pytest.approx(2.5 * DAY)

    def test_the_removed_ids_survive_write_and_read(self, ten_traces, tmp_path):
        log_path = tmp_path / "log.xes"
        log_path.write_text("x")
        split = sp.make_split(make_log(*ten_traces), log_path, 0.3, 0.2, (0.5,), drop_spanning=True)
        sp.write_split(split, tmp_path / "split.json")
        assert sp.read_split(tmp_path / "split.json") == split

    def test_a_split_file_written_before_the_option_still_loads(self):
        old = {"train_fit_ids": ["a"], "val_ids": ["b"], "test_ids": ["c"], "deadline_thresholds_s": {},
               "params": {"strategy": "temporal"}, "log_sha256": "x", "split_hash": "h"}
        assert sp.SplitResult.from_json(old).removed_spanning_ids == []
