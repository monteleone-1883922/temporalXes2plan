"""Tests for paper_eval.prefixes (all prefixes, goals, deadlines, dedup)."""

from unittest.mock import MagicMock

import pytest

from tests.helpers import make_event, make_trace

from paper_eval import prefixes as px
from paper_eval.config import DEFAULT_BUCKETS
from replay.trace_replayer import EvaluationReplayOutcome

DAY = 86400


def _outcome(marking=("p1",), attrs=None, reached_end=True):
    return EvaluationReplayOutcome(
        reached_end=reached_end, matches_expected_final_state=True,
        error_step=None, error_reason=None, steps=[],
        split_marking=set(marking), split_attributes=dict(attrs or {}),
        final_marking=set(), final_attributes={}, expected_final_attributes={},
        accumulated_duration_seconds=None, within_deadline=None,
        min_modeled_duration_seconds=0.0, max_modeled_duration_seconds=0.0, warnings=[],
    )


def _replayer(marking_for_k=None, attrs_for_k=None, fail=False):
    replayer = MagicMock()

    def all_splits(trace, ks):
        if fail:
            raise RuntimeError("alignment failed")
        return {
            k: _outcome(
                marking=(marking_for_k or {}).get(k, (f"p{k}",)),
                attrs=(attrs_for_k or {}).get(k),
            )
            for k in ks
        }

    replayer.replay_all_splits.side_effect = all_splits
    return replayer


def _trace(case_id, acts, day_offsets, lifecycle=None):
    return make_trace(
        *[make_event(a, lifecycle=lifecycle, offset=int(d * DAY)) for a, d in zip(acts, day_offsets)],
        case_id=case_id,
    )


def _completion(records):
    return [r for r in records if r.goal == px.COMPLETION_GOAL]


class TestPrefixRecords:
    def test_k_from_k_min_to_n_minus_one(self):
        trace = _trace("c1", "ABCDE", [0, 1, 2, 3, 4])
        recs = _completion(px.prefix_records_for_trace(trace, _replayer(), {}, DEFAULT_BUCKETS, 2))
        assert [r.k for r in recs] == [2, 3, 4]
        assert recs[0].prefix_acts == ["A", "B"]
        assert recs[0].true_suffix_acts == ["C", "D", "E"]

    def test_times(self):
        trace = _trace("c1", "ABCD", [0, 1, 3, 6])
        rec = _completion(px.prefix_records_for_trace(trace, _replayer(), {}, DEFAULT_BUCKETS, 2))[0]
        assert rec.elapsed_s == pytest.approx(1 * DAY)
        assert rec.true_rt_s == pytest.approx(5 * DAY)
        assert rec.true_rt_days == pytest.approx(5.0)
        assert rec.case_duration_s == pytest.approx(6 * DAY)

    def test_uses_control_flow_view_of_lifecycle_logs(self):
        trace = make_trace(
            make_event("A", lifecycle="start", offset=0),
            make_event("A", lifecycle="complete", offset=DAY),
            make_event("B", lifecycle="start", offset=2 * DAY),
            make_event("B", lifecycle="complete", offset=3 * DAY),
            make_event("C", lifecycle="complete", offset=4 * DAY),
            case_id="c1",
        )
        replayer = _replayer()
        recs = _completion(px.prefix_records_for_trace(trace, replayer, {}, DEFAULT_BUCKETS, 2))
        assert [r.k for r in recs] == [2]
        assert recs[0].prefix_acts == ["A", "B"]
        assert recs[0].elapsed_s == pytest.approx(2 * DAY)  # complete(A) -> complete(B)
        replayed_trace = replayer.replay_all_splits.call_args.args[0]
        assert [e["lifecycle:transition"] for e in replayed_trace] == ["complete"] * 3

    def test_too_short_trace_yields_nothing(self):
        trace = _trace("c1", "AB", [0, 1])
        assert px.prefix_records_for_trace(trace, _replayer(), {}, DEFAULT_BUCKETS, 2) == []

    def test_two_goals_per_prefix(self):
        trace = _trace("c1", "ABC", [0, 1, 4])
        recs = px.prefix_records_for_trace(trace, _replayer(), {}, DEFAULT_BUCKETS, 2)
        assert [r.goal for r in recs] == ["completion", "deadline"]
        assert recs[0].deadline_s is None  # completion has no deadline


class TestDeadlineGoal:
    """deadline = min(variant mean - elapsed, real remaining time)."""

    ABC = ("A", "B", "C")

    @staticmethod
    def _deadline(variant_means, offsets=(0, 1, 4)):
        # case lasts offsets[-1] days; prefix k=2 has elapsed offsets[1] days.
        trace = _trace("c1", "ABC", offsets)
        recs = px.prefix_records_for_trace(trace, _replayer(), variant_means, DEFAULT_BUCKETS, 2)
        return [r for r in recs if r.goal == px.DEADLINE_GOAL][0]

    def test_real_remaining_time_wins_when_it_is_smaller(self):
        # variant term = 5 - 1 = 4 days, real remaining = 4 - 1 = 3 days
        rec = self._deadline({self.ABC: 5 * DAY})
        assert rec.deadline_s == pytest.approx(3 * DAY)
        assert rec.deadline_source == "real_remaining"
        assert rec.deadline_variant_s == pytest.approx(4 * DAY)  # kept for the paper

    def test_variant_mean_wins_when_it_is_smaller(self):
        # variant term = 3 - 1 = 2 days < real remaining 3 days
        rec = self._deadline({self.ABC: 3 * DAY})
        assert rec.deadline_s == pytest.approx(2 * DAY)
        assert rec.deadline_source == "variant_mean"
        assert rec.deadline_variant_s == pytest.approx(2 * DAY)

    def test_a_tie_goes_to_the_real_remaining_time(self):
        rec = self._deadline({self.ABC: 4 * DAY})  # 4 - 1 = 3 days = real remaining
        assert rec.deadline_source == "real_remaining"

    def test_variant_unseen_in_train_uses_only_the_real_remaining_time(self):
        rec = self._deadline({("A", "X", "C"): 9 * DAY})
        assert rec.deadline_s == pytest.approx(3 * DAY)
        assert rec.deadline_source == "real_remaining"
        assert rec.deadline_variant_s is None

    def test_variant_is_the_whole_traces_activity_sequence_not_the_prefix(self):
        # a mean stored for the PREFIX (A, B) must not be used
        rec = self._deadline({("A", "B"): 1 * DAY})
        assert rec.deadline_source == "real_remaining"

    def test_expired_deadline_flagged(self):
        # elapsed 5 days at the cut, variant mean 3 days -> 3 - 5 = -2 days
        rec = self._deadline({self.ABC: 3 * DAY}, offsets=(0, 5, 6))
        assert rec.expired is True
        assert rec.deadline_s == pytest.approx(-2 * DAY)
        assert rec.deadline_source == "variant_mean"

    def test_deadline_is_never_expired_when_only_the_real_remaining_time_applies(self):
        rec = self._deadline({}, offsets=(0, 5, 6))
        assert rec.expired is False and rec.deadline_s == pytest.approx(1 * DAY)

    def test_the_records_deadline_fields_reach_the_cases_ground_truth(self):
        trace = _trace("c1", "ABC", [0, 1, 4])
        _, instances = px.generate([trace], _replayer(), {self.ABC: 3 * DAY}, DEFAULT_BUCKETS, 2, 1.0)
        case = [i for i in instances if i.goal == px.DEADLINE_GOAL][0].cases[0]
        assert case.deadline_source == "variant_mean"
        assert case.deadline_variant_s == pytest.approx(2 * DAY)


    def test_buckets_configurable(self):
        trace = _trace("c1", "ABCDEF", [0, 1, 2, 3, 4, 5])
        buckets = [(2, 3), (4, None)]
        recs = _completion(px.prefix_records_for_trace(trace, _replayer(), {}, buckets, 2))
        assert [r.bucket for r in recs] == ["2-3", "2-3", "4+", "4+"]

    def test_replay_failure_recorded_not_dropped(self):
        trace = _trace("c1", "ABC", [0, 1, 2])
        recs = _completion(px.prefix_records_for_trace(trace, _replayer(fail=True), {}, DEFAULT_BUCKETS, 2))
        assert len(recs) == 1
        assert recs[0].replay_ok is False
        assert recs[0].init_places is None
        assert "alignment failed" in recs[0].replay_error

    def test_state_from_replay(self):
        trace = _trace("c1", "ABC", [0, 1, 2])
        replayer = _replayer(marking_for_k={2: ("pa", "pb")}, attrs_for_k={2: {"crp": "high"}})
        rec = _completion(px.prefix_records_for_trace(trace, replayer, {}, DEFAULT_BUCKETS, 2))[0]
        assert rec.init_places == ["pa", "pb"]
        assert rec.init_effects == [{"attribute": "crp", "value": "high"}]


class TestGrouping:
    def test_same_prefix_and_state_collapse_into_one_instance(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABD", [0, 1, 5])
        records, instances = px.generate([t1, t2], _replayer(), {}, DEFAULT_BUCKETS, 2, 1.0)
        completion = [i for i in instances if i.goal == px.COMPLETION_GOAL]
        assert len(completion) == 1
        cases = completion[0].cases
        assert [m.case_id for m in cases] == ["c1", "c2"]
        assert [m.true_suffix_acts for m in cases] == [["C"], ["D"]]
        # every completion record points to that one instance
        assert {r.group_id for r in records if r.goal == px.COMPLETION_GOAL} == {completion[0].group_id}

    def test_different_marking_gives_different_instances(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABC", [0, 1, 2])
        replayer = MagicMock()
        replayer.replay_all_splits.side_effect = [
            {2: _outcome(marking=("p_x",))}, {2: _outcome(marking=("p_y",))},
        ]
        _, instances = px.generate([t1, t2], replayer, {}, DEFAULT_BUCKETS, 2, 1.0)
        assert len([i for i in instances if i.goal == px.COMPLETION_GOAL]) == 2

    def test_different_attributes_give_different_instances(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABC", [0, 1, 2])
        replayer = MagicMock()
        replayer.replay_all_splits.side_effect = [
            {2: _outcome(attrs={"crp": "low"})}, {2: _outcome(attrs={"crp": "high"})},
        ]
        _, instances = px.generate([t1, t2], replayer, {}, DEFAULT_BUCKETS, 2, 1.0)
        assert len([i for i in instances if i.goal == px.COMPLETION_GOAL]) == 2

    def test_deadline_goals_grouped_at_pddl_resolution(self):
        # Same prefix/state; elapsed 120 s vs 121 s, variant mean 500 s, real
        # remaining times large: deadlines 380 s / 379 s. With durations in
        # minutes (scale 60) both are written as 6.3 in the PDDL problem -> one
        # deadline instance, whose cases keep their own exact deadlines.
        t1 = make_trace(make_event("A", offset=0), make_event("B", offset=120),
                        make_event("C", offset=1000), case_id="c1")
        t2 = make_trace(make_event("A", offset=0), make_event("B", offset=121),
                        make_event("C", offset=1000), case_id="c2")
        means = {("A", "B", "C"): 500.0}
        _, instances = px.generate([t1, t2], _replayer(), means, DEFAULT_BUCKETS, 2, 60.0)
        deadline_instances = [i for i in instances if i.goal == px.DEADLINE_GOAL]
        assert len(deadline_instances) == 1
        assert [m.deadline_s for m in deadline_instances[0].cases] == pytest.approx([380.0, 379.0])
        assert deadline_instances[0].deadline_s == pytest.approx(6.3 * 60)

    def test_deadlines_differing_in_pddl_give_different_instances(self):
        # deadlines 380 s and 340 s: 6.3 and 5.7 minutes in the PDDL
        t1 = make_trace(make_event("A", offset=0), make_event("B", offset=120),
                        make_event("C", offset=1000), case_id="c1")
        t2 = make_trace(make_event("A", offset=0), make_event("B", offset=160),
                        make_event("C", offset=1000), case_id="c2")
        means = {("A", "B", "C"): 500.0}
        _, instances = px.generate([t1, t2], _replayer(), means, DEFAULT_BUCKETS, 2, 60.0)
        assert len([i for i in instances if i.goal == px.DEADLINE_GOAL]) == 2

    def test_stats(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABD", [0, 1, 5])
        records, instances = px.generate([t1, t2], _replayer(), {}, DEFAULT_BUCKETS, 2, 1.0)
        stats = px.prefix_stats(records, instances)
        assert stats["overall"]["n_prefixes"] == 2
        assert stats["overall"]["n_instances_by_goal"]["completion"] == 1
        # the two traces have different real remaining times (1 and 4 days), so
        # their deadline goals do not merge
        assert stats["overall"]["n_instances_by_goal"]["deadline"] == 2
        assert stats["by_bucket"]["2"]["n_prefixes"] == 2

    def test_jsonl_roundtrip(self, tmp_path):
        rows = [{"a": 1, "b": [1, 2]}, {"a": 2, "b": None}]
        px.write_jsonl(rows, tmp_path / "x.jsonl")
        assert px.read_jsonl(tmp_path / "x.jsonl") == rows
