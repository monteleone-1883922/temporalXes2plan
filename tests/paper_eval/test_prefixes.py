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

    def test_deadline_goals(self):
        # case lasts 4 days; prefix k=2 elapsed 1 day.
        trace = _trace("c1", "ABC", [0, 1, 4])
        thresholds = {"q50": 3 * DAY, "q90": 5 * DAY}
        recs = px.prefix_records_for_trace(trace, _replayer(), thresholds, DEFAULT_BUCKETS, 2)
        by_goal = {r.goal: r for r in recs}
        assert set(by_goal) == {"completion", "deadline_q50", "deadline_q90"}
        assert by_goal["deadline_q50"].deadline_s == pytest.approx(2 * DAY)
        assert by_goal["deadline_q50"].label_feasible is False  # 4 days > 3
        assert by_goal["deadline_q90"].label_feasible is True   # 4 days <= 5
        assert by_goal["completion"].deadline_s is None

    def test_expired_deadline_flagged(self):
        trace = _trace("c1", "ABC", [0, 5, 6])
        recs = px.prefix_records_for_trace(trace, _replayer(), {"q50": 3 * DAY}, DEFAULT_BUCKETS, 2)
        deadline = [r for r in recs if r.goal == "deadline_q50"][0]
        assert deadline.expired is True
        assert deadline.deadline_s < 0

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
        assert len(instances) == 1
        cases = instances[0].cases
        assert [m.case_id for m in cases] == ["c1", "c2"]
        assert [m.true_suffix_acts for m in cases] == [["C"], ["D"]]
        assert {r.group_id for r in records} == {0}

    def test_different_marking_gives_different_instances(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABC", [0, 1, 2])
        replayer = MagicMock()
        replayer.replay_all_splits.side_effect = [
            {2: _outcome(marking=("p_x",))}, {2: _outcome(marking=("p_y",))},
        ]
        _, instances = px.generate([t1, t2], replayer, {}, DEFAULT_BUCKETS, 2, 1.0)
        assert len(instances) == 2

    def test_different_attributes_give_different_instances(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABC", [0, 1, 2])
        replayer = MagicMock()
        replayer.replay_all_splits.side_effect = [
            {2: _outcome(attrs={"crp": "low"})}, {2: _outcome(attrs={"crp": "high"})},
        ]
        _, instances = px.generate([t1, t2], replayer, {}, DEFAULT_BUCKETS, 2, 1.0)
        assert len(instances) == 2

    def test_deadline_goals_grouped_at_pddl_resolution(self):
        # Same prefix/state; elapsed 120 s vs 121 s. With durations in
        # minutes (scale 60) the deadlines 380 s / 379 s are both written as
        # 6.3 in the PDDL problem -> one deadline instance, whose cases
        # keep their own labels.
        t1 = make_trace(make_event("A", offset=0), make_event("B", offset=120),
                        make_event("C", offset=200), case_id="c1")
        t2 = make_trace(make_event("A", offset=0), make_event("B", offset=121),
                        make_event("C", offset=900), case_id="c2")
        _, instances = px.generate([t1, t2], _replayer(), {"q50": 500.0}, DEFAULT_BUCKETS, 2, 60.0)
        deadline_instances = [i for i in instances if i.goal == "deadline_q50"]
        assert len(deadline_instances) == 1
        assert [m.label_feasible for m in deadline_instances[0].cases] == [True, False]
        assert deadline_instances[0].deadline_s == pytest.approx(6.3 * 60)

    def test_deadlines_differing_in_pddl_give_different_instances(self):
        t1 = make_trace(make_event("A", offset=0), make_event("B", offset=120),
                        make_event("C", offset=200), case_id="c1")
        t2 = make_trace(make_event("A", offset=0), make_event("B", offset=160),
                        make_event("C", offset=900), case_id="c2")
        _, instances = px.generate([t1, t2], _replayer(), {"q50": 500.0}, DEFAULT_BUCKETS, 2, 60.0)
        assert len([i for i in instances if i.goal == "deadline_q50"]) == 2

    def test_stats(self):
        t1 = _trace("c1", "ABC", [0, 1, 2])
        t2 = _trace("c2", "ABD", [0, 1, 5])
        records, instances = px.generate([t1, t2], _replayer(), {"q50": 3 * DAY}, DEFAULT_BUCKETS, 2, 1.0)
        stats = px.prefix_stats(records, instances)
        assert stats["overall"]["n_prefixes"] == 2
        assert stats["overall"]["n_instances_by_goal"]["completion"] == 1
        assert stats["by_bucket"]["2"]["n_prefixes"] == 2

    def test_jsonl_roundtrip(self, tmp_path):
        rows = [{"a": 1, "b": [1, 2]}, {"a": 2, "b": None}]
        px.write_jsonl(rows, tmp_path / "x.jsonl")
        assert px.read_jsonl(tmp_path / "x.jsonl") == rows
