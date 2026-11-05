"""Tests for paper_eval.query (SQL views over prefixes.jsonl and plan_results.jsonl)."""
import pytest

duckdb = pytest.importorskip("duckdb")

from paper_eval import prefixes as px
from paper_eval import query as q


def _case(case_id, group_id, replayable, goal="completion", k=2, bucket="2"):
    return {"case_id": case_id, "group_id": group_id, "is_replayable": replayable,
            "goal": goal, "k": k, "bucket": bucket, "reached_end": True}


def _plan(group_id, status="solved", planning_s=1.0, goal="completion", k=2, bucket="2"):
    return {"group_id": group_id, "status": status, "goal": goal, "k": k, "bucket": bucket,
            "grounding_s": 0.5, "planning_s": planning_s, "wall_s": planning_s + 0.5, "plan_length": 3}


@pytest.fixture
def model_dir(tmp_path):
    # group 0: 3 cases (2 replayable) = mixed; group 1: 1 case, replayable; group 2: 1 case, not;
    # group 3: planned never (no row in plan_results).
    px.write_jsonl([
        _case("a", 0, True), _case("b", 0, True), _case("c", 0, False),
        _case("d", 1, True), _case("e", 2, False), _case("f", 3, True),
    ], tmp_path / "prefixes.jsonl")
    px.write_jsonl([
        _plan(0, planning_s=1.0), _plan(1, planning_s=3.0), _plan(2, status="timeout", planning_s=None),
    ], tmp_path / "plan_results.jsonl")
    return tmp_path


class TestViews:
    def test_cases_joins_the_plan_outcome_to_every_case(self, model_dir):
        con = q.connect(model_dir)
        rows = dict(con.execute("SELECT case_id, status FROM cases").fetchall())
        assert rows == {"a": "solved", "b": "solved", "c": "solved", "d": "solved",
                        "e": "timeout", "f": None}  # f: instance never planned

    def test_replayability_of_an_instance(self, model_dir):
        con = q.connect(model_dir)
        got = dict(con.execute("SELECT group_id, replayability FROM instances_view").fetchall())
        assert got == {0: "mixed", 1: "all", 2: "none", 3: "all"}

    def test_counts_per_instance(self, model_dir):
        con = q.connect(model_dir)
        row = con.execute(
            "SELECT n_cases, n_cases_replayable FROM instances_view WHERE group_id = 0").fetchone()
        assert row == (3, 2)

    def test_without_plan_results_the_views_still_work(self, tmp_path):
        px.write_jsonl([_case("a", 0, True)], tmp_path / "prefixes.jsonl")
        con = q.connect(tmp_path)
        assert con.execute("SELECT status FROM cases").fetchall() == [(None,)]

    def test_missing_prefixes_says_which_phase_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="prefixes"):
            q.connect(tmp_path)


class TestRq1Times:
    def test_instance_weight_counts_each_run_once(self, model_dir):
        df = q.rq1_times(q.connect(model_dir), "completion", weight="instance")
        assert df["n"].iloc[0] == 2  # groups 0 and 1 (2 is a timeout, 3 never ran)
        assert df["planning_s_mean"].iloc[0] == pytest.approx(2.0)  # (1 + 3) / 2

    def test_case_weight_counts_each_run_once_per_case(self, model_dir):
        df = q.rq1_times(q.connect(model_dir), "completion", weight="case")
        assert df["n"].iloc[0] == 4  # a, b, c (group 0) and d (group 1)
        assert df["planning_s_mean"].iloc[0] == pytest.approx(1.5)  # (1 + 1 + 1 + 3) / 4

    def test_only_keeps_instances_whose_cases_are_all_replayable(self, model_dir):
        df = q.rq1_times(q.connect(model_dir), "completion", replayable="only")
        assert df["n"].iloc[0] == 1 and df["planning_s_mean"].iloc[0] == pytest.approx(3.0)

    def test_exclude_keeps_instances_with_no_replayable_case(self, model_dir):
        # group 2 is the only 'none' one and it timed out: nothing solved
        assert q.rq1_times(q.connect(model_dir), "completion", replayable="exclude").empty

    def test_case_weight_filters_each_case(self, model_dir):
        df = q.rq1_times(q.connect(model_dir), "completion", replayable="only", weight="case")
        assert df["n"].iloc[0] == 3  # a, b, d

    def test_bad_arguments_are_rejected(self, model_dir):
        con = q.connect(model_dir)
        with pytest.raises(ValueError):
            q.rq1_times(con, "completion", replayable="x")
        with pytest.raises(ValueError):
            q.rq1_times(con, "completion", weight="x")
