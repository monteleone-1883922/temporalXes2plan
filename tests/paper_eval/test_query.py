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
            "grounding_s": 0.5, "planning_s": planning_s, "wall_s": None if planning_s is None else planning_s + 0.5, "plan_length": 3}


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


# ---------------------------------------------------------------------------
# A complete run: every table, deadline fit, several logs
# ---------------------------------------------------------------------------
import json


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _deadline_case(case_id, group_id, reached_end, min_s, max_s, deadline_s, goal="deadline"):
    return {"case_id": case_id, "group_id": group_id, "goal": goal, "k": 2, "bucket": "2",
            "is_replayable": True, "reached_end": reached_end, "min_modeled_duration_s": min_s,
            "max_modeled_duration_s": max_s, "deadline_s": deadline_s}


def _full_run(root, log="toy", split_hash="hash", plan=True):
    """results/<log>/<split_hash>/ with every file of a finished run, and data/<log>.stats.json."""
    log_dir = root / "results" / log
    model_dir = log_dir / split_hash
    model_dir.mkdir(parents=True)
    (root / "data").mkdir(exist_ok=True)
    _write_json(log_dir / "split.json", {
        "train_fit_ids": ["t1", "t2"], "val_ids": ["v1"], "test_ids": ["a", "b", "c"],
        "deadline_thresholds_s": {}, "params": {"test_pct": 0.3, "val_pct": 0.2},
        "log_sha256": "abc", "split_hash": split_hash})
    _write_json(root / "data" / f"{log}.stats.json", {"n_traces": 6, "lifecycle": {"complete": 10}})
    _write_json(model_dir / "search_trials.json", [
        {"trial_number": 0, "score": 1.5, "rejected_reason": None, "max_effect_groups": 2},
        {"trial_number": 1, "score": None, "rejected_reason": "too many groups", "max_effect_groups": 9}])
    _write_json(model_dir / "current.json", {
        "attribute_catalog": {
            "crp": {"type": "categorical", "possible_values": ["high", "low"], "bin_boundaries": []},
            "Diagnose": {"type": "categorical", "possible_values": ["none"], "bin_boundaries": []}},
        "transitions": {
            "t_a": {"activity_name": "A", "duration": {
                "effective_min": 1.0, "effective_max": 9.0, "mean": 4.0, "std_dev": 2.0,
                "observed_min": 1.0, "observed_max": 9.0, "count": 5, "source": "inter_event"}},
            "t_b": {"activity_name": "B", "duration": None}}})
    _write_json(model_dir / "attribute_filter.json", {
        "ignored_by_user": ["resource"], "kept_by_user": ["Diagnose"],
        "noise_filter": {"counter": "N3"}, "unused_attribute_filter": ["u1"]})
    _write_json(model_dir / "model_stats.json", {
        "model_dir": "ignored", "petri_net": {"places": 5, "transitions": 7},
        "attributes": {"in_catalog": 2, "by_type": {"categorical": 2}},
        "split": {"split_hash": "x"}, "attribute_filter": {"x": 1}, "log": {"x": 1}})
    _write_json(model_dir / "prefixes_meta.json", {"deadline_variant_only": True})
    px.write_jsonl([
        _deadline_case("a", 0, True, 4.0, 9.0, 6.0),     # best case fits, worst does not
        _deadline_case("b", 1, True, 7.0, 9.0, 6.0),     # neither fits
        _deadline_case("c", 2, True, 1.0, 2.0, None, goal="completion"),
        _deadline_case("d", 3, None, None, None, 5.0),   # replay failed: unknown
    ], model_dir / "prefixes.jsonl")
    px.write_jsonl([{"group_id": g, "goal": "deadline", "k": 2, "bucket": "2", "cases": []}
                    for g in range(4)], model_dir / "instances.jsonl")
    if plan:
        _write_json(model_dir / "plan_meta.json", {
            "settings": {"timeout_s": 60, "deadline_variant_only": True},
            "environment": {"host": "h"}, "duration_scale_factor": 60.0, "cost_weight": 0.5})
        px.write_jsonl([_plan(0), _plan(1, status="unsolvable_structural", planning_s=0.4)] + [
            {**_plan(2), "log": log}], model_dir / "plan_results.jsonl")
    return model_dir


class TestWholeRun:
    def test_runs_table(self, tmp_path):
        con = q.connect(_full_run(tmp_path), tmp_path / "data")
        row = con.execute(
            "SELECT log, split_hash, n_train_fit, n_val, n_test, deadline_variant_only, test_pct FROM runs"
        ).fetchone()
        assert row == ("toy", "hash", 2, 1, 3, True, 0.3)

    def test_logs_table_comes_from_the_prepare_stats(self, tmp_path):
        con = q.connect(_full_run(tmp_path), tmp_path / "data")
        assert con.execute("SELECT log, n_traces, lifecycle_complete FROM logs").fetchall() == [("toy", 6, 10)]

    def test_case_split_counts_per_part(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert dict(con.execute("SELECT part, count(*) FROM case_split GROUP BY part").fetchall()) == {
            "train_fit": 2, "val": 1, "test": 3}

    def test_search_trials(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute(
            "SELECT trial_number, score, rejected_reason FROM search_trials ORDER BY trial_number"
        ).fetchall() == [(0, 1.5, None), (1, None, "too many groups")]

    def test_attribute_catalog_has_kept_and_removed_attributes(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        got = {r[0]: r[1:] for r in con.execute(
            "SELECT attribute, status, reason, type, n_values, kept_by_user FROM attribute_catalog").fetchall()}
        assert got == {
            "crp": ("kept", None, "categorical", 2, False),
            "Diagnose": ("kept", None, "categorical", 1, True),
            "counter": ("removed_noise", "N3", None, None, False),
            "u1": ("removed_unused", None, None, None, False),
            "resource": ("ignored_by_user", None, None, None, False),
        }

    def test_a_transition_without_duration_has_null_fields(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        got = {r[0]: r[1:] for r in con.execute(
            "SELECT transition, activity_name, effective_min, effective_max, count, source "
            "FROM transition_durations").fetchall()}
        assert got == {"t_a": ("A", 1.0, 9.0, 5, "inter_event"), "t_b": ("B", None, None, None, None)}

    def test_models_and_plan_meta_are_flattened(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute("SELECT petri_net_places, petri_net_transitions, attributes_in_catalog FROM models"
                           ).fetchone() == (5, 7, 2)
        assert con.execute("SELECT settings_timeout_s, environment_host, duration_scale_factor FROM plan_meta"
                           ).fetchone() == (60, "h", 60.0)
        # what other tables already hold is not repeated in `models`
        assert "attribute_filter_x" not in {r[0] for r in con.execute("DESCRIBE models").fetchall()}

    def test_instances_come_from_instances_jsonl(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute("SELECT count(*) FROM instances").fetchone() == (4,)

    def test_missing_files_give_empty_tables_not_errors(self, tmp_path):
        px.write_jsonl([_case("a", 0, True)], tmp_path / "prefixes.jsonl")
        con = q.connect(tmp_path)
        for table in ("models", "plan_meta", "search_trials", "attribute_catalog",
                      "transition_durations", "case_split", "instances", "plan_results", "logs"):
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,), table
        assert con.execute("SELECT count(*) FROM runs").fetchone() == (1,)

    def test_a_half_written_last_line_of_plan_results_is_ignored(self, tmp_path):
        model_dir = _full_run(tmp_path)
        with open(model_dir / "plan_results.jsonl", "a", encoding="utf-8") as fh:
            fh.write('{"group_id": 3, "status": "so')  # killed while writing
        con = q.connect(model_dir)
        assert con.execute("SELECT count(*) FROM plan_results").fetchone() == (3,)


class TestFitsDeadline:
    def test_best_and_worst_case(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        got = {r[0]: r[1:] for r in con.execute(
            "SELECT case_id, fits_deadline_best_case, fits_deadline_worst_case FROM cases").fetchall()}
        assert got["a"] == (True, False)     # min 4 <= 6 < max 9
        assert got["b"] == (False, False)    # min 7 > 6
        assert got["c"] == (None, None)      # completion goal: no deadline
        assert got["d"] == (None, None)      # replay failed: unknown

    def test_instance_status_can_be_compared_with_the_fit(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        got = dict(con.execute(
            "SELECT status, fits_deadline_best_case FROM cases WHERE case_id IN ('a', 'b')").fetchall())
        assert got == {"solved": True, "unsolvable_structural": False}


class TestSeveralLogs:
    def test_discover_runs_finds_every_built_model(self, tmp_path):
        _full_run(tmp_path, "toy", "h1")
        _full_run(tmp_path, "other", "h2")
        assert [(r.log, r.split_hash) for r in q.discover_runs(tmp_path / "results")] == [
            ("other", "h2"), ("toy", "h1")]

    def test_every_table_holds_both_logs_and_group_ids_do_not_mix(self, tmp_path):
        _full_run(tmp_path, "toy", "h1")
        other = _full_run(tmp_path, "other", "h2")
        # same group_id 0 in both logs, a different outcome in the second
        px.write_jsonl([_plan(0, status="timeout", planning_s=None)], other / "plan_results.jsonl")
        con = q.connect_all(tmp_path / "results", tmp_path / "data")
        assert {r[0] for r in con.execute("SELECT log FROM runs").fetchall()} == {"toy", "other"}
        assert {r[0] for r in con.execute("SELECT DISTINCT log FROM prefixes").fetchall()} == {"toy", "other"}
        got = dict(con.execute(
            "SELECT log, status FROM cases WHERE group_id = 0 AND case_id = 'a'").fetchall())
        assert got == {"toy": "solved", "other": "timeout"}
        assert con.execute("SELECT count(*) FROM cases").fetchone() == (8,)  # no cross join

    def test_rq1_times_can_be_restricted_to_one_log(self, tmp_path):
        _full_run(tmp_path, "toy", "h1")
        other = _full_run(tmp_path, "other", "h2")
        px.write_jsonl([_plan(0, planning_s=9.0, goal="deadline")], other / "plan_results.jsonl")
        con = q.connect_all(tmp_path / "results")
        df = q.rq1_times(con, "deadline", log="other")
        assert df["planning_s_mean"].iloc[0] == pytest.approx(9.0)
