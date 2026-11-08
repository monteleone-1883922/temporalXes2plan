"""Tests for paper_eval.query (SQL views over prefixes.jsonl and plan_results.jsonl)."""
from pathlib import Path

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


def _attribute(attribute, status, type_=None, reason=None, values=None, bins=None, kept=False, distinct=None):
    return {"attribute": attribute, "type": type_, "status": status, "reason": reason,
            "kept_by_user": kept, "n_values": None if values is None else len(values),
            "possible_values": values, "n_bin_boundaries": None if bins is None else len(bins),
            "bin_boundaries": bins, "n_distinct_observed": distinct}


def _transition(name, activity, with_duration=True):
    return {"transition": name, "activity_name": activity, "n_effect_groups": 1, "cost": 0.0,
            "has_duration": with_duration,
            "effective_min": 1.0 if with_duration else None, "effective_max": 9.0 if with_duration else None,
            "mean": 4.0 if with_duration else None, "std_dev": 2.0 if with_duration else None,
            "observed_min": 1.0 if with_duration else None, "observed_max": 9.0 if with_duration else None,
            "count": 5 if with_duration else None, "source": "inter_event" if with_duration else None}


def _full_run(root, log="toy", split_hash="hash", plan=True):
    """results/<log>/<split_hash>/ with every file of a finished run, and data/<log>.stats.json."""
    model_dir = root / "results" / log / split_hash
    model_dir.mkdir(parents=True)
    (root / "data").mkdir(exist_ok=True)
    _write_json(model_dir / "split.json", {
        "train_fit_ids": ["t1", "t2"], "val_ids": ["v1"], "test_ids": ["a", "b", "c"],
        "removed_spanning_ids": ["r1", "r2"],
        "deadline_thresholds_s": {}, "params": {"test_pct": 0.3, "val_pct": 0.2, "quantiles": [0.5]},
        "log_sha256": "abc", "split_hash": split_hash})
    _write_json(model_dir / "model_meta.json", {
        "split_hash": split_hash, "discovery_algorithm": "powl", "seed": 42,
        "search_n_trials": 30, "coverage_percentage": 0.001, "unit_costs": True})
    (model_dir / "current.json").write_text("{}")  # what marks a built model
    _write_json(root / "data" / f"{log}.stats.json", {
        "n_traces": 6, "lifecycle": {"complete": 10}, "trace_length": {"min": 2, "max": 9}})
    _write_json(model_dir / "search_trials.json", [
        {"trial_number": 0, "score": 1.5, "rejected_reason": None, "max_effect_groups": 2},
        {"trial_number": 1, "score": None, "rejected_reason": "too many groups", "max_effect_groups": 9}])
    px.write_jsonl([
        _attribute("crp", "kept", "categorical", values=["high", "low"]),
        _attribute("Diagnose", "kept", "categorical", values=["none"], kept=True),
        _attribute("lactic", "removed_unused", "numerical", values=["gte_2", "lte_2", "none"], bins=[2.0]),
        _attribute("counter", "removed_noise", "numerical", reason="N3 counter-like", distinct=99),
        _attribute("resource", "ignored_by_user"),
    ], model_dir / "attributes.jsonl")
    px.write_jsonl([_transition("t_a", "A"), _transition("t_b", "B", with_duration=False)],
                   model_dir / "transition_durations.jsonl")
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
            "SELECT log, split_hash, n_train_fit, n_val, n_test, deadline_variant_only, test_pct, log_sha256, "
            "n_removed_spanning, discovery_algorithm, seed, unit_costs FROM runs"
        ).fetchone()
        assert row == ("toy", "hash", 2, 1, 3, True, 0.3, "abc", 2, "powl", 42, True)

    def test_a_run_without_model_meta_has_no_algorithm_or_seed(self, tmp_path):
        model_dir = _full_run(tmp_path)
        (model_dir / "model_meta.json").unlink()
        assert q.connect(model_dir).execute("SELECT discovery_algorithm, seed, unit_costs FROM runs"
                                            ).fetchone() == (None, None, None)

    def test_a_model_meta_written_before_unit_costs_has_no_value_for_it(self, tmp_path):
        model_dir = _full_run(tmp_path)
        _write_json(model_dir / "model_meta.json", {"discovery_algorithm": "powl", "seed": 42})
        assert q.connect(model_dir).execute("SELECT discovery_algorithm, unit_costs FROM runs"
                                            ).fetchone() == ("powl", None)

    def test_a_split_written_before_the_removal_option_removes_no_case(self, tmp_path):
        model_dir = _full_run(tmp_path)
        split = json.loads((model_dir / "split.json").read_text())
        del split["removed_spanning_ids"]
        _write_json(model_dir / "split.json", split)
        con = q.connect(model_dir)
        assert con.execute("SELECT n_removed_spanning FROM runs").fetchone() == (None,)
        assert {r[0] for r in con.execute("SELECT DISTINCT part FROM case_split").fetchall()} == {
            "train_fit", "val", "test"}

    def test_a_run_without_prefixes_meta_used_the_min_rule(self, tmp_path):
        model_dir = _full_run(tmp_path)
        (model_dir / "prefixes_meta.json").unlink()
        assert q.connect(model_dir).execute("SELECT deadline_variant_only FROM runs").fetchone() == (False,)

    def test_logs_table_comes_from_the_prepare_stats(self, tmp_path):
        con = q.connect(_full_run(tmp_path), tmp_path / "data")
        assert con.execute("SELECT log, n_traces, lifecycle.complete, trace_length.max FROM logs"
                           ).fetchall() == [("toy", 6, 10, 9)]

    def test_case_split_counts_per_part(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert dict(con.execute("SELECT part, count(*) FROM case_split GROUP BY part").fetchall()) == {
            "train_fit": 2, "val": 1, "test": 3, "removed_spanning": 2}

    def test_search_trials(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute(
            "SELECT trial_number, score, rejected_reason FROM search_trials ORDER BY trial_number"
        ).fetchall() == [(0, 1.5, None), (1, None, "too many groups")]

    def test_attributes_come_as_written_with_every_status(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        got = {r[0]: r[1:] for r in con.execute(
            "SELECT attribute, status, type, n_values, n_bin_boundaries, n_distinct_observed, kept_by_user "
            "FROM attributes").fetchall()}
        assert got == {
            "crp": ("kept", "categorical", 2, None, None, False),
            "Diagnose": ("kept", "categorical", 1, None, None, True),
            "lactic": ("removed_unused", "numerical", 3, 1, None, False),
            "counter": ("removed_noise", "numerical", None, None, 99, False),
            "resource": ("ignored_by_user", None, None, None, None, False),
        }

    def test_the_values_and_the_boundaries_of_a_removed_attribute_are_kept(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute("SELECT possible_values, bin_boundaries FROM attributes WHERE attribute = 'lactic'"
                           ).fetchone() == (["gte_2", "lte_2", "none"], [2.0])

    def test_a_transition_without_duration_has_null_fields(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        got = {r[0]: r[1:] for r in con.execute(
            "SELECT transition, activity_name, has_duration, effective_min, effective_max, count, source "
            "FROM transition_durations").fetchall()}
        assert got == {"t_a": ("A", True, 1.0, 9.0, 5, "inter_event"),
                       "t_b": ("B", False, None, None, None, None)}

    def test_models_and_plan_meta_keep_their_nesting_as_struct_columns(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute("SELECT petri_net.places, petri_net.transitions, attributes.in_catalog FROM models"
                           ).fetchone() == (5, 7, 2)
        assert con.execute("SELECT settings.timeout_s, environment.host, duration_scale_factor FROM plan_meta"
                           ).fetchone() == (60, "h", 60.0)
        # what other tables already hold is not repeated in `models`
        assert {"model_dir", "split", "attribute_filter"}.isdisjoint(
            {r[0] for r in con.execute("DESCRIBE models").fetchall()})

    def test_instances_come_from_instances_jsonl(self, tmp_path):
        con = q.connect(_full_run(tmp_path))
        assert con.execute("SELECT count(*) FROM instances").fetchone() == (4,)

    def test_missing_files_give_empty_tables_not_errors(self, tmp_path):
        px.write_jsonl([_case("a", 0, True)], tmp_path / "prefixes.jsonl")
        con = q.connect(tmp_path)
        for table in ("models", "plan_meta", "search_trials", "attributes", "transition_durations",
                      "case_split", "instances", "plan_results", "logs"):
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,), table
        assert con.execute("SELECT count(*), any_value(n_test) FROM runs").fetchone() == (1, None)

    def test_a_half_written_last_line_of_plan_results_is_ignored(self, tmp_path):
        model_dir = _full_run(tmp_path)
        with open(model_dir / "plan_results.jsonl", "a", encoding="utf-8") as fh:
            fh.write('{"group_id": 3, "status": "so')  # killed while writing
        con = q.connect(model_dir)
        assert con.execute("SELECT count(*) FROM plan_results").fetchone() == (3,)

    def test_an_empty_prefixes_file_says_which_phase_to_run(self, tmp_path):
        (tmp_path / "prefixes.jsonl").write_text("")
        with pytest.raises(FileNotFoundError, match="prefixes"):
            q.connect(tmp_path)

    def test_a_relative_path_still_gives_the_log_and_hash_names(self, tmp_path, monkeypatch):
        model_dir = _full_run(tmp_path)
        monkeypatch.chdir(model_dir)
        assert q.connect(Path(".")).execute("SELECT log, split_hash FROM runs").fetchone() == ("toy", "hash")


class TestEachRunUsesItsOwnSplit:
    def test_an_old_hash_folder_keeps_its_own_split(self, tmp_path):
        old = _full_run(tmp_path, "toy", "old")
        new = _full_run(tmp_path, "toy", "new")
        _write_json(new / "split.json", {
            "train_fit_ids": ["x"], "val_ids": [], "test_ids": ["y", "z"], "deadline_thresholds_s": {},
            "params": {"test_pct": 0.5, "val_pct": 0.0}, "log_sha256": "def", "split_hash": "new"})
        con = q.connect_all(tmp_path / "results")
        got = {r[0]: r[1:] for r in con.execute("SELECT split_hash, n_train_fit, n_test, log_sha256 FROM runs"
                                                 ).fetchall()}
        assert got == {"old": (2, 3, "abc"), "new": (1, 2, "def")}
        assert con.execute("SELECT count(*) FROM case_split WHERE split_hash = 'new'").fetchone() == (3,)
        assert old.exists()


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

    def test_a_folder_without_a_built_model_is_not_a_run(self, tmp_path):
        model_dir = _full_run(tmp_path, "toy", "h1")
        (model_dir / "current.json").unlink()
        assert q.discover_runs(tmp_path / "results") == []

    def test_every_table_holds_both_logs_and_group_ids_do_not_mix(self, tmp_path):
        _full_run(tmp_path, "toy", "h1")
        other = _full_run(tmp_path, "other", "h2")
        # same group_id 0 in both logs, a different outcome in the second
        px.write_jsonl([_plan(0, status="timeout", planning_s=None)], other / "plan_results.jsonl")
        con = q.connect_all(tmp_path / "results", tmp_path / "data")
        assert {r[0] for r in con.execute("SELECT log FROM runs").fetchall()} == {"toy", "other"}
        assert {r[0] for r in con.execute("SELECT DISTINCT log FROM prefixes").fetchall()} == {"toy", "other"}
        assert {r[0] for r in con.execute("SELECT log FROM logs").fetchall()} == {"toy", "other"}
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
