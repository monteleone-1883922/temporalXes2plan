"""Unit tests for evaluation.report_generator."""
import json
import threading
from pathlib import Path

import pytest

from evaluation.report_generator import (
    LogResult,
    QueryResult,
    _atomic_write_json,
    _build_summary,
    write_log_result,
    write_log_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_result(
    query_id="q1",
    query_type="Q1",
    trace_id="case_1",
    prefix_ratio=0.5,
    n_prefix_events=5,
    attempts=1,
    solvability="solved",
    planner_duration_s=2.5,
    metrics=None,
    validation=None,
    is_replayable=True,
) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        query_type=query_type,
        trace_id=trace_id,
        prefix_ratio=prefix_ratio,
        n_prefix_events=n_prefix_events,
        attempts=attempts,
        solvability=solvability,
        planner_duration_s=planner_duration_s,
        metrics=metrics or {"solved": True, "plan_time_s": 10.0, "weighted_objective": 10.01},
        validation=validation,
        is_replayable=is_replayable,
    )


def _make_log_result(
    log_id="log_1",
    log_name="Test Log",
    log_fmt="xes",
    n_train=80,
    n_test=20,
    n_activities=10,
    pipeline_ok=True,
    pipeline_error=None,
    queries=None,
    used_optimizer=False,
    search_summary=None,
) -> LogResult:
    return LogResult(
        log_id=log_id,
        log_name=log_name,
        log_fmt=log_fmt,
        n_train_cases=n_train,
        n_test_cases=n_test,
        n_activities=n_activities,
        pipeline_ok=pipeline_ok,
        pipeline_error=pipeline_error,
        queries=queries or [_make_query_result()],
        used_optimizer=used_optimizer,
        search_summary=search_summary,
    )


# ---------------------------------------------------------------------------
# write_log_result
# ---------------------------------------------------------------------------

class TestWriteLogResult:
    def test_result_json_written_to_correct_path(self, tmp_path):
        lr = _make_log_result(log_id="42")
        write_log_result("42", lr, tmp_path)
        expected = tmp_path / "42" / "result.json"
        assert expected.exists()

    def test_result_json_is_valid_json(self, tmp_path):
        lr = _make_log_result(log_id="1")
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        assert isinstance(data, dict)

    def test_result_json_matches_schema(self, tmp_path):
        lr = _make_log_result(log_id="1")
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        for key in ("log_id", "log_name", "log_fmt", "n_train_cases",
                    "n_test_cases", "n_activities", "pipeline_ok",
                    "pipeline_error", "queries", "used_optimizer", "search_summary"):
            assert key in data, f"Missing key: {key}"

    def test_used_optimizer_and_search_summary_serialized(self, tmp_path):
        lr = _make_log_result(
            log_id="1", used_optimizer=True,
            search_summary={"n_trials": 30, "best_score": 4.2, "best_trial_number": 3},
        )
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        assert data["used_optimizer"] is True
        assert data["search_summary"] == {"n_trials": 30, "best_score": 4.2, "best_trial_number": 3}

    def test_used_optimizer_defaults_false_and_search_summary_none(self, tmp_path):
        lr = _make_log_result(log_id="1")
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        assert data["used_optimizer"] is False
        assert data["search_summary"] is None

    def test_result_json_queries_list(self, tmp_path):
        lr = _make_log_result(log_id="1")
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        assert isinstance(data["queries"], list)

    def test_result_json_query_schema(self, tmp_path):
        lr = _make_log_result(log_id="1")
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        q = data["queries"][0]
        for key in ("query_id", "query_type", "trace_id", "prefix_ratio",
                    "n_prefix_events", "attempts", "solvability",
                    "planner_duration_s", "metrics", "validation"):
            assert key in q, f"Missing query key: {key}"

    def test_result_json_pipeline_error_preserved(self, tmp_path):
        lr = _make_log_result(log_id="1", pipeline_ok=False, pipeline_error="parse failed", queries=[])
        write_log_result("1", lr, tmp_path)
        data = json.loads((tmp_path / "1" / "result.json").read_text())
        assert data["pipeline_ok"] is False
        assert data["pipeline_error"] == "parse failed"

    def test_creates_parent_directories(self, tmp_path):
        lr = _make_log_result(log_id="nested")
        write_log_result("nested", lr, tmp_path / "deep" / "path")
        assert (tmp_path / "deep" / "path" / "nested" / "result.json").exists()

    def test_returns_path_to_written_file(self, tmp_path):
        lr = _make_log_result(log_id="1")
        dest = write_log_result("1", lr, tmp_path)
        assert dest == tmp_path / "1" / "result.json"


# ---------------------------------------------------------------------------
# write_log_summary
# ---------------------------------------------------------------------------

class TestWriteLogSummary:
    def test_summary_json_written_in_log_subdir(self, tmp_path):
        result = _make_log_result(log_id="1")
        dest = write_log_summary(result, tmp_path)
        assert dest == tmp_path / "1" / "summary_result.json"
        assert dest.exists()

    def test_summary_json_top_level_schema(self, tmp_path):
        result = _make_log_result()
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        for key in ("generated_at", "n_logs", "n_logs_ok", "n_queries_total",
                    "cost_weight", "q1", "q2", "q3", "per_log"):
            assert key in data, f"Missing summary key: {key}"

    def test_summary_json_n_logs_is_always_one(self, tmp_path):
        result = _make_log_result(log_id="1")
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["n_logs"] == 1

    def test_summary_json_n_logs_ok_zero_when_pipeline_failed(self, tmp_path):
        result = _make_log_result(pipeline_ok=False, pipeline_error="err", queries=[])
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["n_logs_ok"] == 0

    def test_summary_overwritten_on_second_call(self, tmp_path):
        result = _make_log_result(log_id="1", pipeline_ok=True)
        write_log_summary(result, tmp_path)
        result2 = _make_log_result(log_id="1", pipeline_ok=False, pipeline_error="err", queries=[])
        dest = write_log_summary(result2, tmp_path)
        data = json.loads(dest.read_text())
        assert data["n_logs_ok"] == 0

    def test_summary_cost_weight_recorded(self, tmp_path):
        result = _make_log_result()
        dest = write_log_summary(result, tmp_path, cost_weight=0.05)
        data = json.loads(dest.read_text())
        assert data["cost_weight"] == pytest.approx(0.05)

    def test_summary_per_log_has_log_id(self, tmp_path):
        result = _make_log_result(log_id="xyz")
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["per_log"][0]["log_id"] == "xyz"


# ---------------------------------------------------------------------------
# Planner failure-mode ratios (timeout / out_of_memory) — see
# planning/fast_downward.py::_classify().
# ---------------------------------------------------------------------------

class TestTimeoutAndOutOfMemoryStats:
    def _log_with_solvabilities(self, *solvabilities, log_id="log_1"):
        queries = [
            _make_query_result(query_id=f"q{i}", solvability=s)
            for i, s in enumerate(solvabilities)
        ]
        return _make_log_result(log_id=log_id, queries=queries)

    def test_top_level_keys_present(self, tmp_path):
        result = self._log_with_solvabilities("solved")
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert "timeout_ratio_mean" in data
        assert "out_of_memory_ratio_mean" in data

    def test_timeout_ratio_computed_across_all_query_types(self, tmp_path):
        result = self._log_with_solvabilities("solved", "timeout", "timeout", "unsolvable_structural")
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["timeout_ratio_mean"] == pytest.approx(0.5)
        assert data["per_log"][0]["pct_timeout"] == pytest.approx(0.5)

    def test_out_of_memory_ratio_computed(self, tmp_path):
        result = self._log_with_solvabilities("out_of_memory", "solved", "solved", "solved")
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["out_of_memory_ratio_mean"] == pytest.approx(0.25)
        assert data["per_log"][0]["pct_out_of_memory"] == pytest.approx(0.25)

    def test_zero_when_no_timeout_or_oom_present(self, tmp_path):
        result = self._log_with_solvabilities("solved", "unsolvable_structural")
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["timeout_ratio_mean"] == 0.0
        assert data["out_of_memory_ratio_mean"] == 0.0

    def test_none_when_no_queries(self, tmp_path):
        # _make_log_result(queries=[]) falls back to its default (non-empty)
        # query list — `[] or [...]` — so LogResult is built directly here to
        # get a genuinely empty queries list.
        result = LogResult(
            log_id="log_1", log_name="Test Log", log_fmt="xes",
            n_train_cases=80, n_test_cases=0, n_activities=10,
            pipeline_ok=True, pipeline_error=None, queries=[],
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["timeout_ratio_mean"] is None
        assert data["out_of_memory_ratio_mean"] is None
        assert data["per_log"][0]["pct_timeout"] is None
        assert data["per_log"][0]["pct_out_of_memory"] is None


# ---------------------------------------------------------------------------
# "Unsatisfiable by construction" — Q2 (compute_min_time_to_end precheck,
# solvability="skipped_unsatisfiable_by_construction") and Q3
# (is_q3_reachable precheck, solvability="skipped_unreachable", pre-existing
# and NOT renamed — see claude_plans/time_aware_evaluation_plan.md §7.1).
# Same metric name in the report for both, but each with its OWN
# query-type-specific denominator (unlike pct_timeout/pct_out_of_memory,
# which are deliberately cross-query-type).
# ---------------------------------------------------------------------------

class TestUnsatisfiableByConstructionStats:
    def _log_with_queries(self, *type_solvability_pairs, log_id="log_1"):
        queries = [
            _make_query_result(query_id=f"q{i}", query_type=qtype, solvability=s)
            for i, (qtype, s) in enumerate(type_solvability_pairs)
        ]
        return _make_log_result(log_id=log_id, queries=queries)

    def test_top_level_keys_present(self, tmp_path):
        result = self._log_with_queries(("Q2", "solved"), ("Q3", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert "pct_unsatisfiable_by_construction_mean" in data["q2"]
        assert "pct_unsatisfiable_by_construction_mean" in data["q3"]

    def test_q2_ratio_computed_from_its_own_solvability_string(self, tmp_path):
        result = self._log_with_queries(
            ("Q2", "solved"),
            ("Q2", "skipped_unsatisfiable_by_construction"),
            ("Q2", "skipped_unsatisfiable_by_construction"),
            ("Q2", "unsolvable_structural"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_unsatisfiable_by_construction_mean"] == pytest.approx(0.5)
        assert data["per_log"][0]["q2_pct_unsatisfiable_by_construction"] == pytest.approx(0.5)

    def test_q3_ratio_computed_from_its_own_solvability_string(self, tmp_path):
        result = self._log_with_queries(
            ("Q3", "skipped_unreachable"),
            ("Q3", "solved"),
            ("Q3", "solved"),
            ("Q3", "solved"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["pct_unsatisfiable_by_construction_mean"] == pytest.approx(0.25)
        assert data["per_log"][0]["q3_pct_unsatisfiable_by_construction"] == pytest.approx(0.25)

    def test_q2_and_q3_ratios_are_independent(self, tmp_path):
        # Q2's own skip string appearing in Q2 queries must not leak into
        # Q3's ratio (and vice versa) -- each is computed only from its own
        # query_type's subset, verified here with different ratios for each.
        result = self._log_with_queries(
            ("Q2", "skipped_unsatisfiable_by_construction"),
            ("Q2", "solved"),
            ("Q3", "skipped_unreachable"),
            ("Q3", "skipped_unreachable"),
            ("Q3", "solved"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_unsatisfiable_by_construction_mean"] == pytest.approx(0.5)
        assert data["q3"]["pct_unsatisfiable_by_construction_mean"] == pytest.approx(2 / 3)

    def test_zero_when_no_skips_present(self, tmp_path):
        result = self._log_with_queries(("Q2", "solved"), ("Q3", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_unsatisfiable_by_construction_mean"] == 0.0
        assert data["q3"]["pct_unsatisfiable_by_construction_mean"] == 0.0

    def test_none_when_query_type_absent(self, tmp_path):
        # Log has only Q1 queries -- Q2/Q3 ratios must be None, not 0.0 or
        # an error (mirrors _solvability_ratio's own empty-list contract).
        result = self._log_with_queries(("Q1", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_unsatisfiable_by_construction_mean"] is None
        assert data["q3"]["pct_unsatisfiable_by_construction_mean"] is None
        assert data["per_log"][0]["q2_pct_unsatisfiable_by_construction"] is None
        assert data["per_log"][0]["q3_pct_unsatisfiable_by_construction"] is None


class TestConstructionSkipsCountAsFailures:
    """solved_ratio_mean must always be computed over every test-case trace
    for a query type (denominator = all queries of that type for the log),
    no matter why a given trace didn't reach the planner -- a structural
    precheck skip (Q2: "skipped_unsatisfiable_by_construction", Q3:
    "skipped_unreachable") or a trace for which the query was never even
    built (Q2: "skipped_no_budget", Q3: "skipped_no_attributes") must all
    count as a non-"solved" outcome, exactly like Q1 (which has no skip path
    at all). The reason for the resulting lower solved ratio is then
    explained by the query type's own pct_*_mean stats, not hidden by
    excluding it from the denominator."""

    def _log_with_queries(self, *type_solvability_solved, log_id="log_1"):
        # Builds QueryResult directly (not via _make_query_result) so a truly
        # empty metrics dict for skipped queries survives -- `metrics or
        # {...default...}` in _make_query_result would otherwise treat an
        # empty dict as falsy and silently substitute its non-empty default.
        queries = [
            QueryResult(
                query_id=f"q{i}", query_type=qtype, trace_id=f"case_{i}",
                prefix_ratio=0.5, n_prefix_events=5, attempts=(0 if solved is None else 1),
                solvability=solvability, planner_duration_s=None,
                metrics=({} if solved is None else {"solved": solved}),
                validation=None,
            )
            for i, (qtype, solvability, solved) in enumerate(type_solvability_solved)
        ]
        return _make_log_result(log_id=log_id, queries=queries)

    def test_q3_construction_skips_lower_the_solved_ratio(self, tmp_path):
        # Mirrors log 55's shape: 4 solved, 1 skipped_unreachable -- must be
        # 4/5 = 0.8, not 4/4 = 1.0 (which is what excluding the skip gives).
        result = self._log_with_queries(
            ("Q3", "solved", True),
            ("Q3", "solved", True),
            ("Q3", "solved", True),
            ("Q3", "solved", True),
            ("Q3", "skipped_unreachable", None),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["solved_ratio_mean"] == pytest.approx(0.8)

    def test_q2_construction_skips_lower_the_solved_ratio(self, tmp_path):
        result = self._log_with_queries(
            ("Q2", "solved", True),
            ("Q2", "skipped_unsatisfiable_by_construction", None),
            ("Q2", "skipped_unsatisfiable_by_construction", None),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["solved_ratio_mean"] == pytest.approx(1 / 3)
        assert data["per_log"][0]["q2_solved_ratio"] == pytest.approx(1 / 3)

    def test_data_availability_skips_also_count_as_failures(self, tmp_path):
        # skipped_no_attributes (Q3) and skipped_no_budget (Q2) are a
        # data-availability gap rather than a proven-unsolvable-by-
        # construction trace, but the denominator must still cover every
        # trace -- so these also count as a non-"solved" outcome.
        result = self._log_with_queries(
            ("Q3", "solved", True),
            ("Q3", "skipped_no_attributes", None),
            ("Q2", "solved", True),
            ("Q2", "skipped_no_budget", None),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["solved_ratio_mean"] == pytest.approx(0.5)
        assert data["q2"]["solved_ratio_mean"] == pytest.approx(0.5)

    def test_q1_unaffected_no_construction_skip_category(self, tmp_path):
        result = self._log_with_queries(
            ("Q1", "solved", True),
            ("Q1", "unsolvable_resource", False),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q1"]["solved_ratio_mean"] == pytest.approx(0.5)

    def test_all_construction_skipped_gives_zero_not_none(self, tmp_path):
        result = self._log_with_queries(
            ("Q3", "skipped_unreachable", None),
            ("Q3", "skipped_unreachable", None),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["solved_ratio_mean"] == 0.0


class TestQ2SkippedNoBudgetStats:
    """Q2-only skip reason (no timestamp data to compute a time budget) --
    like skipped_no_attributes for Q3, tracked separately so the effect on
    Q2's solved_ratio_mean is explained rather than hidden."""

    def _log_with_queries(self, *type_solvability_pairs, log_id="log_1"):
        queries = [
            _make_query_result(query_id=f"q{i}", query_type=qtype, solvability=s)
            for i, (qtype, s) in enumerate(type_solvability_pairs)
        ]
        return _make_log_result(log_id=log_id, queries=queries)

    def test_top_level_key_present(self, tmp_path):
        result = self._log_with_queries(("Q2", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert "pct_skipped_no_budget_mean" in data["q2"]

    def test_ratio_computed_from_its_own_solvability_string(self, tmp_path):
        result = self._log_with_queries(
            ("Q2", "solved"),
            ("Q2", "skipped_no_budget"),
            ("Q2", "skipped_no_budget"),
            ("Q2", "unsolvable_structural"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_skipped_no_budget_mean"] == pytest.approx(0.5)
        assert data["per_log"][0]["q2_pct_skipped_no_budget"] == pytest.approx(0.5)

    def test_independent_from_unsatisfiable_by_construction(self, tmp_path):
        result = self._log_with_queries(
            ("Q2", "skipped_no_budget"),
            ("Q2", "skipped_unsatisfiable_by_construction"),
            ("Q2", "solved"),
            ("Q2", "solved"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_skipped_no_budget_mean"] == pytest.approx(0.25)
        assert data["q2"]["pct_unsatisfiable_by_construction_mean"] == pytest.approx(0.25)

    def test_none_when_no_q2_queries(self, tmp_path):
        result = self._log_with_queries(("Q1", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q2"]["pct_skipped_no_budget_mean"] is None
        assert data["per_log"][0]["q2_pct_skipped_no_budget"] is None


class TestQ3SkippedNoAttributesStats:
    """Q3-only skip reason (final event has no discretized attributes) --
    like skipped_unreachable, this must be visible separately from
    solved_ratio_mean so the denominator effect on Q3's apparent solved
    ratio (vs. Q1, which never skips) isn't hidden."""

    def _log_with_queries(self, *type_solvability_pairs, log_id="log_1"):
        queries = [
            _make_query_result(query_id=f"q{i}", query_type=qtype, solvability=s)
            for i, (qtype, s) in enumerate(type_solvability_pairs)
        ]
        return _make_log_result(log_id=log_id, queries=queries)

    def test_top_level_key_present(self, tmp_path):
        result = self._log_with_queries(("Q3", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert "pct_skipped_no_attributes_mean" in data["q3"]

    def test_ratio_computed_from_its_own_solvability_string(self, tmp_path):
        result = self._log_with_queries(
            ("Q3", "solved"),
            ("Q3", "skipped_no_attributes"),
            ("Q3", "skipped_no_attributes"),
            ("Q3", "unsolvable_structural"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["pct_skipped_no_attributes_mean"] == pytest.approx(0.5)
        assert data["per_log"][0]["q3_pct_skipped_no_attributes"] == pytest.approx(0.5)

    def test_independent_from_skipped_unreachable(self, tmp_path):
        # Both skip reasons can coexist in the same log -- their ratios must
        # not be conflated.
        result = self._log_with_queries(
            ("Q3", "skipped_no_attributes"),
            ("Q3", "skipped_unreachable"),
            ("Q3", "solved"),
            ("Q3", "solved"),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["pct_skipped_no_attributes_mean"] == pytest.approx(0.25)
        assert data["q3"]["pct_unsatisfiable_by_construction_mean"] == pytest.approx(0.25)

    def test_zero_when_no_skips_present(self, tmp_path):
        result = self._log_with_queries(("Q3", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["pct_skipped_no_attributes_mean"] == 0.0

    def test_none_when_no_q3_queries(self, tmp_path):
        result = self._log_with_queries(("Q1", "solved"))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q3"]["pct_skipped_no_attributes_mean"] is None
        assert data["per_log"][0]["q3_pct_skipped_no_attributes"] is None


class TestNotReplayableTraceCounts:
    """Per-query-type count of non-replayable traces -- is_replayable is
    query-type-aware (each of Q1/Q2/Q3 has its own replayability semantics),
    so the count must be computed per type, not shared across types."""

    def _log_with_queries(self, *type_trace_replayable, log_id="log_1"):
        queries = [
            _make_query_result(
                query_id=f"q{i}", query_type=qtype, trace_id=trace_id, is_replayable=rep,
            )
            for i, (qtype, trace_id, rep) in enumerate(type_trace_replayable)
        ]
        return _make_log_result(log_id=log_id, queries=queries)

    def test_per_log_count_for_each_query_type(self, tmp_path):
        result = self._log_with_queries(
            ("Q1", "case_1", True),
            ("Q1", "case_2", False),
            ("Q2", "case_1", False),
            ("Q2", "case_2", False),
            ("Q3", "case_1", True),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        row = data["per_log"][0]
        assert row["q1_n_traces_not_replayable"] == 1
        assert row["q2_n_traces_not_replayable"] == 2
        assert row["q3_n_traces_not_replayable"] == 0

    def test_counts_unique_traces_not_queries(self, tmp_path):
        # Two Q1 queries (different prefix ratios) for the same trace must
        # count as ONE not-replayable trace, not two.
        result = self._log_with_queries(
            ("Q1", "case_1", False),
            ("Q1", "case_1", False),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["per_log"][0]["q1_n_traces_not_replayable"] == 1

    def test_query_types_are_independent(self, tmp_path):
        # Same trace_id, replayable for Q1 but not for Q2 -- reflects that
        # is_replayable is query-type-specific, not a trace-level property.
        result = self._log_with_queries(
            ("Q1", "case_1", True),
            ("Q2", "case_1", False),
        )
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["per_log"][0]["q1_n_traces_not_replayable"] == 0
        assert data["per_log"][0]["q2_n_traces_not_replayable"] == 1

    def test_aggregate_total_sums_across_logs(self):
        log_1 = self._log_with_queries(
            ("Q1", "case_1", False), ("Q1", "case_2", False), log_id="log_1",
        )
        log_2 = self._log_with_queries(
            ("Q1", "case_1", False), log_id="log_2",
        )
        summary = _build_summary([log_1, log_2], cost_weight=0.001)
        assert summary["q1"]["n_traces_not_replayable_total"] == 3

    def test_zero_when_all_replayable(self, tmp_path):
        result = self._log_with_queries(("Q1", "case_1", True), ("Q2", "case_1", True))
        dest = write_log_summary(result, tmp_path)
        data = json.loads(dest.read_text())
        assert data["q1"]["n_traces_not_replayable_total"] == 0
        assert data["q2"]["n_traces_not_replayable_total"] == 0


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_atomic_write_produces_valid_json(self, tmp_path):
        dest = tmp_path / "out.json"
        _atomic_write_json(dest, {"key": "value"})
        assert json.loads(dest.read_text()) == {"key": "value"}

    def test_no_tmp_file_left_after_write(self, tmp_path):
        dest = tmp_path / "out.json"
        _atomic_write_json(dest, {})
        assert not list(tmp_path.glob("*.tmp"))

    def test_atomic_write_no_partial_file_on_rename(self, tmp_path):
        """Simulate concurrent writes: final file must be valid JSON."""
        dest = tmp_path / "summary.json"
        errors = []

        def write_n(n):
            try:
                _atomic_write_json(dest, {"n": n})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_n, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # File must be readable and valid JSON after all threads complete.
        data = json.loads(dest.read_text())
        assert "n" in data

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "out.json"
        _atomic_write_json(dest, {"v": 1})
        _atomic_write_json(dest, {"v": 2})
        assert json.loads(dest.read_text()) == {"v": 2}
