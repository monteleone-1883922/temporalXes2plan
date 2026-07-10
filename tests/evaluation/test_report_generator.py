"""Unit tests for evaluation.report_generator."""
import json
import threading
from pathlib import Path

import pytest

from evaluation.report_generator import (
    LogResult,
    QueryResult,
    _atomic_write_json,
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
