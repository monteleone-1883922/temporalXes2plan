"""Unit tests for evaluation.planner_runner_with_retry."""
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from evaluation.planner_runner_with_retry import (
    RetryConfig,
    _write_failure_pointer,
    run_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(success=True, solvability="solved", duration_s=1.0, cost=0.0,
                 error=None, stdout="", stderr=""):
    r = MagicMock()
    r.success = success
    r.solvability = solvability
    r.duration_s = duration_s
    r.cost = cost
    r.error = error
    r.raw_stdout = stdout
    r.raw_stderr = stderr
    r.plan_steps = ["act_a"] if success else []
    return r


def _make_api(*results):
    api = MagicMock()
    api.run_planner.side_effect = list(results)
    return api


def _cfg(**kwargs) -> RetryConfig:
    defaults = dict(max_attempts=3, base_delay_s=0.0, timeout_s=30, memory_mb=2000)
    defaults.update(kwargs)
    return RetryConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic success / retry behaviour
# ---------------------------------------------------------------------------

class TestRunWithRetryBasic:
    def test_returns_on_first_success(self, tmp_path):
        api = _make_api(_make_result(success=True))
        result, attempts = run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        assert result.success is True
        assert attempts == 1
        assert api.run_planner.call_count == 1

    def test_retries_on_error_and_succeeds(self, tmp_path):
        api = _make_api(
            _make_result(success=False, solvability="error"),
            _make_result(success=False, solvability="error"),
            _make_result(success=True),
        )
        result, attempts = run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        assert result.success is True
        assert attempts == 3
        assert api.run_planner.call_count == 3

    def test_returns_last_result_after_exhaustion(self, tmp_path):
        api = _make_api(*[_make_result(success=False, solvability="error")] * 3)
        result, attempts = run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        assert result.success is False
        assert attempts == 3
        assert api.run_planner.call_count == 3


# ---------------------------------------------------------------------------
# Non-retryable solvability values
# ---------------------------------------------------------------------------

class TestNonRetryable:
    @pytest.mark.parametrize("solvability", [
        "unsolvable_structural",
        "unsolvable_resource",
        "timeout",
    ])
    def test_no_retry_on_non_retryable(self, tmp_path, solvability):
        api = _make_api(_make_result(success=False, solvability=solvability))
        result, attempts = run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        assert attempts == 1
        assert api.run_planner.call_count == 1
        assert result.solvability == solvability

    def test_no_retry_on_unsolvable_structural(self, tmp_path):
        api = _make_api(_make_result(success=False, solvability="unsolvable_structural"))
        _, attempts = run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        assert attempts == 1

    def test_no_retry_on_timeout(self, tmp_path):
        api = _make_api(_make_result(success=False, solvability="timeout"))
        _, attempts = run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        assert attempts == 1


# ---------------------------------------------------------------------------
# Planner call arguments
# ---------------------------------------------------------------------------

class TestPlannerCallArgs:
    def test_planner_called_with_optic(self, tmp_path):
        api = _make_api(_make_result(success=True))
        run_with_retry("domain", "problem", api, _cfg(timeout_s=45, memory_mb=3000), tmp_path / "f", "q1")
        api.run_planner.assert_called_once_with(
            "domain", "problem", planner="optic", timeout=45, memory_mb=3000,
        )

    def test_domain_and_problem_forwarded(self, tmp_path):
        api = _make_api(_make_result(success=True))
        run_with_retry("MY_DOMAIN", "MY_PROBLEM", api, _cfg(), tmp_path / "f", "q1")
        call_args = api.run_planner.call_args
        assert call_args.args[0] == "MY_DOMAIN"
        assert call_args.args[1] == "MY_PROBLEM"


# ---------------------------------------------------------------------------
# Linear backoff
# ---------------------------------------------------------------------------

class TestLinearBackoff:
    def test_linear_backoff_delays(self, tmp_path):
        api = _make_api(
            _make_result(success=False, solvability="error"),
            _make_result(success=False, solvability="error"),
            _make_result(success=True),
        )
        with patch("evaluation.planner_runner_with_retry.time.sleep") as mock_sleep:
            run_with_retry("domain", "problem", api, _cfg(base_delay_s=2.0), tmp_path / "f", "q1")
        # attempt 1 → sleep(1 * 2.0), attempt 2 → sleep(2 * 2.0)
        assert mock_sleep.call_count == 2
        calls = mock_sleep.call_args_list
        assert calls[0] == call(2.0)
        assert calls[1] == call(4.0)

    def test_no_sleep_after_final_attempt(self, tmp_path):
        api = _make_api(*[_make_result(success=False, solvability="error")] * 3)
        with patch("evaluation.planner_runner_with_retry.time.sleep") as mock_sleep:
            run_with_retry("domain", "problem", api, _cfg(max_attempts=3, base_delay_s=1.0), tmp_path / "f", "q1")
        # sleep only between attempts, not after the last one
        assert mock_sleep.call_count == 2

    def test_no_sleep_on_non_retryable(self, tmp_path):
        api = _make_api(_make_result(success=False, solvability="timeout"))
        with patch("evaluation.planner_runner_with_retry.time.sleep") as mock_sleep:
            run_with_retry("domain", "problem", api, _cfg(), tmp_path / "f", "q1")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Failure pointer
# ---------------------------------------------------------------------------

class TestFailurePointer:
    def test_failure_pointer_written_after_exhaustion(self, tmp_path):
        failures = tmp_path / "failures"
        api = _make_api(*[_make_result(success=False, solvability="error")] * 3)
        run_with_retry("domain", "problem", api, _cfg(), failures, "qid_1")
        assert (failures / "qid_1.json").exists()

    def test_no_failure_pointer_on_success(self, tmp_path):
        failures = tmp_path / "failures"
        api = _make_api(_make_result(success=True))
        run_with_retry("domain", "problem", api, _cfg(), failures, "qid_1")
        assert not (failures / "qid_1.json").exists()

    def test_failure_pointer_written_on_non_retryable(self, tmp_path):
        failures = tmp_path / "failures"
        api = _make_api(_make_result(success=False, solvability="unsolvable_structural"))
        run_with_retry("domain", "problem", api, _cfg(), failures, "qid_1")
        assert (failures / "qid_1.json").exists()

    def test_failure_pointer_json_schema(self, tmp_path):
        failures = tmp_path / "failures"
        api = _make_api(*[_make_result(success=False, solvability="error", error="boom")] * 2)
        run_with_retry("DOMAIN", "PROBLEM", api, _cfg(max_attempts=2), failures, "qid_1")
        data = json.loads((failures / "qid_1.json").read_text())
        for key in ("query_id", "last_solvability", "last_error",
                    "last_stdout", "last_stderr", "timestamp"):
            assert key in data, f"Missing key: {key}"
        assert data["query_id"] == "qid_1"
        assert data["last_solvability"] == "error"
        assert data["last_error"] == "boom"

    def test_failure_pointer_creates_directory(self, tmp_path):
        failures = tmp_path / "deep" / "nested" / "failures"
        api = _make_api(*[_make_result(success=False, solvability="error")] * 1)
        run_with_retry("d", "p", api, _cfg(max_attempts=1), failures, "qid")
        assert failures.exists()


# ---------------------------------------------------------------------------
# _write_failure_pointer directly
# ---------------------------------------------------------------------------

class TestWriteFailurePointer:
    def test_writes_valid_json(self, tmp_path):
        result = _make_result(success=False, solvability="error", error="oops", stdout="out", stderr="err")
        _write_failure_pointer(tmp_path, "q42", "PROB", result)
        data = json.loads((tmp_path / "q42.json").read_text())
        assert data["query_id"] == "q42"
        assert data["last_solvability"] == "error"

    def test_timestamp_is_iso(self, tmp_path):
        result = _make_result(success=False, solvability="error")
        _write_failure_pointer(tmp_path, "q1", "p", result)
        data = json.loads((tmp_path / "q1.json").read_text())
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"])  # raises if invalid
