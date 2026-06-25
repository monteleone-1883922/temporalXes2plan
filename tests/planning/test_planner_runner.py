"""Unit and integration tests for planning.planner_runner."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from planning.planner_runner import (
    PlannerResult,
    get_planners_status,
    run_planner,
)

FIXTURES_PDDL = Path(__file__).parent.parent / "fixtures" / "pddl"


# ---------------------------------------------------------------------------
# get_planners_status
# ---------------------------------------------------------------------------

class TestGetPlannersStatus:
    def test_returns_fast_downward_key(self):
        status = get_planners_status()
        assert "fast_downward" in status

    def test_returns_optic_key(self):
        status = get_planners_status()
        assert "optic" in status

    def test_fast_downward_has_available_bool(self):
        status = get_planners_status()
        assert isinstance(status["fast_downward"]["available"], bool)

    def test_optic_has_available_bool(self):
        status = get_planners_status()
        assert isinstance(status["optic"]["available"], bool)

    def test_fast_downward_has_search_configs(self):
        status = get_planners_status()
        assert "search_configs" in status["fast_downward"]
        assert isinstance(status["fast_downward"]["search_configs"], dict)
        assert len(status["fast_downward"]["search_configs"]) > 0


# ---------------------------------------------------------------------------
# run_planner — unknown planner
# ---------------------------------------------------------------------------

class TestRunPlannerUnknown:
    def test_unknown_planner_returns_error_result(self, tmp_path):
        result = run_planner(tmp_path, "nonexistent_planner", {})
        assert isinstance(result, PlannerResult)
        assert result.success is False
        assert result.solvability == "error"
        assert "nonexistent_planner" in result.message

    def test_unknown_planner_sets_planner_field(self, tmp_path):
        result = run_planner(tmp_path, "mystery_planner", {})
        assert result.planner == "mystery_planner"

    def test_unknown_planner_returns_empty_actions(self, tmp_path):
        result = run_planner(tmp_path, "bad_planner", {})
        assert result.plan_actions == []
        assert result.plan_text is None


# ---------------------------------------------------------------------------
# run_planner — fast_downward (mocked)
# ---------------------------------------------------------------------------

class TestRunPlannerFastDownwardMocked:
    _FAKE_RESULT = {
        "success": True,
        "solvability": "solved",
        "plan_text": "(go-b)\n(go-c)\n",
        "plan_actions": ["go-b", "go-c"],
        "metrics": {"cost": 2},
        "stdout": "Solution found.",
        "stderr": "",
        "message": "Plan found with 2 actions.",
    }

    def test_dispatches_to_fast_downward(self, tmp_path):
        with patch("planning.planner_runner.fast_downward.run", return_value=self._FAKE_RESULT) as mock_run:
            result = run_planner(tmp_path, "fast_downward", {"search": "astar_blind", "timeout": 10})
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["search_key"] == "astar_blind"
        assert call_kwargs.kwargs["timeout"] == 10

    def test_returns_planner_result_on_success(self, tmp_path):
        with patch("planning.planner_runner.fast_downward.run", return_value=self._FAKE_RESULT):
            result = run_planner(tmp_path, "fast_downward", {})
        assert isinstance(result, PlannerResult)
        assert result.success is True
        assert result.plan_actions == ["go-b", "go-c"]
        assert result.planner == "fast_downward"

    def test_passes_default_search_when_not_specified(self, tmp_path):
        with patch("planning.planner_runner.fast_downward.run", return_value=self._FAKE_RESULT) as mock_run:
            run_planner(tmp_path, "fast_downward", {})
        assert mock_run.call_args.kwargs["search_key"] == "astar_lmcut"

    def test_passes_memory_mb(self, tmp_path):
        with patch("planning.planner_runner.fast_downward.run", return_value=self._FAKE_RESULT) as mock_run:
            run_planner(tmp_path, "fast_downward", {"memory_mb": 2048})
        assert mock_run.call_args.kwargs["memory_mb"] == 2048

    def test_passes_log_fn(self, tmp_path):
        logs = []
        log_fn = logs.append  # capture bound method reference once
        with patch("planning.planner_runner.fast_downward.run", return_value=self._FAKE_RESULT) as mock_run:
            run_planner(tmp_path, "fast_downward", {}, log_fn=log_fn)
        assert mock_run.call_args.kwargs["log_fn"] is log_fn


# ---------------------------------------------------------------------------
# run_planner — optic (mocked)
# ---------------------------------------------------------------------------

class TestRunPlannerOpticMocked:
    _FAKE_RESULT = {
        "success": True,
        "solvability": "solved",
        "plan_text": "0.000: (exec_work)  [5.000]\n",
        "plan_actions": ["work"],
        "metrics": {},
        "stdout": "; Solution Found",
        "stderr": "",
        "message": "Plan found with 1 actions.",
    }

    def test_dispatches_to_optic(self, tmp_path):
        with patch("planning.planner_runner.optic.run", return_value=self._FAKE_RESULT) as mock_run:
            run_planner(tmp_path, "optic", {"stop_at_first": False, "ignore_costs": True, "timeout": 45})
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["stop_at_first"] is False
        assert kwargs["ignore_costs"] is True
        assert kwargs["timeout"] == 45

    def test_returns_planner_result(self, tmp_path):
        with patch("planning.planner_runner.optic.run", return_value=self._FAKE_RESULT):
            result = run_planner(tmp_path, "optic", {})
        assert isinstance(result, PlannerResult)
        assert result.planner == "optic"
        assert result.success is True

    def test_passes_default_options(self, tmp_path):
        with patch("planning.planner_runner.optic.run", return_value=self._FAKE_RESULT) as mock_run:
            run_planner(tmp_path, "optic", {})
        kwargs = mock_run.call_args.kwargs
        assert kwargs["stop_at_first"] is True
        assert kwargs["ignore_costs"] is False
        assert kwargs["timeout"] == 60
        assert kwargs["memory_mb"] == 4000


# ---------------------------------------------------------------------------
# Integration tests (require compiled Fast Downward binary)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunPlannerIntegration:
    PDDL = Path(__file__).parent.parent / "fixtures" / "pddl"

    def _pddl_dir(self, tmp_path, domain, problem):
        import shutil
        shutil.copy(self.PDDL / domain, tmp_path / "domain.pddl")
        shutil.copy(self.PDDL / problem, tmp_path / "problem.pddl")
        return tmp_path

    def test_fd_solves_simple_problem(self, tmp_path):
        pddl_dir = self._pddl_dir(tmp_path, "simple_domain.pddl", "simple_problem_solvable.pddl")
        result = run_planner(pddl_dir, "fast_downward", {"search": "astar_lmcut", "timeout": 30})
        assert result.success is True
        assert result.solvability == "solved"
        assert len(result.plan_actions) == 2
        assert result.planner == "fast_downward"

    def test_fd_unsolvable_problem(self, tmp_path):
        pddl_dir = self._pddl_dir(tmp_path, "simple_domain.pddl", "simple_problem_unsolvable.pddl")
        result = run_planner(pddl_dir, "fast_downward", {"search": "astar_lmcut", "timeout": 30})
        assert result.success is False
        assert result.solvability == "unsolvable_structural"

    def test_fd_log_fn_receives_output(self, tmp_path):
        pddl_dir = self._pddl_dir(tmp_path, "simple_domain.pddl", "simple_problem_solvable.pddl")
        logs = []
        run_planner(pddl_dir, "fast_downward", {"timeout": 30}, log_fn=logs.append)
        assert len(logs) > 0
