"""planning.optic.run_timed: grounding and planning times read from OPTIC's output.

Most tests run a FAKE planner (a small Python script that prints OPTIC's lines
with controlled pauses), so the timestamps can be checked without the real
binary. One class at the end runs the real OPTIC on the tiny fixture problems.
"""
import shutil
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import planning.optic as optic_module
from planning.optic import run_timed

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pddl"

SOLVED_BODY = """
    import time
    print("Number of literals: 3", flush=True)
    time.sleep(0.4)
    print("Initial heuristic = 2.000, admissible cost estimate 0.000", flush=True)
    time.sleep(0.4)
    # the marker shares its line with the search progress, as in the real planner
    print("b (1.000 | 0.000)(G);;;; Solution Found", flush=True)
    print("; States evaluated: 3", flush=True)
    print("; Cost: 0.001", flush=True)
    print("; Time 0.15", flush=True)
    print("0.000: (go-b)  [0.001]", flush=True)
    print("0.001: (go-c)  [0.001]", flush=True)
"""


def _pddl_dir(tmp_path: Path) -> Path:
    (tmp_path / "domain.pddl").write_text("(define (domain d))")
    (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))")
    return tmp_path


def _fake(tmp_path: Path, body: str):
    """optic_cmd that runs `body` as a Python script instead of OPTIC."""
    script = tmp_path / "fake_optic.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, "-u", str(script)]


class TestTimings:
    def test_grounding_and_planning_times_are_read_from_the_output(self, tmp_path):
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, SOLVED_BODY))
        t = result["timings"]
        assert result["success"] and result["solvability"] == "solved"
        # ~0.4 s before the heuristic line, ~0.4 s more before the plan (wide margins: CI machines are slow)
        assert 0.35 <= t["grounding_s"] <= 1.5
        assert 0.35 <= t["planning_s"] <= 1.5
        assert t["first_plan_s"] == pytest.approx(t["grounding_s"] + t["planning_s"])
        assert t["wall_s"] >= t["first_plan_s"]

    def test_plan_and_metrics_are_parsed(self, tmp_path):
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, SOLVED_BODY))
        assert result["plan_actions"] == ["go-b", "go-c"]
        assert result["metrics"]["expanded_nodes"] == 3
        assert result["timings"]["cpu_s"] == pytest.approx(0.15)
        assert result["metrics"]["duration"] == pytest.approx(0.002)

    def test_stop_at_first_passes_the_n_flag(self, tmp_path):
        body = 'import sys; print("ARGS", sys.argv[1:], flush=True)'
        words = _fake(tmp_path, body)
        with_flag = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=words)["stdout"]
        without = run_timed(tmp_path, timeout=30, stop_at_first=False, optic_cmd=words)["stdout"]
        assert "'-N'" in with_flag and "'-N'" not in without


# What OPTIC prints without -N (copied from a real run): an improving plan with its
# "; Plan found with metric" header, then the final block that repeats the best plan.
IMPROVING_BODY = """
    import time
    print("Initial heuristic = 2.000", flush=True)
    time.sleep(0.3)
    print("b (1.000 | 1.000)(G)", flush=True)
    print("; Plan found with metric 9.000", flush=True)
    print("; Time 0.10", flush=True)
    print("0.000: (slow)  [9.000]", flush=True)
    time.sleep(0.3)
    print("; Plan found with metric 3.000", flush=True)
    print("; Time 0.20", flush=True)
    print("0.000: (b)  [1.000]", flush=True)
    print("1.001: (tau_x)  [0.001]", flush=True)
    print("1.002: (c)  [1.000]", flush=True)
    {ending}
"""
FINAL_BLOCK = (
    'print("Problem Unsolvable", flush=True); print(";;;; Solution Found", flush=True); '
    'print("; Cost: 3.0", flush=True); print("0.000: (b)  [1.000]", flush=True); '
    'print("1.001: (tau_x)  [0.001]", flush=True); print("1.002: (c)  [1.000]", flush=True)'
)


class TestEveryPlanOpticPrints:
    def test_improving_plans_and_the_final_block_are_one_list(self, tmp_path):
        body = IMPROVING_BODY.format(ending=FINAL_BLOCK)
        result = run_timed(_pddl_dir(tmp_path), timeout=30, stop_at_first=False,
                           optic_cmd=_fake(tmp_path, body))
        plans = result["plans"]
        assert [p["metric"] for p in plans] == [9.0, 3.0]  # the final block is not a third plan
        assert [p["final"] for p in plans] == [False, True]
        assert plans[0]["steps"] == [(0.0, "slow", 9.0)]
        assert plans[1]["steps"] == [(0.0, "b", 1.0), (1.001, "tau_x", 0.001), (1.002, "c", 1.0)]
        assert plans[1]["makespan"] == pytest.approx(2.002)
        assert 0 < plans[0]["found_s"] < plans[1]["found_s"]
        assert result["search_completed"] is True and result["solvability"] == "solved"
        assert result["plan_actions"] == ["b", "c"]  # the best plan, silent steps left out
        # the first plan is the first improving one, not the final block
        assert result["timings"]["first_plan_s"] == pytest.approx(plans[0]["found_s"], abs=0.05)

    def test_killed_after_improving_plans_returns_the_best_one_as_not_completed(self, tmp_path):
        body = IMPROVING_BODY.format(ending="time.sleep(60)")
        result = run_timed(_pddl_dir(tmp_path), timeout=3, stop_at_first=False,
                           optic_cmd=_fake(tmp_path, body))
        assert result["solvability"] == "solved" and result["success"]
        assert result["search_completed"] is False
        assert [p["metric"] for p in result["plans"]] == [9.0, 3.0]
        assert not any(p["final"] for p in result["plans"])
        assert result["plan_actions"] == ["b", "c"]
        assert result["metrics"]["duration"] == pytest.approx(2.002)

    def test_killed_before_any_plan_stays_a_timeout(self, tmp_path):
        body = 'import time; print("Initial heuristic = 2.0", flush=True); time.sleep(60)'
        result = run_timed(_pddl_dir(tmp_path), timeout=2, stop_at_first=False,
                           optic_cmd=_fake(tmp_path, body))
        assert result["solvability"] == "timeout"
        assert result["plans"] == [] and result["search_completed"] is False

    def test_with_n_there_is_one_final_plan(self, tmp_path):
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, SOLVED_BODY))
        assert len(result["plans"]) == 1 and result["plans"][0]["final"]
        assert result["plans"][0]["steps"] == [(0.0, "go-b", 0.001), (0.001, "go-c", 0.001)]


class TestEmptyPlan:
    """A problem whose initial state already satisfies the goal gets ";;;; Solution Found" and no
    plan line (OPTIC: "The empty plan is optimal"). It is a solution, not a failure (it used to be
    classified as unsolvable_resource, 34 times on Helpdesk)."""

    BODY = """
        print("Initial heuristic = 0.000", flush=True)
        print("(G); The empty plan is optimal", flush=True)
        print(";;;; Solution Found", flush=True)
        print("; States evaluated: 1", flush=True)
        print("; Cost: 0.000", flush=True)
        print("; Time 0.01", flush=True)
    """

    def test_the_empty_plan_is_solved_with_no_actions(self, tmp_path):
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, self.BODY))
        assert result["solvability"] == "solved" and result["success"] is True
        assert result["plan_text"] == "" and result["plan_actions"] == []
        assert result["plans"][0]["final"] and result["plans"][0]["steps"] == []
        assert result["search_completed"] is True

    def test_without_the_solution_marker_no_plan_lines_still_fails(self, tmp_path):
        body = 'print("Initial heuristic = 2.000", flush=True)'
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, body))
        assert result["solvability"] != "solved" and result["plan_text"] is None


class TestFailures:
    def test_timeout_after_the_heuristic_keeps_the_grounding_time(self, tmp_path):
        body = """
            import time
            print("Initial heuristic = 2.000", flush=True)
            time.sleep(60)
        """
        result = run_timed(_pddl_dir(tmp_path), timeout=2, optic_cmd=_fake(tmp_path, body))
        t = result["timings"]
        assert result["solvability"] == "timeout" and not result["success"]
        assert t["grounding_s"] is not None and t["grounding_s"] < 1.5
        assert t["planning_s"] is None and t["first_plan_s"] is None
        assert 1.9 <= t["wall_s"] < 8  # killed at the timeout, not at 60 s
        assert "Initial heuristic" in result["stdout"]  # partial output is kept

    def test_timeout_during_grounding_has_no_grounding_time(self, tmp_path):
        body = "import time; time.sleep(60)"
        result = run_timed(_pddl_dir(tmp_path), timeout=1, optic_cmd=_fake(tmp_path, body))
        assert result["solvability"] == "timeout"
        assert result["timings"]["grounding_s"] is None

    def test_out_of_memory_abort_is_classified(self, tmp_path):
        body = """
            import sys
            print("Initial heuristic = 2.000", flush=True)
            sys.stderr.write("terminate called after throwing an instance of 'std::bad_alloc'\\n")
            sys.exit(134)
        """
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, body))
        assert result["solvability"] == "out_of_memory"
        assert result["timings"]["planning_s"] is None

    def test_a_problem_proved_unsolvable_still_gets_a_planning_time(self, tmp_path):
        body = """
            import time
            print("Initial heuristic = 2.000", flush=True)
            time.sleep(0.3)
            print(";; Problem unsolvable!", flush=True)
            raise SystemExit(1)
        """
        result = run_timed(_pddl_dir(tmp_path), timeout=30, optic_cmd=_fake(tmp_path, body))
        assert result["solvability"] == "unsolvable_structural"
        assert result["timings"]["planning_s"] >= 0.25

    def test_missing_stdbuf_is_an_error_not_wrong_times(self, tmp_path):
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="stdbuf"):
                run_timed(_pddl_dir(tmp_path), timeout=5, optic_cmd=["true"])

    def test_missing_pddl_files_are_reported(self, tmp_path):
        result = run_timed(tmp_path, timeout=5, optic_cmd=["true"])
        assert not result["success"] and "not found" in result["message"]


@pytest.mark.integration
@pytest.mark.skipif(not optic_module.is_available(), reason="OPTIC binary not compiled")
class TestWithTheRealPlanner:
    @staticmethod
    def _dir(tmp_path: Path, problem: str) -> Path:
        shutil.copy(FIXTURES / "simple_domain.pddl", tmp_path / "domain.pddl")
        shutil.copy(FIXTURES / problem, tmp_path / "problem.pddl")
        return tmp_path

    def test_solves_and_times_a_tiny_problem(self, tmp_path):
        result = run_timed(self._dir(tmp_path, "simple_problem_solvable.pddl"), timeout=60)
        t = result["timings"]
        assert result["success"]
        assert t["grounding_s"] is not None
        assert 0 <= t["grounding_s"] <= t["first_plan_s"] <= t["wall_s"]
        assert set(result["plan_actions"]) >= {"go-b", "go-c"}

    def test_unsolvable_problem_is_not_reported_as_solved(self, tmp_path):
        result = run_timed(self._dir(tmp_path, "simple_problem_unsolvable.pddl"), timeout=60)
        assert not result["success"]


@pytest.mark.integration
@pytest.mark.skipif(not optic_module.is_available(), reason="OPTIC binary not compiled")
class TestOptimisingWithTheRealPlanner:
    """Pins the real output format that _parse_optic_plans reads (OPTIC without -N)."""

    DOMAIN = """(define (domain d)
      (:requirements :strips :typing :durative-actions :duration-inequalities)
      (:predicates (p0) (p1) (p3))
      (:durative-action a :parameters () :duration (and (>= ?duration 4) (<= ?duration 5))
        :condition (and (at start (p0))) :effect (and (at start (not (p0))) (at end (p3))))
      (:durative-action b :parameters () :duration (and (>= ?duration 1) (<= ?duration 2))
        :condition (and (at start (p0))) :effect (and (at start (not (p0))) (at end (p1))))
      (:durative-action c :parameters () :duration (and (>= ?duration 1) (<= ?duration 2))
        :condition (and (at start (p1))) :effect (and (at start (not (p1))) (at end (p3))))
    )"""
    PROBLEM = "(define (problem p) (:domain d) (:init (p0)) (:goal (p3)) (:metric minimize (total-time)))"

    def test_the_fastest_plan_is_found_and_the_search_ends_by_itself(self, tmp_path):
        (tmp_path / "domain.pddl").write_text(self.DOMAIN)
        (tmp_path / "problem.pddl").write_text(self.PROBLEM)
        result = run_timed(tmp_path, timeout=60, stop_at_first=False)
        assert result["solvability"] == "solved" and result["search_completed"] is True
        assert result["plan_actions"] == ["b", "c"]  # 2 time units, faster than a alone (4)
        assert result["plans"][-1]["final"] and result["plans"][-1]["makespan"] == pytest.approx(2.001)
        assert result["metrics"]["duration"] == pytest.approx(2.001)
