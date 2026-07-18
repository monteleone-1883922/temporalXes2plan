"""Tests for planning.optic — focused on solvability classification.

optic.run() uses subprocess.Popen(start_new_session=True) + a manual
communicate(timeout=...)/killpg, not subprocess.run(timeout=...) — because
cmd is `bash -c "ulimit -v N; optic-clp.sh ..."` and optic-clp.sh has no
shebang, so it's interpreted by a nested shell that forks the actual
optic-clp binary; subprocess.run(timeout=...)'s default kill() only signals
the outer `bash -c` process, leaving that binary orphaned and still
consuming memory after a timeout. Tests patch subprocess.Popen accordingly
(not subprocess.run).
"""
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import planning.optic as optic_module


def _make_popen_mock(returncode=0, stdout="", stderr="", pid=12345):
    """A Popen-like mock: communicate() returns (stdout, stderr) once."""
    proc = MagicMock(pid=pid, returncode=returncode)
    proc.communicate.return_value = (stdout, stderr)
    return proc


def _make_timeout_popen_mock(cmd_timeout, pid=12345):
    """A Popen-like mock whose first communicate() call times out (like the
    real one during the planner's run) and whose second call (the post-kill
    reap in optic.run()) returns cleanly, as it would after killpg()."""
    proc = MagicMock(pid=pid, returncode=-9)
    proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="optic", timeout=cmd_timeout),
        ("", ""),
    ]
    return proc


class TestOpticTimeout:
    def test_timeout_emits_timeout_solvability(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=_make_timeout_popen_mock(5)), \
             patch("os.killpg"), patch("os.getpgid", return_value=999):
            result = optic_module.run(tmp_path, timeout=5)

        assert result["solvability"] == "timeout"
        assert result["success"] is False

    def test_timeout_message_mentions_duration(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=_make_timeout_popen_mock(30)), \
             patch("os.killpg"), patch("os.getpgid", return_value=999):
            result = optic_module.run(tmp_path, timeout=30)

        assert "30" in result["message"]

    def test_timeout_differs_from_unsolvable_resource(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=_make_timeout_popen_mock(1)), \
             patch("os.killpg"), patch("os.getpgid", return_value=999):
            result = optic_module.run(tmp_path, timeout=1)

        assert result["solvability"] != "unsolvable_resource"

    def test_timeout_kills_the_whole_process_group_not_just_popen_child(self, tmp_path):
        """Regression test for the orphaned-optic-clp bug: on timeout, the
        whole process group (killpg) must be signalled, not just proc.kill()
        (which would only reach the outer `bash -c`, per the module
        docstring)."""
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        popen_mock = _make_timeout_popen_mock(5, pid=42)
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=popen_mock), \
             patch("os.getpgid", return_value=4242) as mock_getpgid, \
             patch("os.killpg") as mock_killpg:
            optic_module.run(tmp_path, timeout=5)

        mock_getpgid.assert_called_once_with(42)
        mock_killpg.assert_called_once()

    def test_process_group_started_for_the_planner_subprocess(self, tmp_path):
        """The Popen call must opt into its own session/process group —
        otherwise killpg on timeout has no separate group to target."""
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=_make_popen_mock()) as mock_popen:
            optic_module.run(tmp_path, timeout=5)

        assert mock_popen.call_args.kwargs.get("start_new_session") is True

    def test_timeout_survives_process_already_gone(self, tmp_path):
        """killpg racing against the process exiting on its own right after
        the timeout must not raise/propagate ProcessLookupError."""
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=_make_timeout_popen_mock(5)), \
             patch("os.getpgid", return_value=999), \
             patch("os.killpg", side_effect=ProcessLookupError):
            result = optic_module.run(tmp_path, timeout=5)

        assert result["solvability"] == "timeout"


class TestOpticOutOfMemory:
    """OPTIC runs under `ulimit -v` (see optic.run()) — when it exceeds that
    cap, malloc/new fails and (uncaught) terminates via SIGABRT, observed as
    return_code 134 with "std::bad_alloc" on stderr. Real example captured
    in results/55/failures/55_S74983_Q3.json."""

    _REAL_OOM_STDERR = (
        "Warning: rounding numeric constants such as 216503 to an accuracy of 0.001\n"
        "terminate called after throwing an instance of 'std::bad_alloc'\n"
        "  what():  std::bad_alloc\n"
        "optic-clp.sh: line 1: 284014 Aborted (core dumped) ./release/optic/optic-clp $@\n"
    )

    def test_real_oom_capture_emits_out_of_memory_solvability(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        proc = _make_popen_mock(returncode=134, stdout="Number of literals: 37\n", stderr=self._REAL_OOM_STDERR)
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=proc):
            result = optic_module.run(tmp_path, memory_mb=100)

        assert result["solvability"] == "out_of_memory"
        assert result["success"] is False

    def test_out_of_memory_differs_from_unsolvable_resource(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        proc = _make_popen_mock(returncode=134, stdout="", stderr=self._REAL_OOM_STDERR)
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=proc):
            result = optic_module.run(tmp_path, memory_mb=100)

        assert result["solvability"] != "unsolvable_resource"

    def test_message_mentions_out_of_memory(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        proc = _make_popen_mock(returncode=134, stdout="", stderr=self._REAL_OOM_STDERR)
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=proc):
            result = optic_module.run(tmp_path, memory_mb=100)

        assert "memory" in result["message"].lower()


class TestOpticAllTauPlan:
    """A plan can legitimately consist entirely of tau/silent transitions
    (e.g. a structural shortcut straight to the goal) -- plan_actions (the
    non-tau names) ends up empty, but a solution was genuinely found.
    Real example captured in results/55/failures/55_C18710_Q1.json."""

    _REAL_ALL_TAU_STDOUT = (
        "Number of literals: 15\n"
        "Constructing lookup tables:\n"
        "Post filtering unreachable actions: \n"
        "(total-cost) has a finite lower bound: [0.000,inf]\n"
        "Action 2 - (execute_tau_3) is uninteresting once we have fact (marked sink)\n"
        "Initial heuristic = 1.000, admissible cost estimate 0.000\n"
        "(G);;;; Solution Found\n"
        "; States evaluated: 2\n"
        "; Cost: 0.000\n"
        "; Time 0.06\n"
        "0.000: (execute_tau_3)  [0.001]\n"
    )

    def test_all_tau_plan_is_classified_as_solved(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        proc = _make_popen_mock(returncode=0, stdout=self._REAL_ALL_TAU_STDOUT, stderr="")
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=proc):
            result = optic_module.run(tmp_path)

        assert result["solvability"] == "solved"
        assert result["success"] is True

    def test_all_tau_plan_text_is_saved(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        proc = _make_popen_mock(returncode=0, stdout=self._REAL_ALL_TAU_STDOUT, stderr="")
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=proc):
            result = optic_module.run(tmp_path)

        assert result["plan_text"] is not None
        assert (tmp_path / "plan.txt").exists()

    def test_all_tau_plan_has_zero_visible_actions(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        proc = _make_popen_mock(returncode=0, stdout=self._REAL_ALL_TAU_STDOUT, stderr="")
        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.Popen", return_value=proc):
            result = optic_module.run(tmp_path)

        assert result["plan_actions"] == []


class TestClassify:
    def test_solved_when_plan_exists(self):
        assert optic_module._classify(0, True, "", "") == "solved"

    def test_unsolvable_structural_on_keyword(self):
        assert optic_module._classify(1, False, "problem is unsolvable", "") == "unsolvable_structural"

    def test_unsolvable_parse_on_keyword(self):
        assert optic_module._classify(1, False, "", "parse error in domain") == "unsolvable_parse"

    def test_out_of_memory_on_return_code(self):
        assert optic_module._classify(134, False, "", "") == "out_of_memory"

    def test_out_of_memory_on_bad_alloc_keyword(self):
        assert optic_module._classify(1, False, "", "std::bad_alloc") == "out_of_memory"

    def test_out_of_memory_on_cannot_allocate_keyword(self):
        assert optic_module._classify(1, False, "cannot allocate memory", "") == "out_of_memory"

    def test_generic_unsolvable_resource_fallback(self):
        assert optic_module._classify(1, False, "", "") == "unsolvable_resource"


# ---------------------------------------------------------------------------
# _parse_optic_stdout / _normalise — action-name normalisation
#
# Real PDDL action names are generated by
# encoding/transition_action_builder.py (f"execute_{act_name}"[_v{i}]) and
# encoding/domain_builder.py (f"execute_{tau_label}" for silent transitions
# — tau labels look like "tau_1", see parsing/model_discoverer.py). Note
# that *silent* transitions also get the "execute_" prefix, not a "tau_"
# prefix of their own. Each firing is a single action (standard Petri net
# semantics: only places are ever marked — see
# claude_plans/standard_petri_net_marking_plan.md), so there is no separate
# "mark_places_from_*" line to normalise away.
# ---------------------------------------------------------------------------

REALISTIC_OPTIC_STDOUT = """
;;; Solution Found
; States evaluated: 12
; Cost: 8.001
0.000: (execute_run_lab_tests_v2) [8.000]
3.000: (execute_admit_patient) [1.000]
3.001: (execute_tau_1) [0.001]
Time 0.042
"""


class TestNormalise:
    def test_strips_execute_prefix(self):
        # Real action names use "execute_", not "exec_".
        assert optic_module._normalise("execute_admit_patient") == "admit_patient"

    def test_strips_variant_suffix_in_isolation(self):
        # Suffix stripping alone, no "execute_" prefix involved.
        assert optic_module._normalise("run_lab_tests_v2") == "run_lab_tests"

    def test_strips_execute_prefix_and_variant_suffix_together(self):
        assert optic_module._normalise("execute_run_lab_tests_v2") == "run_lab_tests"


class TestParseOpticStdoutActionNames:
    def test_action_names_match_sanitized_log_activity_names(self):
        """Names in `actions` must be directly comparable to
        core_utils.sanitize_name(event["concept:name"]) — this is what
        metrics_collector.sequence_alignment_score() compares them against
        (see evaluation/run_evaluation.py::_run_query)."""
        _, actions, _ = optic_module._parse_optic_stdout(REALISTIC_OPTIC_STDOUT)
        assert actions == ["run_lab_tests", "admit_patient"]

    def test_silent_tau_actions_are_excluded(self):
        _, actions, _ = optic_module._parse_optic_stdout(REALISTIC_OPTIC_STDOUT)
        assert not any("tau" in a for a in actions)

    def test_no_execute_prefix_leaks_into_actions(self):
        _, actions, _ = optic_module._parse_optic_stdout(REALISTIC_OPTIC_STDOUT)
        assert not any(a.startswith("execute_") for a in actions)

    def test_makespan_is_max_end_time_across_all_lines(self):
        # Last line to START is not necessarily the last to END for a
        # temporal plan with concurrent actions — makespan must be
        # max(timestamp + duration), not just the last line's end time.
        # Here run_lab_tests_v2 starts first (t=0.000) but its 8s duration
        # outlasts both later-starting lines (admit_patient ends at 4.000,
        # tau_1 ends at 3.002).
        _, _, makespan = optic_module._parse_optic_stdout(REALISTIC_OPTIC_STDOUT)
        assert makespan == pytest.approx(8.000)


class TestParseOpticMetricsCostAndDuration:
    def test_cost_parsed_from_stdout(self):
        metrics = optic_module._parse_optic_metrics(REALISTIC_OPTIC_STDOUT, "", makespan=8.000)
        assert metrics["cost"] == pytest.approx(8.001)

    def test_duration_equals_makespan(self):
        metrics = optic_module._parse_optic_metrics(REALISTIC_OPTIC_STDOUT, "", makespan=8.000)
        assert metrics["duration"] == pytest.approx(8.000)

    def test_cost_none_when_absent(self):
        stdout = "0.000: (execute_a) [1.000]\nTime 0.01"
        metrics = optic_module._parse_optic_metrics(stdout, "", makespan=1.0)
        assert metrics["cost"] is None
