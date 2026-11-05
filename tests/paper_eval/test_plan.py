"""Tests for paper_eval.plan (phase "plan": OPTIC on every instance).

OPTIC is replaced by a fake runner, and the model by a stand-in that has only
what the runner reads, so these tests need neither the planner nor a built model.
"""
import json
from types import SimpleNamespace

import pytest

from paper_eval import plan as pl
from paper_eval import prefixes as px
from paper_eval.config import PaperEvalConfig

SCALE = 60.0


def _model():
    return SimpleNamespace(
        domain=SimpleNamespace(duration_scale_factor=SCALE),
        serialized={
            "attribute_catalog": {"crp": {"type": "categorical", "possible_values": ["high", "low"]}},
            "metadata": {"end_place": "p_end"},
        },
    )


def _instance(group_id, goal="completion", k=2, bucket="2", deadline_s=None, replay_ok=True, expired=False):
    return {
        "group_id": group_id, "k": k, "bucket": bucket, "prefix_acts": ["A", "B"], "goal": goal,
        "init_places": ["p1"] if replay_ok else None,
        "init_effects": [{"attribute": "crp", "value": "high"}] if replay_ok else None,
        "replay_ok": replay_ok, "expired": expired, "deadline_s": deadline_s,
        "cases": [{"case_id": "c1"}, {"case_id": "c2"}],
    }


def _result(status="solved", makespan=2.0):
    """What optic.run_timed returns for a solved/other run."""
    solved = status == "solved"
    return {
        "success": solved, "solvability": status,
        "plan_text": "0.000: (a)  [2.000]" if solved else None,
        "plan_actions": ["a"] if solved else [],
        "metrics": {"duration": makespan if solved else None, "expanded_nodes": 7, "total_time": 0.12},
        "message": status,
        "timings": {"grounding_s": 0.5, "first_plan_s": 1.5 if solved else None,
                    "planning_s": 1.0 if solved else None, "wall_s": 1.6, "cpu_s": 0.12},
    }


class FakeRunner:
    """Stands in for optic.run_timed and remembers how it was called."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, pddl_dir, timeout, memory_mb, stop_at_first):
        self.calls.append({
            "timeout": timeout, "memory_mb": memory_mb, "stop_at_first": stop_at_first,
            "problem": (pddl_dir / "problem.pddl").read_text(),
            "domain": (pddl_dir / "domain.pddl").read_text(),
        })
        return self.results.get(len(self.calls), _result())


SETTINGS = {"timeout_s": 60, "memory_mb": 4000, "metric": pl.DEFAULT_METRIC, "mode": "first_plan", "stop_at_first": True}


class TestBuildProblemText:
    def test_completion_problem(self):
        text = pl.build_problem_text(_model(), _instance(0), pl.DEFAULT_METRIC, 0.5)
        assert "(:domain paper_eval)" in text  # must match the name written in domain.pddl
        assert "(marked p1)" in text and "crp" in text
        assert "(marked p_end)" in text  # the goal: reach the end of the process
        assert "(not (deadline_ok))" not in text  # no deadline, no timed initial literal

    def test_deadline_is_divided_by_the_duration_scale(self):
        # 150 s at scale 60 -> 2.5 time units in the PDDL
        text = pl.build_problem_text(_model(), _instance(0, "deadline", deadline_s=150.0), pl.DEFAULT_METRIC, 0.5)
        assert "(at 2.5 (not (deadline_ok)))" in text


class TestPlanInstance:
    def _row(self, instance, runner=None, tmp_path=None):
        runner = runner or FakeRunner()
        (tmp_path / "domain.pddl").write_text("(define (domain paper_eval))")  # run_plan copies it there
        return pl.plan_instance(instance, _model(), tmp_path, "toy", SETTINGS, 0.5, runner), runner

    def test_solved_instance_row(self, tmp_path):
        row, runner = self._row(_instance(3, "deadline", k=5, bucket="5", deadline_s=300.0), tmp_path=tmp_path)
        assert row["status"] == "solved"
        assert (row["log"], row["group_id"], row["goal"], row["k"], row["bucket"]) == ("toy", 3, "deadline", 5, "5")
        assert row["n_cases"] == 2 and row["deadline_s"] == 300.0
        assert (row["grounding_s"], row["planning_s"], row["first_plan_s"]) == (0.5, 1.0, 1.5)
        assert row["states_evaluated"] == 7 and row["cpu_s"] == 0.12
        assert row["makespan"] == 2.0 and row["makespan_s"] == pytest.approx(2.0 * SCALE)  # back to real seconds
        assert row["plan_actions"] == ["a"] and row["plan_length"] == 1
        assert (row["timeout_s"], row["memory_mb"]) == (60, 4000)
        assert runner.calls[0]["stop_at_first"] is True and runner.calls[0]["timeout"] == 60

    def test_failed_run_keeps_its_status_and_has_no_plan_fields(self, tmp_path):
        runner = FakeRunner({1: _result("timeout")})
        row, _ = self._row(_instance(0), runner, tmp_path)
        assert row["status"] == "timeout"
        assert row["plan_actions"] is None and row["plan_length"] is None and row["makespan_s"] is None
        assert row["grounding_s"] == 0.5  # grounding finished even if the search did not

    def test_replay_failure_is_skipped_without_calling_optic(self, tmp_path):
        row, runner = self._row(_instance(0, replay_ok=False), tmp_path=tmp_path)
        assert row["status"] == pl.STATUS_SKIPPED_REPLAY and runner.calls == []
        assert row["grounding_s"] is None

    def test_expired_deadline_is_skipped_without_calling_optic(self, tmp_path):
        row, runner = self._row(_instance(0, "deadline", deadline_s=-5.0, expired=True), tmp_path=tmp_path)
        assert row["status"] == pl.STATUS_SKIPPED_EXPIRED and runner.calls == []

    def test_an_exception_becomes_an_error_row(self, tmp_path):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        row, _ = self._row(_instance(0), broken, tmp_path)
        assert row["status"] == pl.STATUS_ERROR and "boom" in row["message"]

    def test_a_problem_that_cannot_be_built_is_an_error_row(self, tmp_path):
        bad = _instance(0)
        del bad["init_effects"]
        row, runner = self._row(bad, tmp_path=tmp_path)
        assert row["status"] == pl.STATUS_ERROR and runner.calls == []


# ---------------------------------------------------------------------------
# run_plan: files, order, resume, settings
# ---------------------------------------------------------------------------

@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A model folder with instances.jsonl and a domain, and the paths/cfg to run on it."""
    model_dir = tmp_path / "results" / "toy" / "hash"
    model_dir.mkdir(parents=True)
    (model_dir / "domain.pddl").write_text("(define (domain paper_eval))")
    px.write_jsonl([
        _instance(0), _instance(1, "deadline", deadline_s=120.0),
        _instance(2, replay_ok=False), _instance(3, "deadline", deadline_s=-1.0, expired=True),
        _instance(4, k=3, bucket="3"),
    ], model_dir / "instances.jsonl")
    paths = SimpleNamespace(split_json=tmp_path / "split.json", model_dir=lambda h: model_dir)
    monkeypatch.setattr(pl, "read_split", lambda p: SimpleNamespace(split_hash="hash"))
    monkeypatch.setattr(pl, "load_model", lambda d: _model())
    monkeypatch.setattr(pl, "compute_cost_scale_factor", lambda domain: 0.5)
    return SimpleNamespace(model_dir=model_dir, paths=paths, cfg=PaperEvalConfig())


def _rows(model_dir):
    return pl.read_results(model_dir / pl.RESULTS_FILE)


class TestRunPlan:
    def test_every_instance_gets_one_row_in_group_order(self, setup):
        runner = FakeRunner()
        counts = pl.run_plan("toy", setup.cfg, setup.paths, runner=runner)
        rows = _rows(setup.model_dir)
        assert [r["group_id"] for r in rows] == [0, 1, 2, 3, 4]
        assert [r["status"] for r in rows] == ["solved", "solved", pl.STATUS_SKIPPED_REPLAY, pl.STATUS_SKIPPED_EXPIRED, "solved"]
        assert counts == {"solved": 3, pl.STATUS_SKIPPED_REPLAY: 1, pl.STATUS_SKIPPED_EXPIRED: 1}
        assert len(runner.calls) == 3  # skipped ones never reach OPTIC

    def test_domain_is_copied_next_to_each_problem(self, setup):
        runner = FakeRunner()
        pl.run_plan("toy", setup.cfg, setup.paths, runner=runner)
        assert all(c["domain"] == "(define (domain paper_eval))" for c in runner.calls)
        assert all("(:domain paper_eval)" in c["problem"] for c in runner.calls)

    def test_metadata_records_the_settings_and_the_environment(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, runner=FakeRunner())
        meta = json.loads((setup.model_dir / pl.META_FILE).read_text())
        assert meta["settings"] == SETTINGS
        assert meta["duration_scale_factor"] == SCALE and meta["cost_weight"] == 0.5
        assert {"host", "platform", "optic_version", "stdbuf", "started_at"} <= set(meta["environment"])

    def test_a_rerun_skips_what_is_already_done(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, runner=FakeRunner())
        runner = FakeRunner()
        counts = pl.run_plan("toy", setup.cfg, setup.paths, runner=runner)
        assert counts == {} and runner.calls == []
        assert len(_rows(setup.model_dir)) == 5

    def test_limit_handles_only_the_next_instances_and_the_rest_follows_later(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, limit=2, runner=FakeRunner())
        assert [r["group_id"] for r in _rows(setup.model_dir)] == [0, 1]
        pl.run_plan("toy", setup.cfg, setup.paths, runner=FakeRunner())
        assert [r["group_id"] for r in _rows(setup.model_dir)] == [0, 1, 2, 3, 4]

    def test_goal_filter(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, goal="deadline", runner=FakeRunner())
        assert {r["goal"] for r in _rows(setup.model_dir)} == {"deadline"}

    def test_unknown_goal_is_rejected(self, setup):
        with pytest.raises(ValueError, match="goal"):
            pl.run_plan("toy", setup.cfg, setup.paths, goal="nope", runner=FakeRunner())

    def test_per_log_settings_come_from_the_config(self, setup):
        setup.cfg.plan_timeout_by_log = {"toy": 5}
        setup.cfg.plan_memory_by_log = {"toy": 777}
        runner = FakeRunner()
        pl.run_plan("toy", setup.cfg, setup.paths, limit=1, runner=runner)
        assert (runner.calls[0]["timeout"], runner.calls[0]["memory_mb"]) == (5, 777)

    def test_command_line_values_override_the_config(self, setup):
        setup.cfg.plan_timeout_by_log = {"toy": 5}
        runner = FakeRunner()
        pl.run_plan("toy", setup.cfg, setup.paths, limit=1, timeout_s=9, memory_mb=123, runner=runner)
        assert (runner.calls[0]["timeout"], runner.calls[0]["memory_mb"]) == (9, 123)

    def test_results_made_with_other_settings_are_not_mixed(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, limit=1, runner=FakeRunner())
        with pytest.raises(ValueError, match="other settings"):
            pl.run_plan("toy", setup.cfg, setup.paths, timeout_s=10, runner=FakeRunner())
        assert len(_rows(setup.model_dir)) == 1  # untouched

    def test_force_discards_the_old_results_and_starts_over(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, limit=1, runner=FakeRunner())
        runner = FakeRunner()
        pl.run_plan("toy", setup.cfg, setup.paths, timeout_s=10, force=True, runner=runner)
        assert len(_rows(setup.model_dir)) == 5 and runner.calls[0]["timeout"] == 10

    def test_a_partial_last_line_is_ignored_and_the_instance_is_redone(self, setup):
        pl.run_plan("toy", setup.cfg, setup.paths, limit=2, runner=FakeRunner())
        with open(setup.model_dir / pl.RESULTS_FILE, "a") as fh:
            fh.write('{"group_id": 2, "status": "sol')  # killed while writing
        pl.run_plan("toy", setup.cfg, setup.paths, runner=FakeRunner())
        assert [r["group_id"] for r in _rows(setup.model_dir)] == [0, 1, 2, 3, 4]

    def test_missing_instances_file_says_which_phase_to_run(self, setup):
        (setup.model_dir / "instances.jsonl").unlink()
        with pytest.raises(FileNotFoundError, match="prefixes"):
            pl.run_plan("toy", setup.cfg, setup.paths, runner=FakeRunner())

    def test_an_optic_error_on_one_instance_does_not_stop_the_run(self, setup):
        calls = {"n": 0}

        def flaky(pddl_dir, timeout, memory_mb, stop_at_first):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return _result()

        pl.run_plan("toy", setup.cfg, setup.paths, runner=flaky)
        statuses = [r["status"] for r in _rows(setup.model_dir)]
        assert statuses[0] == pl.STATUS_ERROR and statuses[1] == "solved" and len(statuses) == 5


class TestConfigPerLog:
    def test_defaults(self):
        cfg = PaperEvalConfig()
        assert (cfg.timeout_for("sepsis"), cfg.memory_for("sepsis")) == (60, 4000)

    def test_overrides_apply_to_one_log_only(self):
        cfg = PaperEvalConfig(plan_timeout_by_log={"bpic17": 300}, plan_memory_by_log={"bpic17": 8000})
        assert (cfg.timeout_for("bpic17"), cfg.memory_for("bpic17")) == (300, 8000)
        assert (cfg.timeout_for("sepsis"), cfg.memory_for("sepsis")) == (60, 4000)
