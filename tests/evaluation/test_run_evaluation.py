"""Unit tests for evaluation.run_evaluation.

evaluate_log() now builds the domain exactly once per log via
EvalAPI.build_network() (a thin wrapper around pipeline.Pipeline.build_network(),
which itself optionally runs the network_search optimizer) — unless
data/<log_id>/current.json already exists, in which case it's reused as a
cache and reconstructed via PreparedDomainInput.from_dict() +
build_petrinet_model_from_prepared() + build_domain_with_variant_map()
instead of re-parsing (see evaluation/run_evaluation.py's own module
docstring / the "3-4. Build the domain" comment block). --force-rebuild
bypasses that cache. Tests patch evaluation.run_evaluation.WEB_DATA_DIR to
tmp_path so they never touch the real project data/ directory.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.run_evaluation import (
    EvalConfig,
    _load_log_result,
    _prompt_csv_mapping,
    _run_query,
    build_parser,
    evaluate_log,
    main,
)
from evaluation.report_generator import LogResult, QueryResult
from models import AnalysisConfig
from pipeline import BuildResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(output_dir: Path, **overrides) -> EvalConfig:
    defaults = dict(
        log_id=1,
        test_pct=0.2,
        min_test_cases=1,
        max_test_cases=100,
        min_prefix_pct=0.2,
        max_prefix_pct=0.8,
        seed=42,
        algorithm="inductive",
        coverage=0.001,
        planner_timeout=60,
        planner_memory_mb=4000,
        max_retries=1,
        retry_delay=0.0,
        cache_dir=output_dir / "cache",
        output_dir=output_dir,
        force_download=False,
        resume=False,
        cost_weight=0.001,
        csv_mapping=None,
        force_rebuild=False,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _make_log_result_data(log_id="log_1", pipeline_ok=True, used_optimizer=False, search_summary=None):
    return {
        "log_id": log_id,
        "log_name": "Test Log",
        "log_fmt": "xes",
        "n_train_cases": 80,
        "n_test_cases": 20,
        "n_activities": 5,
        "pipeline_ok": pipeline_ok,
        "pipeline_error": None,
        "used_optimizer": used_optimizer,
        "search_summary": search_summary,
        "queries": [
            {
                "query_id": f"{log_id}_case1_Q1",
                "query_type": "Q1",
                "trace_id": "case1",
                "prefix_ratio": 0.5,
                "n_prefix_events": 3,
                "attempts": 1,
                "solvability": "solved",
                "planner_duration_s": 1.0,
                "metrics": {"solved": True, "plan_time_s": 1.0, "weighted_objective": 1.0},
                "validation": None,
            }
        ],
    }


def _make_serialized():
    return {
        "graph": {"nodes": [], "edges": []},
        "transitions": {},
        "attribute_catalog": {},
        "metadata": {"start_place": "p_start", "end_place": "p_end"},
    }


def _make_domain_mock(text="(define (domain test))"):
    """A PDDLDomain-like mock: str(domain) == text, and .write(path) actually
    writes the file (mirroring encoding.pddl_model.PDDLDomain.write())."""
    domain = MagicMock()
    domain.__str__.return_value = text

    def _write(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    domain.write.side_effect = _write
    return domain


def _make_build_result(n_activities=3, trial_records=None, domain_text="(define (domain test))"):
    parse_result = MagicMock()
    parse_result.petri_net_model.activities = list(range(n_activities))
    return BuildResult(
        parse_result=parse_result,
        prepared=MagicMock(),
        domain=_make_domain_mock(domain_text),
        variant_map={},
        config=AnalysisConfig(),
        domain_name="test_process",
        trial_records=trial_records,
    )


def _make_mock_api(missing_timestamp=False, build_result=None, serialized=None):
    api = MagicMock()
    vr = MagicMock()
    vr.missing_timestamp = missing_timestamp
    api.validate_log.return_value = vr
    api.build_network.return_value = build_result or _make_build_result()
    api.serialize.return_value = serialized if serialized is not None else _make_serialized()
    api.build_problem.return_value = "(define (problem p) (:domain test))"
    plan_result = MagicMock()
    plan_result.success = True
    plan_result.plan_steps = ["act_a"]
    plan_result.cost = 1.0
    plan_result.duration_s = 1.0
    plan_result.solvability = "solved"
    plan_result.error = None
    plan_result.raw_stdout = ""
    plan_result.raw_stderr = ""
    api.run_planner.return_value = plan_result
    return api


def _make_mock_tts(n_train=80, n_test=2):
    tts = MagicMock()
    tts.n_train = n_train
    tts.n_test = n_test
    tts.train_path = Path("/tmp/train.xes")

    trace = MagicMock()
    trace.attributes = {"concept:name": "case1"}
    tts.test_cases = [trace] * n_test
    tts.__enter__ = lambda s: s
    tts.__exit__ = MagicMock(return_value=False)
    return tts


def _make_replay_outcome(is_replayable=True):
    outcome = MagicMock()
    outcome.is_replayable = is_replayable
    outcome.error_step = None
    outcome.error_reason = None
    outcome.steps = ["act_a"]
    return outcome


def _make_prefix():
    prefix = MagicMock()
    prefix.prefix_ratio = 0.5
    prefix.prefix_events = [MagicMock()]
    prefix.init_places = ["p_start"]
    prefix.init_effects = []
    prefix.full_duration_s = 3600.0
    prefix.prefix_duration_s = 1000.0
    prefix.final_event_attributes = {}
    prefix.is_replayable = True
    return prefix


def _make_q1_spec():
    q1_spec = MagicMock()
    q1_spec.query_type = "Q1"
    q1_spec.init_places = ["p_start"]
    q1_spec.init_effects = []
    q1_spec.goal_sop = [[]]
    q1_spec.metric = "minimize_weighted"
    q1_spec.cost_weight = 0.001
    q1_spec.require_completion = True
    q1_spec.deadline = None
    return q1_spec


def _patched_evaluate_log(*, api, cfg, tmp_path, log_id="log_1", n_test=2, prefix=None,
                           mock_prefix=True):
    """Context-manager-returning helper: patches every collaborator
    evaluate_log() calls (except the cache-reconstruction trio, which only
    the cache-hit tests need — see _patch_cache_reconstruction)."""
    q1_spec = _make_q1_spec()
    prefix_value = (prefix or _make_prefix()) if mock_prefix else None
    return patch.multiple(
        "evaluation.run_evaluation",
        WEB_DATA_DIR=tmp_path / "data",
        _split_log=MagicMock(return_value=_make_mock_tts(n_test=n_test)),
        _sample_prefix=MagicMock(return_value=prefix_value),
        build_q1=MagicMock(return_value=q1_spec),
        build_q2=MagicMock(return_value=None),
        build_q3=MagicMock(return_value=None),
        q1_metrics=MagicMock(return_value={"solved": True, "weighted_objective": 1.0}),
        replay_plan=MagicMock(return_value=_make_replay_outcome()),
        TraceReplayer=MagicMock(),
    )


def _patch_cache_reconstruction(n_activities=9, domain_text="(define (domain cached))"):
    """Patches the three functions the cache-hit branch uses to reconstruct
    prepared/petri_net_model/domain straight from current.json, without a
    real PreparedDomainInput/pm4py PetriNet."""
    prepared = MagicMock()
    petri_net_model = MagicMock()
    petri_net_model.activities = list(range(n_activities))
    domain = _make_domain_mock(domain_text)
    return patch.multiple(
        "evaluation.run_evaluation",
        PreparedDomainInput=MagicMock(**{"from_dict.return_value": prepared}),
        build_petrinet_model_from_prepared=MagicMock(return_value=petri_net_model),
        build_domain_with_variant_map=MagicMock(return_value=(domain, {})),
    )


def _write_cached_current_json(tmp_path, log_id="log_1"):
    data_dir = tmp_path / "data" / log_id
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "current.json").write_text(
        json.dumps(_make_serialized()), encoding="utf-8"
    )
    return data_dir


# ---------------------------------------------------------------------------
# _load_log_result
# ---------------------------------------------------------------------------

class TestLoadLogResult:
    def test_reconstructs_log_result(self):
        data = _make_log_result_data()
        lr = _load_log_result(data)
        assert lr.log_id == "log_1"
        assert lr.pipeline_ok is True
        assert len(lr.queries) == 1

    def test_reconstructs_query_result(self):
        data = _make_log_result_data()
        lr = _load_log_result(data)
        q = lr.queries[0]
        assert q.query_type == "Q1"
        assert q.solvability == "solved"

    def test_empty_queries(self):
        data = _make_log_result_data()
        data["queries"] = []
        lr = _load_log_result(data)
        assert lr.queries == []

    def test_used_optimizer_and_search_summary_round_trip(self):
        data = _make_log_result_data(
            used_optimizer=True,
            search_summary={"n_trials": 30, "best_score": 4.2, "best_trial_number": 7},
        )
        lr = _load_log_result(data)
        assert lr.used_optimizer is True
        assert lr.search_summary == {"n_trials": 30, "best_score": 4.2, "best_trial_number": 7}

    def test_used_optimizer_defaults_false_when_absent(self):
        data = _make_log_result_data()
        del data["used_optimizer"]
        del data["search_summary"]
        lr = _load_log_result(data)
        assert lr.used_optimizer is False
        assert lr.search_summary is None


# ---------------------------------------------------------------------------
# evaluate_log — missing timestamps / validate_log failure
# ---------------------------------------------------------------------------

class TestEvaluateLogMissingTimestamps:
    def test_missing_timestamps_marks_log_failed(self, tmp_path):
        api = _make_mock_api(missing_timestamp=True)
        cfg = _make_cfg(tmp_path)
        lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)
        assert lr.pipeline_ok is False
        assert lr.pipeline_error == "missing_timestamps"

    def test_missing_timestamps_no_queries(self, tmp_path):
        api = _make_mock_api(missing_timestamp=True)
        cfg = _make_cfg(tmp_path)
        lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)
        assert lr.queries == []

    def test_validate_log_exception_marks_failed(self, tmp_path):
        api = MagicMock()
        api.validate_log.side_effect = RuntimeError("disk error")
        cfg = _make_cfg(tmp_path)
        lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)
        assert lr.pipeline_ok is False
        assert "disk error" in lr.pipeline_error


# ---------------------------------------------------------------------------
# evaluate_log — build failures (build-fresh and cache-hit paths)
# ---------------------------------------------------------------------------

class TestEvaluateLogBuildFailures:
    def test_build_network_exception_marks_failed(self, tmp_path):
        api = _make_mock_api()
        api.build_network.side_effect = RuntimeError("build failed")
        cfg = _make_cfg(tmp_path)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        assert lr.pipeline_ok is False
        assert "build failed" in lr.pipeline_error

    def test_cache_hit_reconstruction_exception_marks_failed(self, tmp_path):
        # current.json exists, but reconstructing the domain from it fails.
        _write_cached_current_json(tmp_path)
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        with (
            _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path),
            patch("evaluation.run_evaluation.build_domain_with_variant_map",
                  side_effect=RuntimeError("bad cache")),
        ):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        assert lr.pipeline_ok is False
        assert "bad cache" in lr.pipeline_error
        api.build_network.assert_not_called()


# ---------------------------------------------------------------------------
# evaluate_log — successful flow (build-fresh path, no current.json cached)
# ---------------------------------------------------------------------------

class TestEvaluateLogSuccess:
    def _run(self, tmp_path, n_test=2, mock_prefix=True):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=n_test, mock_prefix=mock_prefix):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        return lr

    def test_pipeline_ok_true_on_success(self, tmp_path):
        lr = self._run(tmp_path)
        assert lr.pipeline_ok is True

    def test_n_activities_from_build_result(self, tmp_path):
        lr = self._run(tmp_path)
        assert lr.n_activities == 3  # _make_build_result()'s default

    def test_domain_pddl_written_under_data_dir(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "data" / "log_1" / "pddl" / "domain.pddl").exists()

    def test_current_json_published_under_data_dir(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "data" / "log_1" / "current.json").exists()

    def test_no_domain_pddl_written_under_output_dir(self, tmp_path):
        # Dedup guard: the domain/network must live only under data/<log_id>/,
        # never duplicated under the results output_dir.
        self._run(tmp_path)
        assert not (tmp_path / "log_1" / "domain.pddl").exists()
        assert not (tmp_path / "log_1" / "current.json").exists()

    def test_used_optimizer_true_by_default(self, tmp_path):
        lr = self._run(tmp_path)
        assert lr.used_optimizer is True

    def test_skipped_trace_produces_no_queries(self, tmp_path):
        lr = self._run(tmp_path, n_test=2, mock_prefix=False)
        assert lr.queries == []

    def test_q3_skipped_produces_skipped_record(self, tmp_path):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        q3_records = [q for q in lr.queries if q.query_type == "Q3"]
        assert len(q3_records) == 1
        assert q3_records[0].solvability == "skipped_no_attributes"


# ---------------------------------------------------------------------------
# _run_query — deadline must be divided by duration_scale_factor before
# entering the PDDL problem, to stay on the domain's (possibly rescaled)
# time axis — see encoding/domain_builder.py::_maybe_rescale_durations.
# ---------------------------------------------------------------------------

class TestRunQueryDurationScale:
    def _spec(self, deadline):
        spec = MagicMock()
        spec.query_type = "Q2"
        spec.init_places = ["p_start"]
        spec.init_effects = []
        spec.goal_sop = [[]]
        spec.metric = "minimize_weighted"
        spec.cost_weight = 0.001
        spec.require_completion = True
        spec.deadline = deadline
        return spec

    def _serialized_trivially_reachable(self):
        # end_place == init place -> compute_min_time_to_end's Q2 precheck
        # (now unconditional, no longer optionally injected) trivially
        # returns 0.0 <= any deadline, so it never interferes with what
        # this test class actually exercises (deadline/scale-factor math).
        s = _make_serialized()
        s["metadata"]["end_place"] = "p_start"
        return s

    def _call(self, tmp_path, deadline, duration_scale_factor):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)
        prefix = _make_prefix()
        plan_result = MagicMock(success=False, plan_steps=None, solvability="unsolvable_resource", search_time_s=1.0)

        with patch("evaluation.run_evaluation._run_with_retry", return_value=(plan_result, 1)), \
             patch("evaluation.run_evaluation.q2_metrics", return_value={}):
            _run_query(
                "log_1", "case1", self._spec(deadline), "(define (domain test))",
                api, cfg, self._serialized_trivially_reachable(), tmp_path / "failures", prefix,
                MagicMock(), {}, pddl_dir=tmp_path / "pddl",
                duration_scale_factor=duration_scale_factor,
            )
        return api

    def test_no_deadline_stays_none_regardless_of_scale(self, tmp_path):
        api = self._call(tmp_path, deadline=None, duration_scale_factor=3600.0)
        assert api.build_problem.call_args.kwargs["deadline"] is None

    def test_deadline_divided_by_scale_factor(self, tmp_path):
        api = self._call(tmp_path, deadline=7200.0, duration_scale_factor=3600.0)
        assert api.build_problem.call_args.kwargs["deadline"] == 2.0

    def test_deadline_unchanged_when_no_rescale(self, tmp_path):
        api = self._call(tmp_path, deadline=7200.0, duration_scale_factor=1.0)
        assert api.build_problem.call_args.kwargs["deadline"] == 7200.0


# ---------------------------------------------------------------------------
# _run_query — Q3 must skip the planner entirely (no build_problem, no
# _run_with_retry) when is_q3_reachable() finds the goal structurally
# unreachable from the current marking, and record solvability=
# "skipped_unreachable" with metrics={} (same shape as the existing
# "skipped_no_attributes" record built in evaluate_log()).
# ---------------------------------------------------------------------------

class TestRunQueryQ3Unreachable:
    def _spec(self):
        spec = MagicMock()
        spec.query_type = "Q3"
        spec.init_places = ["p_start"]
        spec.init_effects = []
        spec.goal_sop = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        spec.metric = "minimize_weighted"
        spec.cost_weight = 0.001
        spec.require_completion = True
        spec.deadline = None
        return spec

    def test_skips_planner_when_goal_unreachable(self, tmp_path):
        # _make_serialized() has an empty graph and end_place="p_end", which
        # is never in the marking — is_q3_reachable() is False regardless of
        # goal_sop, so the planner must never be invoked.
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)
        prefix = _make_prefix()

        with patch("evaluation.run_evaluation._run_with_retry") as mock_run:
            result = _run_query(
                "log_1", "case1", self._spec(), "(define (domain test))",
                api, cfg, _make_serialized(), tmp_path / "failures", prefix,
                MagicMock(), {}, pddl_dir=tmp_path / "pddl",
            )

        mock_run.assert_not_called()
        api.build_problem.assert_not_called()
        assert result.solvability == "skipped_unreachable"
        assert result.attempts == 0
        assert result.metrics == {}
        assert result.planner_duration_s is None
        assert result.validation is None


# ---------------------------------------------------------------------------
# _run_query — Q2 must skip the planner entirely (no build_problem, no
# _run_with_retry) when even the fastest structurally possible completion
# (compute_min_time_to_end, recomputed per query from spec.init_places --
# see its own docstring on why AND-join correctness rules out a once-per-log
# cache) exceeds the deadline, and record
# solvability="skipped_unsatisfiable_by_construction" with metrics={}
# (same shape as Q3's "skipped_unreachable" record).
# ---------------------------------------------------------------------------

class TestRunQueryQ2UnsatisfiableByConstruction:
    def _spec(self, deadline=100.0):
        spec = MagicMock()
        spec.query_type = "Q2"
        spec.init_places = ["p_start"]
        spec.init_effects = []
        spec.goal_sop = [[]]
        spec.metric = "minimize_weighted"
        spec.cost_weight = 0.001
        spec.require_completion = True
        spec.deadline = deadline
        return spec

    def _serialized_with_duration(self, effective_min: float) -> dict:
        return {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t_a", "type": "transition", "label": "act_a"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t_a"},
                    {"source": "t_a", "target": "p_end"},
                ],
            },
            "transitions": {
                "act_a": {
                    "input_places": ["p_start"], "preconditions": [], "effect_groups": [],
                    "cost": 0.0,
                    "duration": {"effective_min": effective_min, "effective_max": effective_min * 2},
                },
            },
            "attribute_catalog": {},
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }

    def test_skips_planner_when_fastest_completion_exceeds_deadline(self, tmp_path):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)
        prefix = _make_prefix()

        with patch("evaluation.run_evaluation._run_with_retry") as mock_run:
            result = _run_query(
                "log_1", "case1", self._spec(deadline=100.0), "(define (domain test))",
                api, cfg, self._serialized_with_duration(500.0), tmp_path / "failures", prefix,
                MagicMock(), {}, pddl_dir=tmp_path / "pddl",
            )

        mock_run.assert_not_called()
        api.build_problem.assert_not_called()
        assert result.solvability == "skipped_unsatisfiable_by_construction"
        assert result.attempts == 0
        assert result.metrics == {}
        assert result.planner_duration_s is None
        assert result.validation is None

    def test_runs_planner_when_fastest_completion_fits_deadline(self, tmp_path):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)
        prefix = _make_prefix()

        with patch("evaluation.run_evaluation._run_with_retry", return_value=(MagicMock(success=True, plan_steps=None), 1)) as mock_run, \
             patch("evaluation.run_evaluation.q2_metrics", return_value={}):
            result = _run_query(
                "log_1", "case1", self._spec(deadline=100.0), "(define (domain test))",
                api, cfg, self._serialized_with_duration(5.0), tmp_path / "failures", prefix,
                MagicMock(), {}, pddl_dir=tmp_path / "pddl",
            )

        mock_run.assert_called_once()
        assert result.solvability != "skipped_unsatisfiable_by_construction"

    def test_skips_planner_when_end_place_structurally_unreachable(self, tmp_path):
        # p_start disconnected from p_end entirely -- compute_min_time_to_end
        # returns +inf, which must exceed any finite deadline (not silently
        # default to 0 / always-reachable).
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)
        prefix = _make_prefix()

        with patch("evaluation.run_evaluation._run_with_retry") as mock_run:
            result = _run_query(
                "log_1", "case1", self._spec(deadline=100.0), "(define (domain test))",
                api, cfg, _make_serialized(), tmp_path / "failures", prefix,  # empty graph
                MagicMock(), {}, pddl_dir=tmp_path / "pddl",
            )

        mock_run.assert_not_called()
        assert result.solvability == "skipped_unsatisfiable_by_construction"

    def test_no_precheck_applied_when_deadline_is_none(self, tmp_path):
        # Q1-shaped spec.deadline=None -- the precheck must be a no-op even
        # on a graph that would otherwise fail it (planner still invoked
        # normally), since "no deadline" means there is nothing to violate.
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)
        prefix = _make_prefix()

        with patch("evaluation.run_evaluation._run_with_retry", return_value=(MagicMock(success=True, plan_steps=None), 1)) as mock_run, \
             patch("evaluation.run_evaluation.q2_metrics", return_value={}):
            result = _run_query(
                "log_1", "case1", self._spec(deadline=None), "(define (domain test))",
                api, cfg, _make_serialized(), tmp_path / "failures", prefix,  # empty graph, would fail if checked
                MagicMock(), {}, pddl_dir=tmp_path / "pddl",
            )

        mock_run.assert_called_once()
        assert result.solvability != "skipped_unsatisfiable_by_construction"


# ---------------------------------------------------------------------------
# evaluate_log — current.json cache (skip parse/encode/optimizer when it
# already exists, unless force_rebuild)
# ---------------------------------------------------------------------------

class TestEvaluateLogCache:
    def test_uses_cache_when_current_json_exists(self, tmp_path):
        _write_cached_current_json(tmp_path)
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        with (
            _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1),
            _patch_cache_reconstruction(n_activities=9),
        ):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        api.build_network.assert_not_called()
        assert lr.pipeline_ok is True
        assert lr.n_activities == 9  # from the cache-reconstruction mock, not the api mock's build_network (3)
        assert lr.used_optimizer is False
        assert lr.search_summary is None

    def test_builds_fresh_when_no_current_json(self, tmp_path):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1):
            evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        api.build_network.assert_called_once()

    def test_force_rebuild_bypasses_cache(self, tmp_path):
        _write_cached_current_json(tmp_path)
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path, force_rebuild=True)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1):
            evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        api.build_network.assert_called_once()

    def test_cache_hit_still_produces_queries(self, tmp_path):
        _write_cached_current_json(tmp_path)
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        with (
            _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1),
            _patch_cache_reconstruction(),
        ):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        q1_records = [q for q in lr.queries if q.query_type == "Q1"]
        assert len(q1_records) == 1


# ---------------------------------------------------------------------------
# evaluate_log — search_summary / search_config.json when the optimizer runs
# ---------------------------------------------------------------------------

class TestEvaluateLogSearchSummary:
    def test_search_summary_and_search_config_written_when_trial_records_present(self, tmp_path):
        trial_a = MagicMock(score=1.0, trial_number=0)
        trial_b = MagicMock(score=4.2, trial_number=1)
        build_result = _make_build_result(trial_records=[trial_a, trial_b])
        api = _make_mock_api(build_result=build_result)
        cfg = _make_cfg(tmp_path, search_n_trials=2)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        assert lr.used_optimizer is True
        assert lr.search_summary == {"n_trials": 2, "best_score": 4.2, "best_trial_number": 1}
        search_config_path = tmp_path / "log_1" / "search_config.json"
        assert search_config_path.exists()
        saved = json.loads(search_config_path.read_text(encoding="utf-8"))
        assert "ignored_attributes" in saved
        assert isinstance(saved["ignored_attributes"], list)

    def test_no_optimizer_means_no_search_summary(self, tmp_path):
        build_result = _make_build_result(trial_records=None)
        api = _make_mock_api(build_result=build_result)
        cfg = _make_cfg(tmp_path, use_optimizer=False)

        with _patched_evaluate_log(api=api, cfg=cfg, tmp_path=tmp_path, n_test=1):
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        assert lr.used_optimizer is False
        assert lr.search_summary is None
        assert not (tmp_path / "log_1" / "search_config.json").exists()


# ---------------------------------------------------------------------------
# main — resume behaviour
# ---------------------------------------------------------------------------

class TestMainResume:
    def test_resume_skips_existing_results(self, tmp_path):
        # Prepare a pre-existing result.json for log_1
        log_dir = tmp_path / "log_1"
        log_dir.mkdir(parents=True)
        (log_dir / "result.json").write_text(
            json.dumps(_make_log_result_data("log_1")), encoding="utf-8"
        )

        selection_row = MagicMock()
        selection_row.get = lambda k, default=None: {
            "Event Log ID": "log_1",
            "Event Log Name": "Test Log",
        }.get(k, default)

        import pandas as pd
        selection = pd.DataFrame([{
            "Event Log ID": "log_1",
            "Event Log Name": "Test Log",
        }])

        args = build_parser().parse_args([
            "--log-id", "1",
            "--output-dir", str(tmp_path),
            "--resume",
            "--metadata", str(tmp_path / "meta.csv"),
        ])

        with (
            patch("evaluation.run_evaluation.get_log_selection", return_value=selection),
            patch("evaluation.run_evaluation.write_log_summary"),
            patch("evaluation.run_evaluation.evaluate_log") as mock_eval,
        ):
            main(args)
            mock_eval.assert_not_called()

    def test_resume_does_not_skip_when_result_absent(self, tmp_path):
        import pandas as pd
        selection = pd.DataFrame([{
            "Event Log ID": "log_1",
            "Event Log Name": "Test Log",
        }])

        args = build_parser().parse_args([
            "--log-id", "1",
            "--output-dir", str(tmp_path),
            "--resume",
            "--metadata", str(tmp_path / "meta.csv"),
        ])

        lr = LogResult(
            log_id="log_1", log_name="Test Log", log_fmt="xes",
            n_train_cases=80, n_test_cases=20, n_activities=5,
            pipeline_ok=True, pipeline_error=None, queries=[],
        )

        with (
            patch("evaluation.run_evaluation.get_log_selection", return_value=selection),
            patch("evaluation.run_evaluation.download_if_needed", return_value=(tmp_path / "log.xes", "xes")),
            patch("evaluation.run_evaluation.evaluate_log", return_value=lr) as mock_eval,
            patch("evaluation.run_evaluation.write_log_result"),
            patch("evaluation.run_evaluation.write_log_summary"),
        ):
            main(args)
            mock_eval.assert_called_once()


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_cli_parses_cost_weight(self):
        args = build_parser().parse_args(["--log-id", "1", "--cost-weight", "0.05"])
        assert args.cost_weight == pytest.approx(0.05)

    def test_cli_cost_weight_default(self):
        args = build_parser().parse_args(["--log-id", "1"])
        assert args.cost_weight == pytest.approx(0.001)

    def test_cli_parses_csv_mapping_json(self):
        mapping = '{"case_id": "col_a", "activity": "col_b"}'
        args = build_parser().parse_args(["--log-id", "1", "--csv-mapping", mapping])
        assert args.csv_mapping == mapping

    def test_cli_parses_test_pct(self):
        args = build_parser().parse_args(["--log-id", "1", "--test-pct", "0.3"])
        assert args.test_pct == pytest.approx(0.3)

    def test_cli_parses_min_test_cases(self):
        args = build_parser().parse_args(["--log-id", "1", "--min-test-cases", "5"])
        assert args.min_test_cases == 5

    def test_cli_parses_max_test_cases(self):
        args = build_parser().parse_args(["--log-id", "1", "--max-test-cases", "150"])
        assert args.max_test_cases == 150

    def test_cli_test_pct_default(self):
        args = build_parser().parse_args(["--log-id", "1"])
        assert args.test_pct == pytest.approx(0.2)

    def test_cli_parses_resume_flag(self):
        args = build_parser().parse_args(["--log-id", "1", "--resume"])
        assert args.resume is True

    def test_cli_parses_force_download_flag(self):
        args = build_parser().parse_args(["--log-id", "1", "--force-download"])
        assert args.force_download is True

    def test_cli_parses_force_rebuild_flag(self):
        args = build_parser().parse_args(["--log-id", "1", "--force-rebuild"])
        assert args.force_rebuild is True

    def test_cli_force_rebuild_default_false(self):
        args = build_parser().parse_args(["--log-id", "1"])
        assert args.force_rebuild is False

    def test_cli_parses_log_id(self):
        args = build_parser().parse_args(["--log-id", "55"])
        assert args.log_id == 55

    def test_cli_log_id_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_cli_default_algorithm(self):
        args = build_parser().parse_args(["--log-id", "1"])
        assert args.algorithm == "inductive"

    def test_cli_optimizer_enabled_by_default(self):
        args = build_parser().parse_args(["--log-id", "1"])
        assert args.no_optimizer is False

    def test_cli_no_optimizer_flag(self):
        args = build_parser().parse_args(["--log-id", "1", "--no-optimizer"])
        assert args.no_optimizer is True

    def test_cli_parses_search_n_trials(self):
        args = build_parser().parse_args(["--log-id", "1", "--search-n-trials", "50"])
        assert args.search_n_trials == 50

    def test_cli_search_n_trials_default(self):
        args = build_parser().parse_args(["--log-id", "1"])
        assert args.search_n_trials == 30

    def test_cli_parses_score_weights(self):
        args = build_parser().parse_args([
            "--log-id", "1",
            "--w-det-xor", "0.9", "--w-det-eff", "0.7", "--w-fb-xor", "1.5", "--w-fb-eff", "1.2",
            "--w-prune-xor", "4.0", "--w-xor", "0.8", "--w-eff", "0.9", "--w-dup", "0.3",
        ])
        assert args.w_det_xor == pytest.approx(0.9)
        assert args.w_det_eff == pytest.approx(0.7)
        assert args.w_fb_xor == pytest.approx(1.5)
        assert args.w_fb_eff == pytest.approx(1.2)
        assert args.w_prune_xor == pytest.approx(4.0)
        assert args.w_xor == pytest.approx(0.8)
        assert args.w_eff == pytest.approx(0.9)
        assert args.w_dup == pytest.approx(0.3)

    def test_cli_score_weight_defaults_match_score_weights(self):
        from network_search.scoring import ScoreWeights
        args = build_parser().parse_args(["--log-id", "1"])
        defaults = ScoreWeights()
        assert args.w_det_xor == pytest.approx(defaults.w_det_xor)
        assert args.w_det_eff == pytest.approx(defaults.w_det_eff)
        assert args.w_fb_xor == pytest.approx(defaults.w_fb_xor)
        assert args.w_fb_eff == pytest.approx(defaults.w_fb_eff)
        assert args.w_prune_xor == pytest.approx(defaults.w_prune_xor)
        assert args.w_xor == pytest.approx(defaults.w_xor)
        assert args.w_eff == pytest.approx(defaults.w_eff)
        assert args.w_dup == pytest.approx(defaults.w_dup)


# ---------------------------------------------------------------------------
# CSV mapping interactive fallback
# ---------------------------------------------------------------------------

class TestCSVMappingInteractiveFallback:
    def test_prompt_reads_csv_headers(self, tmp_path):
        csv_file = tmp_path / "log.csv"
        csv_file.write_text("case_id,activity,timestamp\n1,A,2020-01-01\n", encoding="utf-8")

        inputs = iter(["case_id", "activity", "timestamp", ""])
        with patch("builtins.input", side_effect=inputs):
            mapping = _prompt_csv_mapping(csv_file)

        assert mapping["case_id"] == "case_id"
        assert mapping["activity"] == "activity"
        assert mapping["timestamp"] == "timestamp"
        assert mapping["lifecycle"] is None

    def test_prompt_blank_maps_to_none(self, tmp_path):
        csv_file = tmp_path / "log.csv"
        csv_file.write_text("col_a,col_b\n1,A\n", encoding="utf-8")

        with patch("builtins.input", return_value=""):
            mapping = _prompt_csv_mapping(csv_file)

        assert all(v is None for v in mapping.values())

    def test_csv_mapping_interactive_used_in_main_when_absent(self, tmp_path):
        import pandas as pd
        selection = pd.DataFrame([{"Event Log ID": "log_1", "Event Log Name": "Test Log"}])
        csv_log = tmp_path / "log.csv"
        csv_log.write_text("col_a,col_b\n", encoding="utf-8")

        lr = LogResult(
            log_id="log_1", log_name="Test Log", log_fmt="csv",
            n_train_cases=80, n_test_cases=20, n_activities=5,
            pipeline_ok=True, pipeline_error=None, queries=[],
        )

        args = build_parser().parse_args(["--log-id", "1", "--output-dir", str(tmp_path),
                                          "--metadata", str(tmp_path / "meta.csv")])

        with (
            patch("evaluation.run_evaluation.get_log_selection", return_value=selection),
            patch("evaluation.run_evaluation.download_if_needed", return_value=(csv_log, "csv")),
            patch("evaluation.run_evaluation.evaluate_log", return_value=lr),
            patch("evaluation.run_evaluation.write_log_result"),
            patch("evaluation.run_evaluation.write_log_summary"),
            patch("evaluation.run_evaluation._prompt_csv_mapping", return_value={"case_id": "col_a"}) as mock_prompt,
        ):
            main(args)
            mock_prompt.assert_called_once_with(csv_log)
