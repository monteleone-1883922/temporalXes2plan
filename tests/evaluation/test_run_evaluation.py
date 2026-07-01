"""Unit tests for evaluation.run_evaluation."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from evaluation.run_evaluation import (
    EvalConfig,
    _load_log_result,
    _prompt_csv_mapping,
    build_parser,
    evaluate_log,
    main,
)
from evaluation.report_generator import LogResult, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(output_dir: Path, **overrides) -> EvalConfig:
    defaults = dict(
        log_ids=None,
        n_test_cases=5,
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
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _make_log_result_data(log_id="log_1", pipeline_ok=True):
    return {
        "log_id": log_id,
        "log_name": "Test Log",
        "log_fmt": "xes",
        "n_train_cases": 80,
        "n_test_cases": 20,
        "n_activities": 5,
        "pipeline_ok": pipeline_ok,
        "pipeline_error": None,
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


def _make_mock_api(missing_timestamp=False):
    api = MagicMock()
    vr = MagicMock()
    vr.missing_timestamp = missing_timestamp
    api.validate_log.return_value = vr
    api.parse.return_value = MagicMock()
    api.parse.return_value.petri_net_model.activities = {"a", "b", "c"}
    api.build_domain.return_value = "(define (domain test))"
    api.build_domain_with_variant_effects.return_value = ("(define (domain test))", {})
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


# ---------------------------------------------------------------------------
# evaluate_log — missing timestamps
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
# evaluate_log — pipeline failures
# ---------------------------------------------------------------------------

class TestEvaluateLogPipelineFailures:
    def test_parse_exception_marks_failed(self, tmp_path):
        api = _make_mock_api()
        api.parse.side_effect = RuntimeError("parse failed")
        cfg = _make_cfg(tmp_path)

        with patch("evaluation.run_evaluation._split_log") as mock_split:
            mock_split.return_value = _make_mock_tts()
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        assert lr.pipeline_ok is False
        assert "parse failed" in lr.pipeline_error

    def test_build_domain_exception_marks_failed(self, tmp_path):
        api = _make_mock_api()
        api.build_domain_with_variant_effects.side_effect = RuntimeError("domain failed")
        cfg = _make_cfg(tmp_path)

        with patch("evaluation.run_evaluation._split_log") as mock_split:
            mock_split.return_value = _make_mock_tts()
            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        assert lr.pipeline_ok is False
        assert "domain failed" in lr.pipeline_error


# ---------------------------------------------------------------------------
# evaluate_log — successful flow
# ---------------------------------------------------------------------------

class TestEvaluateLogSuccess:
    def _run(self, tmp_path, n_test=2, mock_prefix=True):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        prefix = MagicMock()
        prefix.prefix_ratio = 0.5
        prefix.prefix_events = [MagicMock(), MagicMock()]
        prefix.init_places = ["p_start"]
        prefix.init_effects = []
        prefix.full_duration_s = 3600.0
        prefix.prefix_duration_s = 1000.0
        prefix.final_event_attributes = {}

        q1_spec = MagicMock()
        q1_spec.query_type = "Q1"
        q1_spec.init_places = ["p_start"]
        q1_spec.init_effects = []
        q1_spec.goal_sop = [[]]
        q1_spec.metric = "minimize_weighted"
        q1_spec.cost_weight = 0.001
        q1_spec.require_completion = True
        q1_spec.deadline = None

        with (
            patch("evaluation.run_evaluation._split_log") as mock_split,
            patch("evaluation.run_evaluation._sample_prefix") as mock_prefix_fn,
            patch("evaluation.run_evaluation.build_q1", return_value=q1_spec),
            patch("evaluation.run_evaluation.build_q2", return_value=None),
            patch("evaluation.run_evaluation.build_q3", return_value=None),
            patch("evaluation.run_evaluation.q1_metrics", return_value={"solved": True, "weighted_objective": 1.0}),
            patch("evaluation.run_evaluation.validate_plan") as mock_validate,
            patch("evaluation.run_evaluation.serialize_parse_result", return_value={"transitions": {}, "attribute_catalog": {}, "metadata": {"end_place": "p_end"}, "graph": {}}),
        ):
            mock_split.return_value = _make_mock_tts(n_test=n_test)
            mock_prefix_fn.return_value = prefix if mock_prefix else None
            vr = MagicMock()
            vr.valid = True
            vr.attribute_checked = True
            vr.error_step = None
            vr.error_action = None
            vr.error_reason = None
            vr.steps_executed = 1
            mock_validate.return_value = vr

            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        return lr

    def test_pipeline_ok_true_on_success(self, tmp_path):
        lr = self._run(tmp_path)
        assert lr.pipeline_ok is True

    def test_n_activities_from_parse_result(self, tmp_path):
        lr = self._run(tmp_path)
        assert lr.n_activities == 3  # {"a", "b", "c"}

    def test_domain_pddl_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "log_1" / "domain.pddl").exists()

    def test_skipped_trace_produces_no_queries(self, tmp_path):
        lr = self._run(tmp_path, n_test=2, mock_prefix=False)
        assert lr.queries == []

    def test_q3_skipped_produces_skipped_record(self, tmp_path):
        api = _make_mock_api()
        cfg = _make_cfg(tmp_path)

        prefix = MagicMock()
        prefix.prefix_ratio = 0.5
        prefix.prefix_events = [MagicMock()]
        prefix.init_places = ["p_start"]
        prefix.init_effects = []
        prefix.full_duration_s = 3600.0
        prefix.prefix_duration_s = 1000.0
        prefix.final_event_attributes = {}

        q_spec = MagicMock()
        q_spec.query_type = "Q1"
        q_spec.init_places = ["p_start"]
        q_spec.init_effects = []
        q_spec.goal_sop = [[]]
        q_spec.metric = "minimize_weighted"
        q_spec.cost_weight = 0.001
        q_spec.require_completion = True
        q_spec.deadline = None

        with (
            patch("evaluation.run_evaluation._split_log") as mock_split,
            patch("evaluation.run_evaluation._sample_prefix", return_value=prefix),
            patch("evaluation.run_evaluation.build_q1", return_value=q_spec),
            patch("evaluation.run_evaluation.build_q2", return_value=None),
            patch("evaluation.run_evaluation.build_q3", return_value=None),
            patch("evaluation.run_evaluation.q1_metrics", return_value={"solved": True, "weighted_objective": 1.0}),
            patch("evaluation.run_evaluation.validate_plan") as mock_vp,
            patch("evaluation.run_evaluation.serialize_parse_result", return_value={
                "transitions": {}, "attribute_catalog": {},
                "metadata": {"end_place": "p_end"}, "graph": {},
            }),
        ):
            mock_split.return_value = _make_mock_tts(n_test=1)
            vr = MagicMock()
            vr.valid = True
            vr.error_step = None
            vr.error_action = None
            vr.error_reason = None
            vr.steps_executed = 1
            mock_vp.return_value = vr

            lr = evaluate_log("log_1", "Test", tmp_path / "log.xes", "xes", tmp_path, api, cfg)

        q3_records = [q for q in lr.queries if q.query_type == "Q3"]
        assert len(q3_records) == 1
        assert q3_records[0].solvability == "skipped_no_attributes"


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
            "--output-dir", str(tmp_path),
            "--resume",
            "--metadata", str(tmp_path / "meta.csv"),
        ])

        with (
            patch("evaluation.run_evaluation.get_log_selection", return_value=selection),
            patch("evaluation.run_evaluation.write_cross_log_summary"),
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
            patch("evaluation.run_evaluation.write_cross_log_summary"),
        ):
            main(args)
            mock_eval.assert_called_once()


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_cli_parses_cost_weight(self):
        args = build_parser().parse_args(["--cost-weight", "0.05"])
        assert args.cost_weight == pytest.approx(0.05)

    def test_cli_cost_weight_default(self):
        args = build_parser().parse_args([])
        assert args.cost_weight == pytest.approx(0.001)

    def test_cli_parses_csv_mapping_json(self):
        mapping = '{"case_id": "col_a", "activity": "col_b"}'
        args = build_parser().parse_args(["--csv-mapping", mapping])
        assert args.csv_mapping == mapping

    def test_cli_parses_n_test_cases(self):
        args = build_parser().parse_args(["--n-test-cases", "100"])
        assert args.n_test_cases == 100

    def test_cli_parses_resume_flag(self):
        args = build_parser().parse_args(["--resume"])
        assert args.resume is True

    def test_cli_parses_force_download_flag(self):
        args = build_parser().parse_args(["--force-download"])
        assert args.force_download is True

    def test_cli_parses_log_ids(self):
        args = build_parser().parse_args(["--log-ids", "A,B,C"])
        assert args.log_ids == "A,B,C"

    def test_cli_default_algorithm(self):
        args = build_parser().parse_args([])
        assert args.algorithm == "inductive"


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

        args = build_parser().parse_args(["--output-dir", str(tmp_path),
                                          "--metadata", str(tmp_path / "meta.csv")])

        with (
            patch("evaluation.run_evaluation.get_log_selection", return_value=selection),
            patch("evaluation.run_evaluation.download_if_needed", return_value=(csv_log, "csv")),
            patch("evaluation.run_evaluation.evaluate_log", return_value=lr),
            patch("evaluation.run_evaluation.write_log_result"),
            patch("evaluation.run_evaluation.write_cross_log_summary"),
            patch("evaluation.run_evaluation._prompt_csv_mapping", return_value={"case_id": "col_a"}) as mock_prompt,
        ):
            main(args)
            mock_prompt.assert_called_once_with(csv_log)
