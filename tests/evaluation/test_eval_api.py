"""Unit tests for evaluation.eval_api.EvalAPI.

EvalAPI is a thin facade: build_network() delegates entirely to
pipeline.Pipeline.build_network() (no logic of its own beyond argument
forwarding), serialize()/build_problem()/run_planner() wrap
encoding/planning primitives directly. Tests mock at those boundaries —
no real XES parsing, domain building, or planner invocation.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.eval_api import EvalAPI, PlanResult
from network_search.scoring import ScoreWeights


# ---------------------------------------------------------------------------
# build_network — delegates to Pipeline.build_network()
# ---------------------------------------------------------------------------

class TestBuildNetwork:
    def test_delegates_to_pipeline_build_network(self):
        sentinel = MagicMock(name="BuildResult")
        with patch("evaluation.eval_api.Pipeline") as MockPipeline:
            MockPipeline.return_value.build_network.return_value = sentinel
            result = EvalAPI().build_network("logs/sample.xes")

        assert result is sentinel
        MockPipeline.return_value.build_network.assert_called_once()

    def test_forwards_all_arguments(self):
        weights = ScoreWeights()
        with patch("evaluation.eval_api.Pipeline") as MockPipeline:
            EvalAPI().build_network(
                "logs/sample.xes",
                domain_name="my_domain",
                discovery_algorithm="powl",
                coverage_percentage=0.5,
                use_durative=True,
                use_costs=True,
                use_activity_classifier=True,
                config=None,
                search=True,
                search_n_trials=7,
                search_test_pct=0.3,
                search_seed=99,
                search_weights=weights,
                snapshot_dir="snap",
                discretizer_cache_path=Path("cache.json"),
                force_rediscretize=True,
            )
            kwargs = MockPipeline.return_value.build_network.call_args.kwargs

        assert kwargs["log_path"] == "logs/sample.xes"
        assert kwargs["domain_name"] == "my_domain"
        assert kwargs["discovery_algorithm"] == "powl"
        assert kwargs["coverage_percentage"] == pytest.approx(0.5)
        assert kwargs["use_durative"] is True
        assert kwargs["use_costs"] is True
        assert kwargs["use_activity_classifier"] is True
        assert kwargs["search"] is True
        assert kwargs["search_n_trials"] == 7
        assert kwargs["search_test_pct"] == pytest.approx(0.3)
        assert kwargs["search_seed"] == 99
        assert kwargs["search_weights"] is weights
        assert kwargs["snapshot_dir"] == "snap"
        assert kwargs["discretizer_cache_path"] == Path("cache.json")
        assert kwargs["force_rediscretize"] is True

    def test_defaults_are_search_disabled(self):
        with patch("evaluation.eval_api.Pipeline") as MockPipeline:
            EvalAPI().build_network("logs/sample.xes")
            kwargs = MockPipeline.return_value.build_network.call_args.kwargs

        assert kwargs["search"] is False
        assert kwargs["config"] is None
        assert kwargs["snapshot_dir"] is None


# ---------------------------------------------------------------------------
# serialize — PreparedDomainInput.to_dict() + start/end place metadata
# ---------------------------------------------------------------------------

class TestSerialize:
    def test_merges_metadata_onto_prepared_dict(self):
        prepared = MagicMock()
        prepared.to_dict.return_value = {"transitions": {}, "attribute_catalog": {}}
        parse_result = MagicMock()
        parse_result.start_place = "p_start"
        parse_result.end_place = "p_end"

        data = EvalAPI().serialize(prepared, parse_result)

        assert data["transitions"] == {}
        assert data["metadata"] == {"start_place": "p_start", "end_place": "p_end"}

    def test_does_not_mutate_prepared_dict_keys_other_than_metadata(self):
        prepared = MagicMock()
        prepared.to_dict.return_value = {"graph": {"nodes": [], "edges": []}}
        parse_result = MagicMock(start_place="a", end_place="b")

        data = EvalAPI().serialize(prepared, parse_result)

        assert data["graph"] == {"nodes": [], "edges": []}
        assert set(data.keys()) == {"graph", "metadata"}


# ---------------------------------------------------------------------------
# validate_log — dispatches XES vs CSV
# ---------------------------------------------------------------------------

class TestValidateLog:
    def test_xes_uses_pm4py_importer(self):
        with (
            patch("evaluation.eval_api.pm4py.objects.log.importer.xes.importer.apply") as mock_apply,
            patch("evaluation.eval_api.validate_event_log") as mock_validate,
        ):
            mock_apply.return_value = "log_obj"
            mock_validate.return_value = "validation_result"

            result = EvalAPI().validate_log("logs/sample.xes")

        mock_apply.assert_called_once_with("logs/sample.xes")
        mock_validate.assert_called_once_with("log_obj")
        assert result == "validation_result"

    def test_csv_requires_mapping(self):
        with pytest.raises(ValueError):
            EvalAPI().validate_log("logs/sample.csv", mapping=None)

    def test_csv_uses_csv_loader_with_mapping(self):
        mapping = {"case_id": "col_a", "activity": "col_b"}
        with (
            patch("evaluation.eval_api.csv_to_event_log") as mock_csv_loader,
            patch("evaluation.eval_api.validate_event_log") as mock_validate,
        ):
            mock_csv_loader.return_value = "log_obj"
            mock_validate.return_value = "validation_result"

            result = EvalAPI().validate_log("logs/sample.csv", mapping=mapping)

        mock_csv_loader.assert_called_once_with("logs/sample.csv", mapping)
        mock_validate.assert_called_once_with("log_obj")
        assert result == "validation_result"


# ---------------------------------------------------------------------------
# build_problem — thin wrapper over ProblemBuilder
# ---------------------------------------------------------------------------

class TestBuildProblem:
    def test_extracts_attribute_catalog_and_end_place_from_serialized(self):
        serialized = {
            "attribute_catalog": {"status": {"type": "categorical"}},
            "metadata": {"end_place": "p_end"},
        }
        with patch("evaluation.eval_api.ProblemBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = "(define (problem p))"
            result = EvalAPI().build_problem(
                serialized=serialized,
                init_places=["p_start"],
                goal_sop=[[]],
            )

        assert result == "(define (problem p))"
        kwargs = MockBuilder.return_value.build.call_args.kwargs
        assert kwargs["attribute_catalog"] == {"status": {"type": "categorical"}}
        assert kwargs["end_place"] == "p_end"
        assert kwargs["init_places"] == ["p_start"]
        assert kwargs["goal_sop"] == [[]]

    def test_missing_metadata_yields_none_end_place(self):
        serialized = {"attribute_catalog": {}}
        with patch("evaluation.eval_api.ProblemBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = "problem text"
            EvalAPI().build_problem(serialized=serialized, init_places=[], goal_sop=[[]])
            kwargs = MockBuilder.return_value.build.call_args.kwargs

        assert kwargs["end_place"] is None

    def test_init_effects_defaults_to_empty_list(self):
        with patch("evaluation.eval_api.ProblemBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = "problem text"
            EvalAPI().build_problem(serialized={}, init_places=[], goal_sop=[[]])
            kwargs = MockBuilder.return_value.build.call_args.kwargs

        assert kwargs["init_effects"] == []


# ---------------------------------------------------------------------------
# run_planner — writes domain/problem to a temp dir, maps PlannerResult -> PlanResult
# ---------------------------------------------------------------------------

class TestRunPlanner:
    def test_maps_successful_result(self):
        raw = MagicMock()
        raw.success = True
        raw.plan_actions = ["execute_a", "execute_b"]
        raw.metrics = {"cost": 3.0, "duration": 1.5}
        raw.solvability = "solved"
        raw.message = None
        raw.stdout = "out"
        raw.stderr = ""
        raw.planner = "optic"

        with patch("evaluation.eval_api._run_planner", return_value=raw) as mock_run:
            result = EvalAPI().run_planner("DOMAIN", "PROBLEM", planner="optic", timeout=30)

        assert isinstance(result, PlanResult)
        assert result.success is True
        assert result.plan_steps == ["execute_a", "execute_b"]
        assert result.cost == pytest.approx(3.0)
        assert result.duration_s == pytest.approx(1.5)
        assert result.solvability == "solved"
        assert result.error is None
        assert result.planner == "optic"
        mock_run.assert_called_once()

    def test_maps_failure_message_to_error(self):
        raw = MagicMock()
        raw.success = False
        raw.plan_actions = []
        raw.metrics = {}
        raw.solvability = "unsolvable_structural"
        raw.message = "no plan found"
        raw.stdout = ""
        raw.stderr = "err"
        raw.planner = "optic"

        with patch("evaluation.eval_api._run_planner", return_value=raw):
            result = EvalAPI().run_planner("DOMAIN", "PROBLEM")

        assert result.success is False
        assert result.error == "no plan found"

    def test_writes_domain_and_problem_to_temp_dir(self):
        captured = {}

        def _capture(pddl_dir, planner, options):
            captured["domain"] = (Path(pddl_dir) / "domain.pddl").read_text(encoding="utf-8")
            captured["problem"] = (Path(pddl_dir) / "problem.pddl").read_text(encoding="utf-8")
            raw = MagicMock()
            raw.success = True
            raw.plan_actions = []
            raw.metrics = {}
            raw.solvability = "solved"
            raw.message = None
            raw.stdout = ""
            raw.stderr = ""
            raw.planner = "optic"
            return raw

        with patch("evaluation.eval_api._run_planner", side_effect=_capture):
            EvalAPI().run_planner("MY DOMAIN TEXT", "MY PROBLEM TEXT")

        assert captured["domain"] == "MY DOMAIN TEXT"
        assert captured["problem"] == "MY PROBLEM TEXT"
