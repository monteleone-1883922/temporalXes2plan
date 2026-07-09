"""End-to-end test for Pipeline.run(search=True) — the integration point
decided in docs/network_improvement_loop_plan.md §0: network_search runs
inside Pipeline.run(), the single entry point already shared by the CLI
(src/main.py) and the web import flow (web/api.py::_pipeline_thread).

Marked @pytest.mark.integration (skipped by default, run with -m
integration) — same convention as tests/parsing/test_parser.py and
tests/network_search/test_runner.py, whose synthetic XOR log this test
reuses.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest.mock import patch

import pm4py
import pytest
from pm4py.objects.log.obj import Event, EventLog, Trace

import pipeline as pipeline_module
from models import AnalysisConfig
from pipeline import Pipeline


def _ts(offset_seconds: int = 0) -> datetime:
    base = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _event(name: str, offset: int = 0, **attrs) -> Event:
    e = Event()
    e["concept:name"] = name
    e["time:timestamp"] = _ts(offset)
    for k, v in attrs.items():
        e[k] = v
    return e


def _trace(case_id: str, events: List[Event]) -> Trace:
    t = Trace()
    t.attributes["concept:name"] = case_id
    for e in events:
        t.append(e)
    return t


def _make_xor_log() -> EventLog:
    log = EventLog()
    for i in range(60):
        log.append(_trace(f"high_{i}", [
            _event("Register", 0, risk="high"),
            _event("Approve", 10, outcome="approved"),
            _event("Close", 20, status="done"),
        ]))
    for i in range(60):
        log.append(_trace(f"low_{i}", [
            _event("Register", 0, risk="low"),
            _event("Reject", 10, outcome="rejected"),
            _event("Close", 20, status="done"),
        ]))
    return log


@pytest.fixture(scope="module")
def xes_path(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("pipeline_search_integration")
    path = str(tmp / "xor_log.xes")
    pm4py.write_xes(_make_xor_log(), path)
    return path


@pytest.mark.integration
class TestPipelineRunWithSearch:
    def test_runs_end_to_end_and_writes_expected_files(self, xes_path, tmp_path):
        data_dir = str(tmp_path / "data")
        pddl_dir = str(tmp_path / "pddl")

        pddl_path = Pipeline().run(
            log_path=xes_path,
            data_dir=data_dir,
            pddl_output_dir=pddl_dir,
            coverage_percentage=0.8,
            search=True,
            search_n_trials=3,
            search_seed=0,
        )

        assert Path(pddl_path).exists()

        config_dir = Path(data_dir) / "xor_log"
        assert (config_dir / "current.json").exists()
        assert (config_dir / "original.json").exists()

    def test_network_search_trials_are_not_persisted(self, xes_path, tmp_path):
        data_dir = str(tmp_path / "data2")
        pddl_dir = str(tmp_path / "pddl2")

        Pipeline().run(
            log_path=xes_path,
            data_dir=data_dir,
            pddl_output_dir=pddl_dir,
            coverage_percentage=0.8,
            search=True,
            search_n_trials=3,
            search_seed=0,
        )

        trials_path = Path(data_dir) / "xor_log" / "network_search_trials.jsonl"
        assert not trials_path.exists()

    def test_final_network_covers_both_xor_branches_from_the_full_log(self, xes_path, tmp_path):
        # The search's own train/test split only scores candidates -- the
        # network actually persisted must still be built on the complete
        # log, so both XOR branches (Approve and Reject) must survive, not
        # just whichever happened to dominate some particular split.
        data_dir = str(tmp_path / "data3")
        pddl_dir = str(tmp_path / "pddl3")

        Pipeline().run(
            log_path=xes_path,
            data_dir=data_dir,
            pddl_output_dir=pddl_dir,
            coverage_percentage=0.8,
            search=True,
            search_n_trials=2,
            search_seed=0,
        )

        current = json.loads((Path(data_dir) / "xor_log" / "current.json").read_text(encoding="utf-8"))
        transitions = current.get("transitions", {})
        assert "approve" in transitions
        assert "reject" in transitions

    def test_without_search_behaves_exactly_as_before(self, xes_path, tmp_path):
        # search=False (default) must be unaffected by this change.
        data_dir = str(tmp_path / "data4")
        pddl_dir = str(tmp_path / "pddl4")

        pddl_path = Pipeline().run(
            log_path=xes_path,
            data_dir=data_dir,
            pddl_output_dir=pddl_dir,
            coverage_percentage=0.8,
        )

        assert Path(pddl_path).exists()
        config_dir = Path(data_dir) / "xor_log"
        assert (config_dir / "current.json").exists()
        assert not (config_dir / "network_search_trials.jsonl").exists()

    def test_search_uses_config_as_base_for_untouched_fields(self, xes_path, tmp_path):
        # GUI optimizer toggle: fields the search doesn't tune (e.g.
        # attr_precondition_min_firings) must come from the caller-supplied
        # `config`, not AnalysisConfig()'s hardcoded default -- spy on
        # find_best_config to capture what base_config it actually received.
        # Deliberately avoids replay_engine="alignments" here: it exercises a
        # different code path that deterministically hits the pre-existing,
        # unrelated prepared_graph_utils.py bug this test suite is already
        # blocked on elsewhere (see the other failing tests in this class).
        data_dir = str(tmp_path / "data5")
        pddl_dir = str(tmp_path / "pddl5")
        custom = AnalysisConfig(attr_precondition_min_firings=99, ignored_attributes={"custom_attr"})

        captured = {}
        original = pipeline_module.find_best_config

        def _spy(*args, **kwargs):
            captured["base_config"] = kwargs.get("base_config")
            return original(*args, **kwargs)

        with patch("pipeline.find_best_config", side_effect=_spy):
            Pipeline().run(
                log_path=xes_path,
                data_dir=data_dir,
                pddl_output_dir=pddl_dir,
                coverage_percentage=0.8,
                search=True,
                search_n_trials=2,
                search_seed=0,
                config=custom,
            )

        assert captured["base_config"] is custom
        assert captured["base_config"].attr_precondition_min_firings == 99
        assert captured["base_config"].ignored_attributes == {"custom_attr"}
