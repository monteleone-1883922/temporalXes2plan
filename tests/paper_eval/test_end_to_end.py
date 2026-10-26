"""End-to-end run of the data layer (prepare -> split -> model -> prefixes)
on a small synthetic log. Builds a real network (TPE with 2 trials), so it
is an integration test: run with `pytest -m integration`."""

import json

import pytest

from tests.helpers import make_event, make_log, make_trace

from core_utils import write_traces_xes
from paper_eval import prefixes as px
from paper_eval import run as runner
from paper_eval.config import PaperEvalConfig
from paper_eval.prepare import preprocess

HOUR = 3600
DAY = 24 * HOUR


def _synthetic_log(n_cases=60):
    traces = []
    for i in range(n_cases):
        start = i * DAY
        branch = "B" if i % 3 else "C"
        traces.append(make_trace(
            make_event("A", lifecycle="complete", offset=start),
            make_event(branch, lifecycle="complete", offset=start + (2 + i % 4) * HOUR),
            make_event("D", lifecycle="complete", offset=start + (5 + i % 7) * HOUR),
            make_event("E", lifecycle="complete", offset=start + (12 + i % 5) * HOUR),
            case_id=f"case_{i:03d}",
        ))
    return make_log(*traces)


@pytest.mark.integration
def test_data_layer_end_to_end(tmp_path):
    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                          search_n_trials=2)
    paths = runner.Paths("toy", cfg)
    paths.log_dir.mkdir(parents=True)
    cfg.data_dir.mkdir(parents=True)
    log = preprocess(_synthetic_log(), cfg.min_trace_len)
    write_traces_xes(log, list(log), paths.log_xes)

    runner.run_split("toy", cfg, paths, force=False)
    split = json.loads(paths.split_json.read_text())
    assert (len(split["train_fit_ids"]), len(split["val_ids"]), len(split["test_ids"])) == (34, 8, 18)

    from paper_eval.build_model import build_model
    from paper_eval.split import read_split
    split_obj = read_split(paths.split_json)
    model_dir = build_model(paths.log_xes, split_obj, cfg, paths.model_dir(split_obj.split_hash))
    assert (model_dir / "current.json").exists()
    assert (model_dir / "domain.pddl").exists()

    runner.run_prefixes("toy", cfg, paths, force=False)
    records = px.read_jsonl(model_dir / "prefixes.jsonl")
    instances = px.read_jsonl(model_dir / "instances.jsonl")
    completion = [r for r in records if r["goal"] == "completion"]
    # 18 test traces x prefixes k=2,3 (n=4).
    assert len(completion) == 36
    assert all(r["replay_ok"] for r in completion)
    # Activity prefixes are A,B / A,C / A,B,D / A,C,D: at most 4 completion
    # instances unless the discretized state splits them further.
    n_completion_instances = sum(1 for i in instances if i["goal"] == "completion")
    assert 4 <= n_completion_instances <= 36
    stats = json.loads((model_dir / "prefix_stats.json").read_text())
    assert stats["overall"]["n_prefixes"] == 36

    runner.run_volume("toy", cfg, paths)
    volume = json.loads(paths.volume_json.read_text())
    assert volume["n_prefixes"] == 36
    assert volume["n_distinct_activity_prefixes"] == 4
