"""End-to-end run of the data layer (prepare -> split -> model -> prefixes)
on a small synthetic log. Builds a real network (TPE with 2 trials), so it
is an integration test: run with `pytest -m integration`."""

import json
import shutil

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
    # unit_costs asked for explicitly, so the test does not depend on the default
    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                          search_n_trials=2, unit_costs=True)
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
    # how the model was made (the default is the paper's: POWL)
    meta = json.loads((model_dir / "model_meta.json").read_text())
    assert meta == {"split_hash": split_obj.split_hash, "discovery_algorithm": "powl", "seed": 42,
                    "search_n_trials": 2, "coverage_percentage": cfg.coverage_percentage, "unit_costs": True}
    # RQ1 uses unit costs: every action of the domain increases total-cost by exactly 1
    import re
    domain_text = (model_dir / "domain.pddl").read_text()
    costs = re.findall(r"\(increase \(total-cost\) ([0-9.]+)\)", domain_text)
    assert costs and set(costs) == {"1.0000"}
    # ... and none is missing: exactly one increase per action (durative or not)
    assert len(costs) == len(re.findall(r"\(:(?:durative-)?action ", domain_text))
    from paper_eval.build_model import load_model
    assert load_model(model_dir).domain.unit_costs is True  # the rebuilt domain agrees with the file
    assert split["removed_spanning_ids"] == []  # the synthetic cases never overlap the split point
    # tables written as the model is built, ready to be queried
    assert json.loads((model_dir / "split.json").read_text()) == split  # the folder's own copy
    attributes = px.read_jsonl(model_dir / "attributes.jsonl")
    # the synthetic log has no data attributes: the table is empty, not missing
    assert attributes == []
    durations = px.read_jsonl(model_dir / "transition_durations.jsonl")
    assert {r["activity_name"] for r in durations} >= {"A", "D", "E"}

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

    # the real files of the run, read by the SQL views
    from paper_eval import query as q
    con = q.connect(model_dir)
    assert con.execute("SELECT n_train_fit, n_val, n_test, n_removed_spanning, discovery_algorithm, seed, unit_costs "
                       "FROM runs").fetchone() == (34, 8, 18, 0, "powl", 42, True)
    assert con.execute("SELECT count(*) FROM case_split").fetchone() == (60,)
    assert con.execute("SELECT count(*) FROM attributes").fetchone() == (len(attributes),)
    assert con.execute("SELECT count(*) FROM prefixes").fetchone() == (len(records),)
    # nothing planned yet: every case is there, without an outcome
    assert con.execute("SELECT count(*), count(status) FROM cases").fetchone() == (len(records), 0)

    runner.run_volume("toy", cfg, paths)
    volume = json.loads(paths.volume_json.read_text())
    assert volume["n_prefixes"] == 36
    assert volume["n_distinct_activity_prefixes"] == 4


def test_run_split_takes_the_cases_spanning_the_split_point_out_of_train(tmp_path):
    """20 cases, one starting per day and lasting 3 days: the test period starts on day 14, so the
    training cases that end after it (those starting on days 12 and 13) are removed."""
    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                          drop_spanning_train=True)  # asked for explicitly, off by default
    paths = runner.Paths("toy", cfg)
    paths.log_dir.mkdir(parents=True)
    cfg.data_dir.mkdir(parents=True)
    traces = [
        make_trace(
            make_event("A", lifecycle="complete", offset=i * DAY),
            make_event("B", lifecycle="complete", offset=i * DAY + 36 * HOUR),
            make_event("C", lifecycle="complete", offset=i * DAY + 3 * DAY),
            case_id=f"case_{i:03d}",
        )
        for i in range(20)
    ]
    log = preprocess(make_log(*traces), cfg.min_trace_len)
    write_traces_xes(log, list(log), paths.log_xes)

    runner.run_split("toy", cfg, paths, force=False)
    split = json.loads(paths.split_json.read_text())
    # case_011 ends exactly when the test period starts: kept
    assert split["removed_spanning_ids"] == ["case_012", "case_013"]
    assert len(split["test_ids"]) == 6
    assert (len(split["train_fit_ids"]), len(split["val_ids"])) == (10, 2)
    assert split["params"]["drop_spanning_train"] is True

    # the option can be switched off from the configuration: another split, another hash
    other = PaperEvalConfig(data_dir=cfg.data_dir, results_dir=tmp_path / "results_keep",
                            drop_spanning_train=False)
    other_paths = runner.Paths("toy", other)
    other_paths.log_dir.mkdir(parents=True)
    runner.run_split("toy", other, other_paths, force=False)
    kept = json.loads(other_paths.split_json.read_text())
    assert kept["removed_spanning_ids"] == [] and kept["split_hash"] != split["split_hash"]


@pytest.mark.integration
def test_attribute_lists_and_filters_end_to_end(tmp_path):
    """build_model with a LogSpec: the log's ignore list is applied, keep protects
    from the automatic filters, and attribute_filter.json reports what happened."""
    from paper_eval.build_model import build_model, config_from_dict, config_to_dict, load_model
    from paper_eval.config import LogSpec
    from paper_eval.split import read_split

    # event_id: unique per event (noise, rule N2/N3); junk: constant but kept by the
    # user's keep list; mine: ignored by name; risk: the one real attribute.
    traces, n = [], 0
    for i, trace in enumerate(_synthetic_log()):
        for event in trace:
            event["event_id"] = f"e{n}"
            event["junk"] = "same"
            event["mine"] = f"m{n % 3}"
            event["risk"] = "high" if i % 3 else "low"
            n += 1
        traces.append(trace)
    log = make_log(*traces)

    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results", search_n_trials=2)
    paths = runner.Paths("toy", cfg)
    paths.log_dir.mkdir(parents=True)
    cfg.data_dir.mkdir(parents=True)
    log = preprocess(log, cfg.min_trace_len)
    write_traces_xes(log, list(log), paths.log_xes)
    runner.run_split("toy", cfg, paths, force=False)

    spec = LogSpec("toy", "toy.xes", "", False, ignore_attributes=("mine",), keep_attributes=("junk",))
    split = read_split(paths.split_json)
    model_dir = build_model(paths.log_xes, split, cfg, paths.model_dir(split.split_hash), log_spec=spec)

    report = json.loads((model_dir / "attribute_filter.json").read_text())
    assert report["ignored_by_user"] == ["mine"]
    assert report["kept_by_user"] == ["junk"]
    assert "event_id" in report["noise_filter"]            # found by the noise filter
    assert "junk" not in report["noise_filter"]            # constant, but protected by keep
    assert "mine" not in report["noise_filter"]            # ignored: not reported again
    assert report["attribute_catalog_size_after"] <= report["attribute_catalog_size_before"]

    # the saved config round-trips with the lists (keep_attributes goes through JSON as a list)
    saved = json.loads((model_dir / "analysis_config.json").read_text())
    assert saved["keep_attributes"] == ["junk"] and "mine" in saved["ignored_attributes"]
    assert config_from_dict(saved).keep_attributes == ("junk",)
    catalog = load_model(model_dir).prepared.attribute_catalog
    assert "mine" not in catalog and "event_id" not in catalog

    # attributes.jsonl: every attribute with its fate, and the numbers agree with the report
    from paper_eval.prefixes import read_jsonl
    rows = {r["attribute"]: r for r in read_jsonl(model_dir / "attributes.jsonl")}
    assert rows["event_id"]["status"] == "removed_noise"
    assert rows["event_id"]["type"] == "categorical" and rows["event_id"]["n_distinct_observed"] > 50
    assert rows["event_id"]["n_values"] is None and rows["event_id"]["reason"]
    assert rows["mine"]["status"] == "ignored_by_user" and rows["mine"]["type"] is None
    # constant, protected by keep: no effect or guard ever uses it, so it is not in the
    # catalog; the table says so instead of leaving it out
    assert rows["junk"]["kept_by_user"] is True
    assert rows["junk"]["status"] == "removed_cascade" and rows["junk"]["type"] == "categorical"
    analysed = [r for r in rows.values() if r["status"] in ("kept", "removed_unused")]
    assert len(analysed) == report["attribute_catalog_size_before"]
    assert sum(r["status"] == "kept" for r in analysed) == report["attribute_catalog_size_after"]
    assert {a for a, r in rows.items() if r["status"] == "removed_unused"} == set(report["unused_attribute_filter"])
    # whatever was removed as unused still has its type and values
    assert all(r["type"] and r["possible_values"] is not None
               for r in rows.values() if r["status"] == "removed_unused")

    # start-of-trace attributes: the report lists them with the shares seen in training, and
    # every attribute row has the two columns (an ignored attribute was never seen: share None)
    assert isinstance(report["start_of_trace"], list)
    # risk and junk have one value per trace, so the first event sets them and nothing changes
    assert report["start_of_trace_share"]["risk"] == 1.0
    assert report["start_of_trace_share"]["junk"] == 1.0
    assert "event_id" not in report["start_of_trace_share"]  # removed as noise before the preprocessor
    # the model carries its own list in current.json (only attributes a guard reads can be there)
    start_attrs = json.loads((model_dir / "current.json").read_text())["start_of_trace_attributes"]
    assert set(start_attrs) <= set(report["start_of_trace"]) and set(start_attrs) <= set(catalog)
    assert "mine" not in report["start_of_trace_share"]
    assert all(isinstance(r["start_of_trace"], bool) for r in rows.values())
    assert rows["mine"]["start_of_trace_share"] is None
    assert {a for a, r in rows.items() if r["start_of_trace"]} == set(report["start_of_trace"])


@pytest.mark.integration
def test_plan_stats_and_report_chain_on_a_real_model(tmp_path):
    """Model and instances are real; only OPTIC is replaced by a fake. Checks that
    the plan, stats and report phases fit together on what the earlier phases wrote."""
    from paper_eval import model_stats as ms
    from paper_eval import plan as pl
    from paper_eval import report as rp
    from paper_eval.build_model import build_model
    from paper_eval.split import read_split

    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results", search_n_trials=2)
    paths = runner.Paths("toy", cfg)
    paths.log_dir.mkdir(parents=True)
    cfg.data_dir.mkdir(parents=True)
    log = preprocess(_synthetic_log(), cfg.min_trace_len)
    write_traces_xes(log, list(log), paths.log_xes)
    runner.run_split("toy", cfg, paths, force=False)
    split = read_split(paths.split_json)
    model_dir = build_model(paths.log_xes, split, cfg, paths.model_dir(split.split_hash))
    runner.run_prefixes("toy", cfg, paths, force=False)
    instances = px.read_jsonl(model_dir / "instances.jsonl")
    assert {i["goal"] for i in instances} == {"completion", "deadline"}

    problems = []

    def fake_optic(pddl_dir, timeout, memory_mb, stop_at_first):
        # the problem the runner built for a REAL instance of the REAL model
        problems.append((pddl_dir / "problem.pddl").read_text())
        return {
            "success": True, "solvability": "solved", "plan_text": "0.000: (a)  [1.000]",
            "plan_actions": ["a"], "message": "ok",
            "metrics": {"duration": 1.0, "expanded_nodes": 3, "total_time": 0.01},
            "timings": {"grounding_s": 0.1, "first_plan_s": 0.3, "planning_s": 0.2, "wall_s": 0.4, "cpu_s": 0.01},
        }

    counts = pl.run_plan("toy", cfg, runner.Paths("toy", cfg), runner=fake_optic)
    rows = pl.read_results(model_dir / pl.RESULTS_FILE)
    assert len(rows) == len(instances) and sum(counts.values()) == len(instances)
    assert all("(:domain paper_eval)" in p and "(marked" in p for p in problems)
    # an instance whose deadline is already past is not sent to OPTIC
    assert counts["solved"] == len(problems) <= len(instances)

    summary = rp.run_report(model_dir)
    assert set(summary) <= {"completion", "deadline"} and "completion" in summary
    assert (model_dir / rp.SUMMARY_MD).exists()

    stats = ms.run_stats(model_dir, cfg.data_dir / "toy.stats.json")
    assert (model_dir / ms.STATS_FILE).exists()
    net = stats["petri_net"]
    assert net["places"] > 0 and net["transitions"] == net["transitions_visible"] + net["transitions_silent"]
    assert stats["pddl"]["actions"] >= 1 and stats["pddl"]["domain_bytes"] > 0
    assert stats["split"]["test"] == 18 and stats["search"]["trials"] == 2


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz is not installed")
def test_model_phase_draws_the_petri_net_and_redraws_it_for_a_cached_model(tmp_path):
    from paper_eval.build_model import PETRI_NET_PNG, PETRI_NET_SVG, build_model
    from paper_eval.split import read_split

    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results", search_n_trials=2)
    paths = runner.Paths("toy", cfg)
    paths.log_dir.mkdir(parents=True)
    cfg.data_dir.mkdir(parents=True)
    log = preprocess(_synthetic_log(), cfg.min_trace_len)
    write_traces_xes(log, list(log), paths.log_xes)
    runner.run_split("toy", cfg, paths, force=False)
    split = read_split(paths.split_json)
    model_dir = build_model(paths.log_xes, split, cfg, paths.model_dir(split.split_hash))

    png, svg = model_dir / PETRI_NET_PNG, model_dir / PETRI_NET_SVG
    assert png.stat().st_size > 0 and svg.stat().st_size > 0
    assert "<svg" in svg.read_text(encoding="utf-8")

    # a model built before pictures existed gets one without being rebuilt
    current_mtime = (model_dir / "current.json").stat().st_mtime_ns
    png.unlink()
    build_model(paths.log_xes, split, cfg, model_dir)
    assert png.stat().st_size > 0
    assert (model_dir / "current.json").stat().st_mtime_ns == current_mtime
