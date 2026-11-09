"""Phase "rq2": OPTIC optimising total-time on the completion instances, every plan saved.

The model and the instances are real (built from a synthetic log, as in test_end_to_end);
only OPTIC is replaced by a fake that returns what planning.optic.run_timed returns."""
import json

import pytest

from tests.helpers import make_event, make_log, make_trace

from core_utils import write_traces_xes
from paper_eval import prefixes as px
from paper_eval import rq2
from paper_eval import run as runner
from paper_eval.build_model import build_model, load_model
from paper_eval.config import PaperEvalConfig
from paper_eval.prepare import preprocess
from paper_eval.split import read_split

HOUR = 3600
DAY = 24 * HOUR


def _log(n_cases=60):
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


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rq2")
    cfg = PaperEvalConfig(data_dir=tmp_path / "data", results_dir=tmp_path / "results", search_n_trials=2)
    paths = runner.Paths("toy", cfg)
    paths.log_dir.mkdir(parents=True)
    cfg.data_dir.mkdir(parents=True)
    log = preprocess(_log(), cfg.min_trace_len)
    write_traces_xes(log, list(log), paths.log_xes)
    runner.run_split("toy", cfg, paths, force=False)
    split = read_split(paths.split_json)
    model_dir = build_model(paths.log_xes, split, cfg, paths.model_dir(split.split_hash))
    runner.run_prefixes("toy", cfg, paths, force=False)
    return cfg, paths, model_dir


def _plans(*sequences):
    """What run_timed returns in `plans`: improving plans, the last one final."""
    out = []
    for n, steps in enumerate(sequences):
        timed, t = [], 0.0
        for action in steps:
            timed.append((t, action, 1.0))
            t += 1.0
        out.append({"found_s": 0.5 + n, "metric": t, "steps": timed, "makespan": t,
                    "final": n == len(sequences) - 1})
    return out


class FakeOptic:
    """Records the problems it is given and returns canned plans."""

    def __init__(self, plans, completed=True, status="solved"):
        self.plans, self.completed, self.status = plans, completed, status
        self.problems, self.domains, self.calls = [], [], []

    def __call__(self, pddl_dir, timeout, memory_mb, stop_at_first):
        self.problems.append((pddl_dir / "problem.pddl").read_text())
        self.domains.append((pddl_dir / "domain.pddl").read_text())
        self.calls.append({"timeout": timeout, "memory_mb": memory_mb, "stop_at_first": stop_at_first})
        return {
            "success": self.status == "solved", "solvability": self.status,
            "plan_text": "x", "plan_actions": [], "message": "ok", "plans": self.plans,
            "search_completed": self.completed,
            "metrics": {"duration": None, "expanded_nodes": 1, "total_time": 0.01},
            "timings": {"grounding_s": 0.1, "first_plan_s": 0.5, "planning_s": 0.4, "wall_s": 3.0, "cpu_s": 0.01},
        }


def _completion_instances(model_dir):
    return [i for i in px.read_jsonl(model_dir / "instances.jsonl")
            if i["goal"] == "completion" and i["replay_ok"]]


def test_runs_only_the_completion_instances_and_asks_for_total_time(built):
    cfg, paths, model_dir = built
    fake = FakeOptic(_plans(["execute_a", "execute_b", "execute_d", "execute_e"]))
    counts = rq2.run_rq2("toy", cfg, paths, runner=fake, force=True)
    instances = _completion_instances(model_dir)
    assert sum(counts.values()) == len(instances) == len(fake.problems) > 0
    # no -N: OPTIC optimises; the limits are the RQ2 ones of the config
    assert all(c == {"timeout": cfg.rq2_timeout_s, "memory_mb": cfg.rq2_memory_mb, "stop_at_first": False}
               for c in fake.calls)
    for problem in fake.problems:
        assert "(:metric minimize (total-time))" in problem
        assert "total-cost" not in problem
        assert "(at " not in problem.split("(:init")[1].split("(:goal")[0].replace("(at start", "")  # no TIL deadline
    # the domain has no costs either
    assert all("total-cost" not in d for d in fake.domains)
    assert "total-cost" not in (model_dir / rq2.DOMAIN_FILE).read_text()
    # ... while the model's own domain (RQ1) keeps them
    assert "total-cost" in (model_dir / "domain.pddl").read_text()


def test_every_plan_and_step_is_saved_with_the_activity_and_real_seconds(built):
    cfg, paths, model_dir = built
    fake = FakeOptic(_plans(["execute_a", "execute_c", "execute_d", "execute_e"],
                            ["execute_a", "execute_tau_x", "execute_b", "execute_d", "execute_e"]))
    rq2.run_rq2("toy", cfg, paths, runner=fake, force=True, limit=1)
    results = px.read_jsonl(model_dir / rq2.RESULTS_FILE)
    plans = px.read_jsonl(model_dir / rq2.PLANS_FILE)
    steps = px.read_jsonl(model_dir / rq2.STEPS_FILE)
    scale = load_model(model_dir).domain.duration_scale_factor

    assert len(results) == 1
    row = results[0]
    assert row["status"] == "solved" and row["search_completed"] is True and row["n_plans"] == 2
    assert row["best_makespan_s"] == pytest.approx(5.0 * scale)  # the last plan, in real seconds
    assert row["first_plan_s"] == 0.5 and row["last_plan_s"] == 1.5
    assert row["plan_length"] == 4  # visible steps of the best plan: a, b, d, e (the tau does not count)
    assert row["timeout_s"] == cfg.rq2_timeout_s and isinstance(row["prefix_acts"], list)

    assert [(p["plan_index"], p["is_final"], p["is_best"]) for p in plans] == [(0, False, False), (1, True, True)]
    assert plans[0]["makespan_s"] == pytest.approx(4.0 * scale)
    first_steps = [s for s in steps if s["plan_index"] == 0]
    assert [s["activity"] for s in first_steps] == ["A", "C", "D", "E"]  # names of the log
    assert [s["start_s"] for s in first_steps] == pytest.approx([0.0, scale, 2 * scale, 3 * scale])
    tau = next(s for s in steps if s["action"] == "execute_tau_x")
    assert tau["activity"] is None and tau["is_tau"] is True


def test_a_search_stopped_by_the_time_limit_is_recorded_as_such(built):
    cfg, paths, model_dir = built
    fake = FakeOptic(_plans(["execute_a", "execute_b", "execute_d", "execute_e"]), completed=False)
    rq2.run_rq2("toy", cfg, paths, runner=fake, force=True, limit=1)
    row = px.read_jsonl(model_dir / rq2.RESULTS_FILE)[0]
    assert row["status"] == "solved" and row["search_completed"] is False


def test_a_timeout_without_plans_is_a_row_without_plans(built):
    cfg, paths, model_dir = built
    fake = FakeOptic([], completed=False, status="timeout")
    rq2.run_rq2("toy", cfg, paths, runner=fake, force=True, limit=1)
    row = px.read_jsonl(model_dir / rq2.RESULTS_FILE)[0]
    assert row["status"] == "timeout" and row["n_plans"] == 0 and row["best_makespan_s"] is None
    assert px.read_jsonl(model_dir / rq2.PLANS_FILE) == []


def test_an_instance_that_fails_becomes_an_error_row_and_the_run_goes_on(built):
    cfg, paths, model_dir = built
    calls = []

    def flaky(pddl_dir, timeout, memory_mb, stop_at_first):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return FakeOptic(_plans(["execute_a"]))(pddl_dir, timeout, memory_mb, stop_at_first)

    rq2.run_rq2("toy", cfg, paths, runner=flaky, force=True, limit=2)
    rows = px.read_jsonl(model_dir / rq2.RESULTS_FILE)
    assert [r["status"] for r in rows] == ["error", "solved"]
    assert "boom" in rows[0]["message"]


def test_resuming_skips_done_instances_and_drops_the_leftovers_of_an_interrupted_one(built):
    cfg, paths, model_dir = built
    fake = FakeOptic(_plans(["execute_a", "execute_d"]))
    rq2.run_rq2("toy", cfg, paths, runner=fake, force=True, limit=2)
    done = [r["group_id"] for r in px.read_jsonl(model_dir / rq2.RESULTS_FILE)]
    # a run killed while writing the third instance: its plan and step rows, no results row
    orphan = max(i["group_id"] for i in _completion_instances(model_dir))
    with open(model_dir / rq2.PLANS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"group_id": orphan, "plan_index": 0}) + "\n")
    with open(model_dir / rq2.STEPS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"group_id": orphan, "plan_index": 0, "step_index": 0}) + "\n")

    fake2 = FakeOptic(_plans(["execute_a", "execute_d"]))
    rq2.run_rq2("toy", cfg, paths, runner=fake2)
    results = px.read_jsonl(model_dir / rq2.RESULTS_FILE)
    ids = [r["group_id"] for r in results]
    assert ids[:2] == done and len(ids) == len(set(ids)) == len(_completion_instances(model_dir))
    assert len(fake2.problems) == len(ids) - 2  # the two done ones were not run again
    plan_rows = px.read_jsonl(model_dir / rq2.PLANS_FILE)
    assert len(plan_rows) == len(ids)  # one plan each: no duplicate of the orphan
    assert sorted(p["group_id"] for p in plan_rows) == sorted(ids)


def test_results_made_with_other_settings_are_refused(built):
    cfg, paths, model_dir = built
    rq2.run_rq2("toy", cfg, paths, runner=FakeOptic(_plans(["execute_a"])), force=True, limit=1)
    with pytest.raises(ValueError, match="other settings"):
        rq2.run_rq2("toy", cfg, paths, runner=FakeOptic(_plans(["execute_a"])), timeout_s=cfg.rq2_timeout_s + 1)
    # --force discards them
    rq2.run_rq2("toy", cfg, paths, runner=FakeOptic(_plans(["execute_a"])),
                timeout_s=cfg.rq2_timeout_s + 1, force=True, limit=1)
    meta = json.loads((model_dir / rq2.META_FILE).read_text())
    assert meta["settings"]["timeout_s"] == cfg.rq2_timeout_s + 1
    assert meta["settings"]["metric"] == "minimize_time" and meta["settings"]["use_costs"] is False


def test_action_names_as_optic_prints_them_map_to_the_activities(built):
    _, _, model_dir = built
    mapping = rq2.activity_of_action(load_model(model_dir, use_costs=False))
    assert mapping["execute_a"] == "A" and mapping["execute_d"] == "D"
    assert {v for v in mapping.values() if v} == {"A", "B", "C", "D", "E"}


def test_report_summarises_the_differences_for_the_instances_with_a_reference(built):
    from paper_eval import report as rp

    cfg, paths, model_dir = built
    # the plan the synthetic log's fastest case follows: A, then B or C, D, E
    fake = FakeOptic(_plans(["execute_a", "execute_b", "execute_d", "execute_e"]))
    rq2.run_rq2("toy", cfg, paths, runner=fake, force=True)
    summary = rp.run_report_rq2(model_dir)

    overall = summary["overall"]
    assert overall["n_instances"] == len(_completion_instances(model_dir))
    assert overall["n_with_plan"] == overall["n_instances"] == overall["n_search_completed"]
    assert 0 <= overall["n_with_reference"] <= overall["n_with_plan"]
    assert overall["difference_s"]["n"] == overall["n_with_reference"]
    assert sum(g["n_instances"] for g in summary["by_bucket"].values()) == overall["n_instances"]
    md = (model_dir / rp.RQ2_SUMMARY_MD).read_text()
    assert md.startswith("### RQ2") and "| all |" in md
    assert json.loads((model_dir / rp.RQ2_SUMMARY_JSON).read_text())["overall"]["n_instances"] == overall["n_instances"]


def test_report_group_arithmetic():
    from paper_eval import report as rp

    rows = [
        {"bucket": "2", "k": 2, "status": "solved", "search_completed": True, "best_makespan_s": 8.0, "reference_rt_s": 10.0},
        {"bucket": "2", "k": 2, "status": "solved", "search_completed": False, "best_makespan_s": 12.0, "reference_rt_s": 10.0},
        {"bucket": "2", "k": 2, "status": "solved", "search_completed": True, "best_makespan_s": 5.0, "reference_rt_s": None},
        {"bucket": "2", "k": 2, "status": "timeout", "search_completed": False, "best_makespan_s": None, "reference_rt_s": None},
    ]
    g = rp.summarize_rq2_group(rows)
    assert g["n_instances"] == 4 and g["by_status"] == {"solved": 3, "timeout": 1}
    assert g["n_with_plan"] == 3 and g["n_search_completed"] == 2 and g["n_with_reference"] == 2
    assert g["share_with_reference"] == pytest.approx(2 / 3)
    assert g["difference_s"]["mean"] == pytest.approx(0.0)       # (-2 + 2) / 2, the sign is kept
    assert g["abs_difference_s"]["mean"] == pytest.approx(2.0)
    assert g["ratio"]["mean"] == pytest.approx(1.0)               # (0.8 + 1.2) / 2


@pytest.mark.integration
def test_with_the_real_optic_the_fastest_plan_is_saved_with_the_log_activities(built):
    """Real OPTIC on the real (synthetic) model: the domain without costs is accepted, the plans
    OPTIC prints are parsed, and their action names map to the activities of the log."""
    import planning.optic as optic_module

    if not optic_module.is_available():
        pytest.skip("OPTIC binary not compiled")
    cfg, paths, model_dir = built
    rq2.run_rq2("toy", cfg, paths, force=True, limit=3, timeout_s=60)
    results = px.read_jsonl(model_dir / rq2.RESULTS_FILE)
    steps = px.read_jsonl(model_dir / rq2.STEPS_FILE)
    assert len(results) == 3 and all(r["status"] == "solved" for r in results)
    assert all(r["search_completed"] and r["n_plans"] >= 1 and r["best_makespan_s"] > 0 for r in results)
    visible = {s["activity"] for s in steps if not s["is_tau"]}
    assert visible <= {"A", "B", "C", "D", "E"} and visible
    assert all(s["start_s"] >= 0 and s["duration_s"] > 0 for s in steps if not s["is_tau"])


def test_a_silent_transition_with_guards_is_not_an_activity():
    """A silent transition that has XOR guards sits in prepared.transitions with variants
    (execute_tau_15_v0, ...): it must still map to None, not to the activity "tau_15"."""
    from types import SimpleNamespace

    from encoding.prepared_input import GraphNode, PreparedDomainInput, PreparedTransition, VariantInfo

    def transition(name):
        return PreparedTransition(activity_name=name, input_places=[], preconditions=[], effect_groups=[])

    prepared = PreparedDomainInput(
        transitions={"A": transition("A"), "tau_15": transition("tau_15"), "tau_2": transition("tau_2")},
        xor_branches={}, attribute_catalog={},
        nodes=[GraphNode("t_a", "transition", "A"), GraphNode("t_15", "silent", "tau_15"),
               GraphNode("t_2", "silent", "tau_2")],
    )
    model = SimpleNamespace(
        prepared=prepared,
        domain=SimpleNamespace(actions=[SimpleNamespace(name=n) for n in (
            "execute_A", "execute_tau_15_v0", "execute_tau_15_v1", "execute_tau_2")]),
        variant_map={n: VariantInfo(activity_name="tau_15", preconditions=[], effect_group=None)
                     for n in ("execute_tau_15_v0", "execute_tau_15_v1")},
    )
    assert rq2.activity_of_action(model) == {
        "execute_a": "A", "execute_tau_15_v0": None, "execute_tau_15_v1": None, "execute_tau_2": None,
    }
