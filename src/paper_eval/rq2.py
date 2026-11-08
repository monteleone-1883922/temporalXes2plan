"""Phase "rq2": the fastest plan OPTIC finds for every instance, all plans saved.

RQ2 asks how far the fastest completion the model admits is from the fastest observed one.
This phase produces the plans; nothing is compared here. The comparison with the test
traces (reference trace, differences, ratios) is made later, in the DuckDB views and the
report, from the files below.

For each "completion" instance of instances.jsonl (the ones whose prefix could be replayed)
  1. builds the PDDL problem: initial state of the prefix, goal = complete the process,
     metric (total-time), no deadline;
  2. runs OPTIC WITHOUT -N, so it keeps improving the plan until it has nothing better or the
     time limit is reached, on a domain WITHOUT action costs (RQ2 ignores them);
  3. appends what it found to three JSONL files in the model folder.

Files (all times in real seconds, the domain's duration scale already undone):
  rq2_results.jsonl     one row per instance: status, whether OPTIC ended by itself
                        (search_completed) or was stopped, number of plans, timings, best makespan
  rq2_plans.jsonl       one row per plan OPTIC printed: when, makespan, final / best
  rq2_plan_steps.jsonl  one row per step of every plan: PDDL action, activity of the log
                        (null for silent steps), start and duration
  rq2_domain.pddl       the domain used (no costs)
  rq2_meta.json         the settings of the run and the machine

Same design as the "plan" phase: one OPTIC at a time, resumable (an instance is done when its
row is in rq2_results.jsonl; the plans and steps of an interrupted instance are dropped and
redone), no retries, results made with other settings refused instead of mixed.
Not part of `--phase all`: launch it with `--phase rq2`.

"search_completed" means OPTIC ended by itself: it has no better plan under its own pruning
(the plan is the best one OPTIC can find), not a proof of optimality.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core_utils import sanitize_name
from evaluation.eval_api import EvalAPI
from paper_eval import prefixes as px
from paper_eval.build_model import DOMAIN_NAME, LoadedModel, load_model
from paper_eval.config import PaperEvalConfig
from paper_eval.plan import STATUS_ERROR, _environment, _open_for_append, read_results
from paper_eval.split import read_split
from planning import optic

logger = logging.getLogger("paper_eval")

RESULTS_FILE = "rq2_results.jsonl"
PLANS_FILE = "rq2_plans.jsonl"
STEPS_FILE = "rq2_plan_steps.jsonl"
META_FILE = "rq2_meta.json"
DOMAIN_FILE = "rq2_domain.pddl"

METRIC = "minimize_time"  # (:metric minimize (total-time)); costs are not in the domain
MODE_OPTIMISE = "optimise"

Runner = Callable[..., Dict[str, Any]]


def build_problem_text(model: LoadedModel, instance: Dict[str, Any]) -> str:
    """PDDL problem of one instance: initial state of the prefix, goal "reach the end of
    the process", metric total-time and no deadline (RQ2 does not use one)."""
    return EvalAPI().build_problem(
        serialized=model.serialized,
        init_places=instance["init_places"],
        goal_sop=[[]],
        init_effects=instance["init_effects"],
        metric=METRIC,
        cost_weight=1.0,
        require_completion=True,
        deadline=None,
        domain_name=DOMAIN_NAME,
        temporal=True,
    )


def activity_of_action(model: LoadedModel) -> Dict[str, Optional[str]]:
    """PDDL action name as OPTIC prints it -> activity of the log (None for a silent step).

    OPTIC prints the sanitised names of the domain ("execute_act_one" for the action
    "execute_Act One"), so the keys are sanitised. An action is a variant of an activity
    (variant_map) or the activity itself; anything that is not an activity of the net
    is a silent transition."""
    visible = set(model.prepared.transitions)
    mapping: Dict[str, Optional[str]] = {}
    for action in model.domain.actions:
        raw = action.name
        info = model.variant_map.get(raw)
        if info is not None:
            activity: Optional[str] = info.activity_name
        else:
            base = raw[len("execute_"):] if raw.startswith("execute_") else raw
            activity = base if base in visible else None
        mapping[sanitize_name(raw)] = activity
    return mapping


def _row(base: Dict[str, Any], status: str, **values: Any) -> Dict[str, Any]:
    row = {
        "search_completed": None, "n_plans": 0, "plan_length": None,
        "grounding_s": None, "first_plan_s": None, "last_plan_s": None, "wall_s": None, "cpu_s": None,
        "best_makespan_s": None, "message": None,
    }
    row.update(values)
    return {**base, "status": status, **row}


def plan_instance(
    instance: Dict[str, Any],
    model: LoadedModel,
    work_dir: Path,
    log_name: str,
    settings: Dict[str, Any],
    actions: Dict[str, Optional[str]],
    runner: Runner,
) -> Dict[str, Any]:
    """Everything one instance produced: {"result": row, "plans": [rows], "steps": [rows]}.
    Never raises: a failure becomes a result row with status "error"."""
    gid = instance["group_id"]
    base = {
        "log": log_name,
        "group_id": gid,
        "k": instance["k"],
        "bucket": instance["bucket"],
        "prefix_acts": instance["prefix_acts"],
        "n_cases": len(instance["cases"]),
        "timeout_s": settings["timeout_s"],
        "memory_mb": settings["memory_mb"],
    }
    try:
        problem = build_problem_text(model, instance)
        (work_dir / "problem.pddl").write_text(problem, encoding="utf-8")
        result = runner(
            work_dir, timeout=settings["timeout_s"], memory_mb=settings["memory_mb"],
            stop_at_first=False,
        )
    except Exception as exc:  # this instance is an "error" row, the run goes on
        logger.warning("[%s] instance %s failed: %s", log_name, gid, exc)
        return {"result": _row(base, STATUS_ERROR, message=f"{type(exc).__name__}: {exc}"),
                "plans": [], "steps": []}

    scale = model.domain.duration_scale_factor
    plans = result.get("plans") or []
    plan_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for index, plan in enumerate(plans):
        makespan = plan["makespan"]
        plan_rows.append({
            "log": log_name, "group_id": gid, "plan_index": index,
            "found_s": plan["found_s"],
            "makespan_s": makespan * scale if makespan is not None else None,
            "is_final": bool(plan["final"]),
            "is_best": index == len(plans) - 1,  # every plan improves on the previous one
        })
        for step_index, (start, action, duration) in enumerate(plan["steps"]):
            activity = actions.get(action)
            step_rows.append({
                "log": log_name, "group_id": gid, "plan_index": index, "step_index": step_index,
                "action": action, "activity": activity, "is_tau": activity is None,
                "start_s": start * scale, "duration_s": duration * scale,
            })

    timings = result["timings"]
    best = plan_rows[-1] if plan_rows else None
    row = _row(
        base, result["solvability"],
        search_completed=bool(result.get("search_completed")),
        n_plans=len(plans),
        plan_length=sum(1 for s in step_rows if s["plan_index"] == len(plans) - 1 and not s["is_tau"])
        if plans else None,
        grounding_s=timings["grounding_s"], first_plan_s=timings["first_plan_s"],
        last_plan_s=best["found_s"] if best else None,
        wall_s=timings["wall_s"], cpu_s=timings["cpu_s"],
        best_makespan_s=best["makespan_s"] if best else None,
        message=result["message"],
    )
    return {"result": row, "plans": plan_rows, "steps": step_rows}


def _drop_unfinished(path: Path, done: Set[int]) -> None:
    """Remove from a plans/steps file the rows of instances without a results row: the run
    was interrupted while writing them, they are done again."""
    rows = read_results(path)
    kept = [r for r in rows if r.get("group_id") in done]
    if len(kept) != len(rows):
        with open(path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_rq2(
    log_name: str,
    cfg: PaperEvalConfig,
    paths: Any,
    limit: Optional[int] = None,
    timeout_s: Optional[int] = None,
    memory_mb: Optional[int] = None,
    force: bool = False,
    runner: Runner = optic.run_timed,
) -> Counter:
    """Run OPTIC (optimising total-time) on the completion instances of one log and return
    the count of each status among the instances handled by THIS call.

    limit: handle at most this many instances not done yet (to try the run).
    timeout_s, memory_mb: override the values of the config.
    force: discard the previous results; without it, results made with other settings
        are refused instead of being mixed in.
    runner: function that runs OPTIC (optic.run_timed); the tests replace it.
    """
    split = read_split(paths.split_json)
    model_dir = paths.model_dir(split.split_hash)
    instances_path = model_dir / "instances.jsonl"
    if not instances_path.exists():
        raise FileNotFoundError(f"{instances_path} not found: run the 'prefixes' phase first")

    settings = {
        "timeout_s": timeout_s if timeout_s is not None else cfg.rq2_timeout_s,
        "memory_mb": memory_mb if memory_mb is not None else cfg.rq2_memory_mb,
        "metric": METRIC,
        "mode": MODE_OPTIMISE,
        "use_costs": False,
        "deadline": None,
    }
    results_path, plans_path, steps_path, meta_path = (
        model_dir / RESULTS_FILE, model_dir / PLANS_FILE, model_dir / STEPS_FILE, model_dir / META_FILE,
    )
    if force:
        for path in (results_path, plans_path, steps_path, meta_path):
            path.unlink(missing_ok=True)
    elif meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8")).get("settings")
        if previous != settings:
            raise ValueError(
                f"{results_path} was made with other settings ({previous}) than the current ones "
                f"({settings}); mixing them would make the results incomparable. "
                f"Use --force to discard the old results."
            )

    done: Set[int] = {row["group_id"] for row in read_results(results_path)}
    for path in (plans_path, steps_path):
        _drop_unfinished(path, done)
    instances = sorted(px.read_jsonl(instances_path), key=lambda i: i["group_id"])
    todo = [
        i for i in instances
        if i["goal"] == "completion" and i["replay_ok"] and i["group_id"] not in done
    ]
    if limit is not None:
        todo = todo[:limit]
    logger.info("[%s] rq2: %d to do (%d already done), timeout %ss, memory %s MB",
                log_name, len(todo), len(done), settings["timeout_s"], settings["memory_mb"])
    if not todo:
        return Counter()

    model = load_model(model_dir, use_costs=False)
    actions = activity_of_action(model)
    model.domain.write(model_dir / DOMAIN_FILE)
    if not meta_path.exists():
        meta = {
            "settings": settings,
            "environment": _environment(),
            "duration_scale_factor": model.domain.duration_scale_factor,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    counts: Counter = Counter()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp, \
            _open_for_append(results_path) as out_results, \
            _open_for_append(plans_path) as out_plans, \
            _open_for_append(steps_path) as out_steps:
        work_dir = Path(tmp)
        shutil.copyfile(model_dir / DOMAIN_FILE, work_dir / "domain.pddl")
        for n, instance in enumerate(todo, start=1):
            produced = plan_instance(instance, model, work_dir, log_name, settings, actions, runner)
            # Plans and steps first, the results row last: it is what marks the instance as done.
            for row in produced["steps"]:
                out_steps.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in produced["plans"]:
                out_plans.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_steps.flush()
            out_plans.flush()
            out_results.write(json.dumps(produced["result"], ensure_ascii=False) + "\n")
            out_results.flush()
            counts[produced["result"]["status"]] += 1
            if n % 20 == 0 or n == len(todo):
                logger.info("[%s] rq2: %d/%d done in %.0fs, %s",
                            log_name, n, len(todo), time.perf_counter() - started, dict(counts))
    return counts
