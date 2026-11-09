"""Phase "plan": run OPTIC on every instance and record how long it takes.

This is the instrument behind RQ1 (scalability). For each instance of
instances.jsonl it
  1. builds the PDDL problem (initial marking and attribute values of the
     prefix, goal = complete the process, optional deadline),
  2. runs OPTIC until it finds the FIRST plan, with a time and a memory limit,
  3. appends one JSON line to plan_results.jsonl with the grounding time, the
     planning time, the outcome and the plan.

Design choices:
  * One OPTIC at a time, in a fixed order (by group_id): the times are the
    measurement, so nothing else may compete for the machine.
  * Resumable: a line is written (and flushed) as soon as an instance is done,
    and a rerun skips the instances already in the file.
  * No retries: a timeout or an out-of-memory abort is a result, reported apart
    from the averages (see report.py), not something to hide.
  * Results with different settings (e.g. another timeout) are never mixed in
    the same file: they would not be comparable.

Not part of `--phase all` (it can run for hours): launch it with `--phase plan`.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from evaluation.eval_api import EvalAPI
from evaluation.query_builder import compute_cost_scale_factor
from paper_eval import prefixes as px
from paper_eval.build_model import DOMAIN_NAME, LoadedModel, load_model
from paper_eval.config import PaperEvalConfig
from paper_eval.split import read_split
from planning import optic

logger = logging.getLogger("paper_eval")

RESULTS_FILE = "plan_results.jsonl"
META_FILE = "plan_meta.json"
PREFIXES_META = "prefixes_meta.json"  # written by the prefixes phase

# Statuses besides the ones OPTIC's outcome gives (solved, timeout, out_of_memory,
# unsolvable_structural, unsolvable_parse, unsolvable_resource, error).
STATUS_SKIPPED_REPLAY = "skipped_replay_failed"  # no initial state: the prefix could not be replayed
STATUS_SKIPPED_EXPIRED = "skipped_expired"  # the deadline was already past at the cut
STATUS_ERROR = "error"
# A run whose first instances all end as "unsolvable_parse" has a PDDL OPTIC cannot read (domain
# and problem disagree): it is stopped instead of writing thousands of useless rows.
PARSE_FAILURE_CHECK_AFTER = 30

# Same metric the thesis runner uses for Q1/Q2 (time plus weighted cost). With
# OPTIC's -N (first plan, no optimisation) the metric does not change how long the first plan takes.
DEFAULT_METRIC = "minimize_weighted"
MODE_FIRST_PLAN = "first_plan"

# The goals an instance can have; "all" runs every instance.
GOAL_CHOICES = ("completion", "deadline", "all")

# Optional hook for the tests: replaces the function that runs OPTIC.
Runner = Callable[..., Dict[str, Any]]


def prefixes_deadline_variant_only(model_dir: Path) -> bool:
    """Deadline rule the instances of model_dir were made with (False when the
    prefixes phase left no meta file: the original min rule)."""
    path = Path(model_dir) / PREFIXES_META
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("deadline_variant_only", False))


def build_problem_text(
    model: LoadedModel, instance: Dict[str, Any], metric: str, cost_weight: float,
) -> str:
    """PDDL problem of one instance, built like the thesis runner builds Q1/Q2.

    The goal is "reach the end of the process" (empty attribute goal plus the end
    place). The deadline, when the instance has one, is in real seconds and is
    divided by the domain's duration_scale_factor here, so it stays on the same
    time axis as the (possibly rescaled) action durations.

    instance: one row of instances.jsonl (a dict).
    """
    scale = model.domain.duration_scale_factor
    deadline = instance["deadline_s"] / scale if instance.get("deadline_s") is not None else None
    return EvalAPI().build_problem(
        serialized=model.serialized,
        init_places=instance["init_places"],
        goal_sop=[[]],
        init_effects=instance["init_effects"],
        metric=metric,
        cost_weight=cost_weight,
        require_completion=True,
        deadline=deadline,
        # Must be the name written in domain.pddl, or OPTIC rejects the pair.
        domain_name=DOMAIN_NAME,
        temporal=True,
    )


def _measures(**values: Any) -> Dict[str, Any]:
    """All the measured fields, None unless given, so every row has the same keys."""
    row = {
        "grounding_s": None, "planning_s": None, "first_plan_s": None, "wall_s": None, "cpu_s": None,
        "states_evaluated": None, "makespan": None, "makespan_s": None,
        "plan_length": None, "plan_actions": None, "message": None,
    }
    row.update(values)
    return row


def _row(base: Dict[str, Any], status: str, **measures: Any) -> Dict[str, Any]:
    return {**base, "status": status, **_measures(**measures)}


def plan_instance(
    instance: Dict[str, Any],
    model: LoadedModel,
    work_dir: Path,
    log_name: str,
    settings: Dict[str, Any],
    cost_weight: float,
    runner: Runner,
) -> Dict[str, Any]:
    """Result row of one instance. Never raises: a failure becomes a row with
    status "error" (or a "skipped_*" status), so one bad instance does not stop the run."""
    base = {
        "log": log_name,
        "group_id": instance["group_id"],
        "goal": instance["goal"],
        "k": instance["k"],
        "bucket": instance["bucket"],
        "n_cases": len(instance["cases"]),  # how many test cases the instance stands for
        # ... and how many of them can be replayed to the real final state.
        "n_cases_replayable": sum(1 for c in instance["cases"] if c.get("is_replayable")),
        "deadline_s": instance["deadline_s"],
        "timeout_s": settings["timeout_s"],
        "memory_mb": settings["memory_mb"],
    }
    # Instances OPTIC cannot be asked about are recorded, not dropped, so they can be counted.
    if not instance["replay_ok"]:
        return _row(base, STATUS_SKIPPED_REPLAY)
    if instance["expired"]:
        return _row(base, STATUS_SKIPPED_EXPIRED)

    try:
        problem = build_problem_text(model, instance, settings["metric"], cost_weight)
        (work_dir / "problem.pddl").write_text(problem, encoding="utf-8")
        result = runner(
            work_dir, timeout=settings["timeout_s"], memory_mb=settings["memory_mb"],
            stop_at_first=settings["stop_at_first"],
        )
    except Exception as exc:  # keep going: this instance is an "error" row
        logger.warning("[%s] instance %s failed: %s", log_name, instance["group_id"], exc)
        return _row(base, STATUS_ERROR, message=f"{type(exc).__name__}: {exc}")

    timings, metrics = result["timings"], result["metrics"]
    scale = model.domain.duration_scale_factor
    makespan = metrics.get("duration")  # in the domain's (rescaled) time units
    actions = result["plan_actions"] or []
    return _row(
        base, result["solvability"],
        grounding_s=timings["grounding_s"], planning_s=timings["planning_s"],
        first_plan_s=timings["first_plan_s"], wall_s=timings["wall_s"], cpu_s=timings["cpu_s"],
        states_evaluated=metrics.get("expanded_nodes"),
        makespan=makespan,
        makespan_s=makespan * scale if makespan is not None else None,  # real seconds
        # A plan made only of silent steps is still a plan (length 0): "no plan" is None.
        plan_length=len(actions) if result["plan_text"] is not None else None,
        plan_actions=actions if result["plan_text"] is not None else None,
        message=result["message"],
    )


def _environment() -> Dict[str, Any]:
    """What a reader needs to judge the timings: machine and tool versions."""
    def first_line(cmd: List[str]) -> Optional[str]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
            return out.splitlines()[0] if out else None
        except Exception:
            return None

    version_file = optic.OPTIC_DIR / "VERSION"
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "optic_version": version_file.read_text().strip() if version_file.is_file() else None,
        "stdbuf": first_line(["stdbuf", "--version"]),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def read_results(path: Path) -> List[Dict[str, Any]]:
    """Rows of plan_results.jsonl. A truncated last line (the run was killed
    while writing) is ignored: that instance is simply done again."""
    rows: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("%s line %d is not valid JSON (partial write?): ignored", path, number)
    return rows


def _open_for_append(path: Path):
    """Append handle; if the file ends in a partial line, start a fresh one so
    the next row is not glued to it."""
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    fh = open(path, "a", encoding="utf-8")
    if needs_newline:
        fh.write("\n")
    return fh


def run_plan(
    log_name: str,
    cfg: PaperEvalConfig,
    paths: Any,
    goal: str = "all",
    limit: Optional[int] = None,
    timeout_s: Optional[int] = None,
    memory_mb: Optional[int] = None,
    force: bool = False,
    runner: Runner = optic.run_timed,
) -> Counter:
    """Run OPTIC on the instances of one log and return the count of each status
    among the instances handled by THIS call.

    Args:
        goal: "completion", "deadline" or "all".
        limit: Handle at most this many instances not done yet (to try the run).
        timeout_s, memory_mb: Override the per-log values of the config.
        force: Discard the previous results and start over. Without it, results
            made with other settings are refused instead of being mixed in.
        runner: Function that runs OPTIC (optic.run_timed); the tests replace it.
    """
    if goal not in GOAL_CHOICES:
        raise ValueError(f"goal must be one of {GOAL_CHOICES}, got {goal!r}")

    split = read_split(paths.split_json)
    model_dir = paths.model_dir(split.split_hash)
    instances_path = model_dir / "instances.jsonl"
    if not instances_path.exists():
        raise FileNotFoundError(f"{instances_path} not found: run the 'prefixes' phase first")

    settings = {
        "timeout_s": timeout_s if timeout_s is not None else cfg.timeout_for(log_name),
        "memory_mb": memory_mb if memory_mb is not None else cfg.memory_for(log_name),
        "metric": DEFAULT_METRIC,
        "mode": MODE_FIRST_PLAN,
        "stop_at_first": True,  # OPTIC -N
        # Rule that made the instances: other instances, other group_ids, not comparable.
        "deadline_variant_only": prefixes_deadline_variant_only(model_dir),
    }
    results_path, meta_path = model_dir / RESULTS_FILE, model_dir / META_FILE

    if force:
        results_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    elif meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8")).get("settings")
        if previous is not None:  # results older than this setting used the min rule
            previous = {"deadline_variant_only": False, **previous}
        if previous != settings:
            raise ValueError(
                f"{results_path} was made with other settings ({previous}) than the current ones "
                f"({settings}); mixing them would make the times incomparable. "
                f"Use --force to discard the old results."
            )

    done: Set[int] = {row["group_id"] for row in read_results(results_path)}
    instances = sorted(px.read_jsonl(instances_path), key=lambda i: i["group_id"])
    todo = [
        i for i in instances
        if i["group_id"] not in done and (goal == "all" or i["goal"] == goal)
    ]
    if limit is not None:
        todo = todo[:limit]
    logger.info("[%s] plan: %d to do (%d already done), timeout %ss, memory %s MB",
                log_name, len(todo), len(done), settings["timeout_s"], settings["memory_mb"])
    if not todo:
        return Counter()

    model = load_model(model_dir)
    cost_weight = compute_cost_scale_factor(model.domain)
    if not meta_path.exists():
        meta = {
            "settings": settings,
            "environment": _environment(),
            "duration_scale_factor": model.domain.duration_scale_factor,
            "cost_weight": cost_weight,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    counts: Counter = Counter()
    started = time.perf_counter()
    # One working directory for the whole run: the domain is written once, only
    # problem.pddl changes from one instance to the next.
    with tempfile.TemporaryDirectory() as tmp, _open_for_append(results_path) as out:
        work_dir = Path(tmp)
        # The domain OPTIC reads is the one the model builds NOW (the same code that wrote
        # domain.pddl, plus any later fix of it, e.g. durations OPTIC can represent): if it
        # differs from the file in the model folder, the file is brought up to date, so the
        # folder always shows the domain the results were obtained with.
        domain_text = str(model.domain)
        (work_dir / "domain.pddl").write_text(domain_text, encoding="utf-8")
        disk_domain = model_dir / "domain.pddl"
        if not disk_domain.exists() or disk_domain.read_text(encoding="utf-8") != domain_text:
            logger.warning("[%s] domain.pddl of the model folder differs from the rebuilt domain: updated", log_name)
            disk_domain.write_text(domain_text, encoding="utf-8")
        for n, instance in enumerate(todo, start=1):
            row = plan_instance(instance, model, work_dir, log_name, settings, cost_weight, runner)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()  # on disk now, so a killed run loses at most this instance
            counts[row["status"]] += 1
            elapsed = time.perf_counter() - started
            if n == PARSE_FAILURE_CHECK_AFTER and counts["unsolvable_parse"] == n:
                raise RuntimeError(
                    f"[{log_name}] the first {n} instances all ended as unsolvable_parse: OPTIC cannot read "
                    f"the PDDL (see its message with `optic-clp.sh -N domain.pddl problem.pddl`). Stopped; "
                    f"the rows written so far are in {results_path}."
                )
            # One line per instance, so a run that has stopped is noticed at once.
            logger.info("[%s] plan %d/%d (%.1f%%) group %s %s k=%s: %s%s",
                        log_name, n, len(todo), 100.0 * n / len(todo), row["group_id"], row["goal"],
                        row["k"], row["status"], _times(row))
            if n % 50 == 0 or n == len(todo):
                eta = elapsed / n * (len(todo) - n)
                logger.info("[%s] plan: %d/%d done in %s, about %s left, %s",
                            log_name, n, len(todo), _hms(elapsed), _hms(eta), dict(counts))
    return counts


def _times(row: Dict[str, Any]) -> str:
    """" grounding Xs planning Ys" for the log line (nothing for a row without times)."""
    parts = []
    if row.get("grounding_s") is not None:
        parts.append(f"grounding {row['grounding_s']:.2f}s")
    if row.get("planning_s") is not None:
        parts.append(f"planning {row['planning_s']:.2f}s")
    if not parts and row.get("wall_s") is not None:
        parts.append(f"wall {row['wall_s']:.2f}s")
    return " " + ", ".join(parts) if parts else ""


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"
