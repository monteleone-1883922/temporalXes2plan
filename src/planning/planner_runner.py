"""Unified planning dispatcher — routes to Fast Downward or OPTIC."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from planning import fast_downward, optic


@dataclass
class PlannerResult:
    """Structured result returned by any planner wrapper."""

    success: bool
    solvability: str
    plan_text: Optional[str]
    plan_actions: List[str]
    metrics: Dict[str, Any]
    stdout: str
    stderr: str
    message: str
    planner: str = ""
    # Only filled when OPTIC runs with show_all_plans: every plan it printed (see
    # optic._parse_optic_plans), the timings, and whether it ended by itself.
    plans: List[Dict[str, Any]] = field(default_factory=list)
    timings: Optional[Dict[str, Any]] = None
    search_completed: Optional[bool] = None


def get_planners_status() -> Dict[str, Any]:
    """Return availability and configuration for all supported planners."""
    return {
        "fast_downward": {
            "available": fast_downward.is_built(),
            "search_configs": fast_downward.get_search_configs(),
        },
        "optic": {
            "available": optic.is_available(),
        },
    }


def run_planner(
    pddl_dir: Path,
    planner: str,
    options: Dict[str, Any],
    log_fn: Optional[Callable[[str], None]] = None,
) -> PlannerResult:
    """Dispatch a planning run to the requested planner.

    Args:
        pddl_dir: Directory containing domain.pddl and problem.pddl.
        planner: "fast_downward" or "optic".
        options: Planner-specific options dict.
        log_fn: Optional progress callback (called with each log line).

    Returns:
        PlannerResult with outcome details.
    """
    if planner == "fast_downward":
        raw = fast_downward.run(
            pddl_dir=pddl_dir,
            search_key=options.get("search", "astar_lmcut"),
            timeout=int(options.get("timeout", 30)),
            memory_mb=options.get("memory_mb"),  # Optional[int]; None = no limit
            log_fn=log_fn,
        )
    elif planner == "optic" and options.get("show_all_plans"):
        # The streaming runner keeps every plan OPTIC prints (with the time it was
        # found and its metric value); the plain runner only keeps the last one.
        if log_fn:
            log_fn("Running OPTIC (all plans are kept)")
        raw = optic.run_timed(
            pddl_dir=pddl_dir,
            timeout=int(options.get("timeout", 60)),
            memory_mb=int(options.get("memory_mb", 4000)),
            stop_at_first=bool(options.get("stop_at_first", True)),
            ignore_costs=bool(options.get("ignore_costs", False)),
        )
        _save_all_plans(pddl_dir, raw)
        if log_fn:
            log_fn(f"{len(raw.get('plans') or [])} plan(s) printed — {raw.get('message', '')}")
    elif planner == "optic":
        raw = optic.run(
            pddl_dir=pddl_dir,
            stop_at_first=bool(options.get("stop_at_first", True)),
            ignore_costs=bool(options.get("ignore_costs", False)),
            timeout=int(options.get("timeout", 60)),
            memory_mb=int(options.get("memory_mb", 4000)),
            log_fn=log_fn,
        )
    else:
        return PlannerResult(
            success=False,
            solvability="error",
            plan_text=None,
            plan_actions=[],
            metrics={},
            stdout="",
            stderr="",
            message=f"Unknown planner: {planner!r}",
            planner=planner,
        )

    return PlannerResult(planner=planner, **raw)


def _save_all_plans(pddl_dir: Path, raw: Dict[str, Any]) -> None:
    """Write OPTIC's raw output and the list of plans next to the PDDL files."""
    import json

    (pddl_dir / "optic_output.txt").write_text(raw.get("stdout") or "", encoding="utf-8")
    (pddl_dir / "plans.json").write_text(
        json.dumps(
            {"plans": raw.get("plans") or [], "timings": raw.get("timings"),
             "search_completed": raw.get("search_completed")},
            indent=2,
        ),
        encoding="utf-8",
    )
