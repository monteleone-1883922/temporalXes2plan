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
