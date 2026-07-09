"""Compute per-query and aggregate metrics for the evaluation harness.

Each q*_metrics() function takes a PlanResult (and optional context) and
returns a flat dict ready for JSON serialisation.  aggregate() reduces a list
of such dicts to mean/std/median across numeric fields, ignoring None values.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from evaluation.eval_api import PlanResult
from evaluation.query_builder import _remaining_budget
from evaluation.trace_sampler import PrefixSample


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def q1_metrics(
    result: PlanResult,
    prefix_sample: PrefixSample,
    cost_weight: float = 0.001,
) -> Dict[str, Any]:
    """Compute Q1 metrics: process completion, no deadline.

    Q1's PDDL problem has no deadline, so `budget_s`/`within_budget` never
    affect whether the planner finds a plan — they are computed here purely
    for statistics, to compare `plan_time_s` against how long the process
    actually took in the original trace (same as Q2/Q3).

    Args:
        result: PlanResult from the planner invocation.
        prefix_sample: PrefixSample used to compute the reference real
            duration of the trace suffix.
        cost_weight: Scaling factor α used in the weighted objective.

    Returns:
        Dict with keys: solved, solvability, plan_time_s, plan_cost,
        weighted_objective, budget_s, within_budget.
    """
    budget_s = _remaining_budget(prefix_sample)
    within_budget = _within_budget(result.duration_s, budget_s)
    weighted = _weighted_objective(result.duration_s, result.cost, cost_weight)
    return {
        "solved": result.success,
        "solvability": result.solvability,
        "plan_time_s": result.duration_s,
        "plan_cost": result.cost,
        "weighted_objective": weighted,
        "budget_s": budget_s,
        "within_budget": within_budget,
    }


def q2_metrics(
    result: PlanResult,
    prefix_sample: PrefixSample,
    cost_weight: float = 0.001,
) -> Dict[str, Any]:
    """Compute Q2 metrics: completion within remaining time budget.

    Args:
        result: PlanResult from the planner invocation.
        prefix_sample: PrefixSample used to compute the time budget.
        cost_weight: Scaling factor α used in the weighted objective.

    Returns:
        Dict with keys: solved, solvability, plan_time_s, plan_cost,
        weighted_objective, budget_s, within_budget.
    """
    budget_s = _remaining_budget(prefix_sample)
    within_budget = _within_budget(result.duration_s, budget_s)
    weighted = _weighted_objective(result.duration_s, result.cost, cost_weight)
    return {
        "solved": result.success,
        "solvability": result.solvability,
        "plan_time_s": result.duration_s,
        "plan_cost": result.cost,
        "weighted_objective": weighted,
        "budget_s": budget_s,
        "within_budget": within_budget,
    }


def q3_metrics(
    result: PlanResult,
    prefix_sample: PrefixSample,
    goal_sop: List[List[Dict[str, Any]]],
    ground_truth_reachable: bool,
    cost_weight: float = 0.001,
) -> Dict[str, Any]:
    """Compute Q3 metrics: completion within budget with attribute constraints.

    Args:
        result: PlanResult from the planner invocation.
        prefix_sample: PrefixSample used to compute the time budget.
        goal_sop: Goal used in the Q3 query (kept for reference).
        ground_truth_reachable: Pre-computed structural reachability flag.
        cost_weight: Scaling factor α used in the weighted objective.

    Returns:
        Dict with keys: solved, solvability, plan_time_s, plan_cost,
        weighted_objective, budget_s, within_budget, ground_truth_reachable,
        correct.
    """
    budget_s = _remaining_budget(prefix_sample)
    within_budget = _within_budget(result.duration_s, budget_s)
    weighted = _weighted_objective(result.duration_s, result.cost, cost_weight)
    correct = result.success == ground_truth_reachable
    return {
        "solved": result.success,
        "solvability": result.solvability,
        "plan_time_s": result.duration_s,
        "plan_cost": result.cost,
        "weighted_objective": weighted,
        "budget_s": budget_s,
        "within_budget": within_budget,
        "ground_truth_reachable": ground_truth_reachable,
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_trace_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce a list of per-trace metric dicts to mean/std/median per field.

    Only numeric fields (int or float) are aggregated; None values are excluded
    from each field's computation.  Fields that contain only None values produce
    {"mean": None, "std": None, "median": None}.  Non-numeric fields are omitted
    from the output.

    Args:
        per_trace_metrics: List of dicts as returned by q*_metrics().

    Returns:
        Dict mapping each numeric field name to a sub-dict with keys
        mean, std, median.
    """
    if not per_trace_metrics:
        return {}

    all_keys = {k for d in per_trace_metrics for k in d}
    result: Dict[str, Any] = {}

    for key in sorted(all_keys):
        values = [
            d[key] for d in per_trace_metrics
            if key in d and isinstance(d[key], (int, float)) and not isinstance(d[key], bool)
        ]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        median = statistics.median(values)
        result[key] = {"mean": mean, "std": std, "median": median}

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_objective(
    plan_time_s: Optional[float],
    plan_cost: Optional[float],
    cost_weight: float,
) -> Optional[float]:
    """Compute the combined weighted objective: time + α * cost.

    Args:
        plan_time_s: Plan duration in seconds, or None if unavailable.
        plan_cost: Plan cost, or None if unavailable.
        cost_weight: Scaling factor α.

    Returns:
        Weighted objective value, or None if plan_time_s is None.
    """
    if plan_time_s is None:
        return None
    cost = plan_cost if plan_cost is not None else 0.0
    return plan_time_s + cost_weight * cost


def _within_budget(
    plan_time_s: Optional[float],
    budget_s: Optional[float],
) -> Optional[bool]:
    """Check whether the plan duration fits within the time budget.

    Args:
        plan_time_s: Plan duration in seconds.
        budget_s: Available time budget in seconds.

    Returns:
        True if plan_time_s <= budget_s, False if over budget, None if either
        value is unavailable.
    """
    if plan_time_s is None or budget_s is None:
        return None
    return plan_time_s <= budget_s
