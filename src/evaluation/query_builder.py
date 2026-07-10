"""Build Q1/Q2/Q3 QuerySpec objects from a PrefixSample.

Each query maps to a specific EvalAPI.build_problem() call:

- Q1: reach end place, no deadline.
- Q2: reach end place within the remaining time budget (TIL deadline).
- Q3: reach end place + attribute conditions from the final trace event, with deadline.

All queries use metric="minimize_weighted" so that time dominates and cost acts as
a tiebreaker. Q2 and Q3 are skipped (return None) when no timestamp budget is available.
Q3 is also skipped when the final event contains no discretized attributes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.trace_sampler import PrefixSample

logger = logging.getLogger(__name__)


@dataclass
class QuerySpec:
    """Parameters for a single planner invocation.

    Attributes:
        query_type: One of "Q1", "Q2", "Q3".
        problem_name: PDDL problem name.
        init_places: Token marking derived from prefix replay.
        init_effects: Attribute values derived from prefix replay.
        goal_sop: Goal in Sum-of-Products form (outer=OR, inner=AND).
        metric: Always "minimize_weighted".
        cost_weight: Scaling factor α for the weighted metric.
        require_completion: Always True — every goal clause includes the end place.
        deadline: TIL value in seconds (None → no deadline, i.e. Q1).
    """

    query_type: str
    problem_name: str
    init_places: List[str]
    init_effects: List[Dict[str, Any]]
    goal_sop: List[List[Dict[str, Any]]]
    metric: str
    cost_weight: float
    require_completion: bool
    deadline: Optional[float]


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_q1(
    prefix_sample: PrefixSample,
    cost_weight: float = 0.001,
) -> QuerySpec:
    """Build Q1: process completion, no deadline.

    Args:
        prefix_sample: Output of trace_sampler.sample_prefix().
        serialized: current.json dict (output of serialize_parse_result).
        cost_weight: Scaling factor α for the weighted metric.

    Returns:
        QuerySpec with no deadline and an empty goal clause (completion only).
    """
    return QuerySpec(
        query_type="Q1",
        problem_name="q1_problem",
        init_places=prefix_sample.init_places,
        init_effects=prefix_sample.init_effects,
        goal_sop=[[]],
        metric="minimize_weighted",
        cost_weight=cost_weight,
        require_completion=True,
        deadline=None,
    )


def build_q2(
    prefix_sample: PrefixSample,
    cost_weight: float = 0.001,
) -> Optional[QuerySpec]:
    """Build Q2: process completion within the remaining time budget.

    Returns None when no timestamp information is available (budget is unknown).

    Args:
        prefix_sample: Output of trace_sampler.sample_prefix().
        serialized: current.json dict.
        cost_weight: Scaling factor α for the weighted metric.

    Returns:
        QuerySpec with a TIL deadline, or None if budget cannot be computed.
    """
    budget = _remaining_budget(prefix_sample)
    if budget is None:
        logger.debug("Q2 skipped: no timestamp data available for budget computation.")
        return None

    return QuerySpec(
        query_type="Q2",
        problem_name="q2_problem",
        init_places=prefix_sample.init_places,
        init_effects=prefix_sample.init_effects,
        goal_sop=[[]],
        metric="minimize_weighted",
        cost_weight=cost_weight,
        require_completion=True,
        deadline=budget,
    )


def build_q3(
    prefix_sample: PrefixSample,
    serialized: Dict[str, Any],
    cost_weight: float = 0.001,
) -> Optional[QuerySpec]:
    """Build Q3: completion within budget with attribute constraints from the final event.

    Returns None when:
    - No timestamp budget is available (same reason as Q2).
    - The final event contains no discretized attributes in the catalog.

    Args:
        prefix_sample: Output of trace_sampler.sample_prefix().
        serialized: current.json dict.
        cost_weight: Scaling factor α for the weighted metric.

    Returns:
        QuerySpec with deadline and attribute goal, or None if skipped.
    """
    budget = _remaining_budget(prefix_sample)
    if budget is None:
        logger.debug("Q3 skipped: no timestamp data available for budget computation.")
        return None

    catalog = serialized.get("attribute_catalog", {})
    goal_sop = _goal_from_final_event(prefix_sample.final_event_attributes, catalog)

    if not goal_sop or not goal_sop[0]:
        logger.warning(
            "Q3 skipped: final event has no discretized attributes in the catalog. "
            "Recorded as solvability='skipped_no_attributes'."
        )
        return None

    return QuerySpec(
        query_type="Q3",
        problem_name="q3_problem",
        init_places=prefix_sample.init_places,
        init_effects=prefix_sample.init_effects,
        goal_sop=goal_sop,
        metric="minimize_weighted",
        cost_weight=cost_weight,
        require_completion=True,
        deadline=budget,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remaining_budget(prefix_sample: PrefixSample) -> Optional[float]:
    """Compute the remaining time budget for the suffix.

    Args:
        prefix_sample: A PrefixSample with optional duration fields.

    Returns:
        full_duration_s - prefix_duration_s in seconds, or None if either is None.
    """
    if prefix_sample.full_duration_s is None or prefix_sample.prefix_duration_s is None:
        return None
    return prefix_sample.full_duration_s - prefix_sample.prefix_duration_s


def _goal_from_final_event(
    final_attrs: Dict[str, Any],
    attribute_catalog: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """Build a single-AND-clause goal from the final prefix event's attributes.

    final_attrs comes from TraceReplayer.replay_evaluation_split()'s
    expected_final_attributes — the log-observed accumulated state (see
    trace_replayer.py's _build_snapshots/_normalize_event_value) — which
    already carries values in the *same representation* as
    attribute_catalog's possible_values: numerical values are already
    discretized into their bin label (e.g. "gte_10_lte_20"), and categorical
    values are the raw, unsanitized strings observed in the log (PDDL
    sanitization only happens later, at problem/domain render time via
    core_utils.sanitize_value — see prepared_schema_builder.py). So matching
    is a direct string comparison for both — no re-discretization, no
    sanitize_name(), which previously always mismatched (re-discretizing an
    already-discretized label raised ValueError; lowercasing a categorical
    value before comparing it against possible_values' original casing never
    matched) and made every Q3 goal come out empty in practice.

    Only attributes present in the catalog are included. Boolean attributes
    are matched case-insensitively. Attributes whose value does not match any
    known catalog value are skipped.

    Args:
        final_attrs: Dict of attribute name → value from the final replayed
            state (see docstring above for its representation).
        attribute_catalog: Serialized catalog from current.json.

    Returns:
        One-element list containing the AND-clause: [[cond, cond, ...]].
        The inner list may be empty if no attribute matches.
    """
    conditions: List[Dict[str, Any]] = []

    for attr, raw_value in final_attrs.items():
        entry = attribute_catalog.get(attr)
        if entry is None:
            continue

        attr_type = entry.get("type", "")
        possible_values: List[str] = entry.get("possible_values", [])

        matched_label: Optional[str] = None

        if attr_type == "boolean":
            str_val = str(raw_value).lower()
            if str_val in ("true", "false") and str_val in possible_values:
                matched_label = str_val
        else:
            str_val = str(raw_value)
            if str_val in possible_values:
                matched_label = str_val

        if matched_label is not None:
            conditions.append({
                "attribute": attr,
                "predicate": "=",
                "value": matched_label,
            })

    return [conditions]



def is_q3_reachable(
    goal_sop: List[List[Dict[str, Any]]],
    serialized: Dict[str, Any],
) -> bool:
    """Check whether at least one goal clause is structurally reachable.

    For each AND-clause, every attribute condition (attr = value) is checked
    against the transitions in serialized["transitions"]: at least one
    transition must have an effect that can produce (attr = value). The clause
    is reachable only if all its conditions are covered. Returns True if at
    least one clause is fully covered.

    This is a structural pre-check only — it does not verify token-flow
    reachability. It is used to populate ground_truth_reachable in result
    records; planning always runs regardless.

    Args:
        goal_sop: Goal in SOP form (list of AND-clauses).
        serialized: current.json dict.

    Returns:
        True if at least one AND-clause is fully covered by transition effects.
    """
    transitions: Dict[str, Any] = serialized.get("transitions", {})

    # Build a set of (attr, value) pairs producible by any transition effect.
    reachable_pairs: set[tuple[str, str]] = set()
    for t_info in transitions.values():
        effects: Dict[str, Any] = t_info.get("effects", {})
        for attr, value_map in effects.items():
            for value_str in value_map:
                reachable_pairs.add((attr, value_str))

    for clause in goal_sop:
        if not clause:
            continue
        if all((c["attribute"], c["value"]) in reachable_pairs for c in clause):
            return True

    return False
