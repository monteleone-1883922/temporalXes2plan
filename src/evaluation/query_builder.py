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

import heapq
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from encoding.prepared_input import PLACE_NODE_TYPES, TRANS_NODE_TYPES
from evaluation.trace_sampler import PrefixSample, _remaining_budget

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

# _remaining_budget lives in trace_sampler.py (imported above) — it is also
# needed there, inside sample_prefix, to compute PrefixSample.reached_within_time
# from the same raw duration fields before the PrefixSample itself exists, and
# trace_sampler.py cannot import from this module (query_builder.py already
# imports PrefixSample from trace_sampler.py, so the reverse import would be
# circular). Re-imported here (not redefined) so every existing caller of
# evaluation.query_builder._remaining_budget keeps working unchanged.


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



def _index_graph(
    serialized: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Index serialized["graph"] once: node lookup + transition in/out places.

    Shared by is_q3_reachable and compute_min_time_to_end — both are
    structural, condition-free graph walks over the exact same
    place<->transition adjacency, one for plain reachability (BFS), the
    other for minimum modeled time (Dijkstra).

    Returns:
        (node_by_id, in_places, out_places):
        - node_by_id: node id -> its full node dict.
        - in_places: transition id -> input place ids (place -> transition edges).
        - out_places: transition id -> output place ids (transition -> place edges).
    """
    nodes: List[Dict[str, Any]] = serialized.get("graph", {}).get("nodes", [])
    edges: List[Dict[str, Any]] = serialized.get("graph", {}).get("edges", [])

    node_by_id: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes}

    in_places: Dict[str, List[str]] = {}
    out_places: Dict[str, List[str]] = {}
    for edge in edges:
        src = node_by_id.get(edge["source"])
        tgt = node_by_id.get(edge["target"])
        if src is None or tgt is None:
            continue
        if src["type"] in PLACE_NODE_TYPES and tgt["type"] in TRANS_NODE_TYPES:
            in_places.setdefault(tgt["id"], []).append(src["id"])
        elif src["type"] in TRANS_NODE_TYPES and tgt["type"] in PLACE_NODE_TYPES:
            out_places.setdefault(src["id"], []).append(tgt["id"])

    return node_by_id, in_places, out_places


def compute_min_time_to_end(
    serialized: Dict[str, Any],
    init_places: List[str],
    cache: Optional[Dict[FrozenSet[str], float]] = None,
) -> float:
    """Minimum modeled time to reach the end place from this specific marking.

    Purely structural and condition-free, same philosophy as is_q3_reachable
    (ignores guards, XOR probabilities), but respects AND-join
    synchronization: a transition with k input places cannot fire until
    ALL k of them have been reached, and it fires at the time of the
    SLOWEST (max) of them, not the fastest — "non si può eseguire la
    transizione finché tutti i places non sono marcati". This is why the
    result depends on the full marking (init_places) and cannot be reduced
    to independent per-place values computed once for the whole domain: a
    transition may become fireable only because SOME OTHER, unrelated
    branch of this specific marking also reaches its remaining input(s) —
    information a single place, considered in isolation, cannot have.

    Algorithm: a generalization of Dijkstra to AND-join "hyperedges". Every
    place starts at distance 0 once reached; a transition is only relaxed
    (its output places' distances updated) once ALL of its input places
    have been settled, using the MAX of their settled distances as its own
    ready time, plus its own effective_max duration. effective_min/max are
    not planner-chosen values here -- they are the observed range of how
    long that activity actually took across the historical log, i.e. real
    execution-time uncertainty the planner does not control. A plan is
    only genuinely reliable if it still meets the deadline in the worst
    realistic case, so effective_max (not effective_min) is used: OPTIC's
    own temporal-feasibility checking is understood to reason the same
    way. Transitions with no mined duration data contribute 0 (absence of
    information treated as non-constraining, same convention as
    is_q3_reachable and _sum_modeled_max_duration). Transitions with no
    input places at all (net sources) are seeded as fireable at time 0.

    Args:
        serialized: current.json dict.
        init_places: Place names holding a token in the marking to check
            (e.g. QuerySpec.init_places / PrefixSample.init_places).
        cache: Optional dict shared across calls (e.g. one built per log,
            outside the per-trace loop), keyed by the exact marking
            (frozenset(init_places)) -- NOT per-place, which is what makes
            this safe: two Q2 queries whose traces happen to produce the
            EXACT SAME marking get the identical answer by construction
            (same serialized domain, same init_places -> same result), so
            memoizing on the full marking loses no AND-join information
            (unlike a per-place table, which would -- see this function's
            git history / claude_plans/time_aware_evaluation_plan.md §5
            for why a per-place cache was rejected). A cache miss still
            costs one full Dijkstra pass; only identical markings are free.

    Returns:
        Minimum modeled time (seconds) to reach the end place from this
        marking, or float("inf") if structurally unreachable (or no
        metadata.end_place is declared).
    """
    cache_key = frozenset(init_places) if cache is not None else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    node_by_id, in_places_map, out_places = _index_graph(serialized)
    transitions: Dict[str, Any] = serialized.get("transitions", {})
    end_place = serialized.get("metadata", {}).get("end_place")
    if end_place is None:
        if cache is not None:
            cache[cache_key] = float("inf")
        return float("inf")

    transition_ids = [nid for nid, n in node_by_id.items() if n["type"] in TRANS_NODE_TYPES]
    transitions_by_input_place: Dict[str, List[str]] = {}
    for tid in transition_ids:
        for p in in_places_map.get(tid, []):
            transitions_by_input_place.setdefault(p, []).append(tid)

    # Per-transition AND-join bookkeeping: how many of its inputs are still
    # unsettled, and the max distance seen so far among its settled inputs.
    remaining_inputs: Dict[str, int] = {tid: len(in_places_map.get(tid, [])) for tid in transition_ids}
    ready_time_so_far: Dict[str, float] = {tid: 0.0 for tid in transition_ids}

    dist: Dict[str, float] = {}
    settled: set = set()
    heap: List[Tuple[float, str]] = []

    def _fire(tid: str, ready_time: float) -> None:
        """Relax tid's output places once all of its inputs are settled."""
        t_info = transitions.get(node_by_id[tid].get("label"))
        duration = t_info.get("duration") if t_info else None
        dmax = duration["effective_max"] if duration else 0.0
        fire_time = ready_time + dmax
        for out_p in out_places.get(tid, []):
            if fire_time < dist.get(out_p, float("inf")):
                dist[out_p] = fire_time
                heapq.heappush(heap, (fire_time, out_p))

    for p in init_places:
        if p not in dist:
            dist[p] = 0.0
            heapq.heappush(heap, (0.0, p))

    # Net sources (no input places at all) fire unconditionally at time 0 —
    # same convention as is_q3_reachable's own unconditional-source handling.
    for tid in transition_ids:
        if remaining_inputs[tid] == 0:
            _fire(tid, 0.0)

    while heap:
        d, place = heapq.heappop(heap)
        if place in settled:
            continue
        if d > dist.get(place, float("inf")):
            continue  # stale heap entry, a shorter distance already settled it
        settled.add(place)
        for tid in transitions_by_input_place.get(place, []):
            if ready_time_so_far[tid] < d:
                ready_time_so_far[tid] = d  # MAX over this transition's inputs
            remaining_inputs[tid] -= 1
            if remaining_inputs[tid] == 0:
                _fire(tid, ready_time_so_far[tid])

    result = dist.get(end_place, float("inf"))
    if cache is not None:
        cache[cache_key] = result
    return result


def is_q3_reachable(
    goal_sop: List[List[Dict[str, Any]]],
    serialized: Dict[str, Any],
    init_places: List[str],
    init_effects: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Check whether the goal is reachable from the current token marking.

    Performs a real BFS graph visit of serialized["graph"] (nodes/edges),
    starting from init_places (the marking after replaying the trace prefix
    so far): a newly-reached place is pushed onto a frontier queue, and each
    place popped from the frontier tries to fire every transition that has it
    as one of its inputs (only once ALL of that transition's inputs are
    reachable). A transition firing marks its output places reached — pushing
    any NEW ones onto the frontier so they, in turn, retry every transition
    they feed — and records every (attribute, value) pair from its
    effect_groups' assignments as a reachable effect. Each transition is
    tried exactly as many times as one of its own inputs newly becomes
    reachable (bounded by its in-degree) and fires at most once, giving
    O(places + edges) total work — unlike a plain "rescan every transition
    until nothing changes" fixpoint, which needs a full extra pass over every
    transition per newly-reached place in the worst case. Attribute values
    already fixed by the replayed prefix (init_effects) are seeded into the
    same set up front — they are already true in the PDDL initial state (see
    run_evaluation.py's build_problem(init_effects=...) call), so a clause
    condition they satisfy must not be treated as unreachable just because
    the transition that set them no longer fires forward from init_places
    (e.g. it consumed a place, like the process's start place, that isn't
    part of the current marking anymore).

    The goal is reachable if, once no more transitions can fire: (1) the
    model's end place was reached, and (2) at least one AND-clause of
    goal_sop has every (attribute, value) condition covered either by
    init_effects or by an effect from a transition that actually fired
    during the visit (not merely present somewhere in the net).

    This is still a structural over-approximation, not full token-flow model
    checking: it ignores exact token counts, firing order, and AND-join
    simultaneity, and just asks "can this place/effect ever be reached at
    all". It is used to populate ground_truth_reachable in result records,
    and (in run_evaluation.py) to skip invoking the planner on Q3 queries
    whose goal is structurally impossible from the current marking.

    Args:
        goal_sop: Goal in SOP form (list of AND-clauses).
        serialized: current.json dict.
        init_places: Place names holding a token in the current marking.
        init_effects: Attribute values already fixed by the replayed prefix
            (QuerySpec.init_effects / PrefixSample.init_effects) — each a
            {"attribute": str, "value": str} dict. Treated as already
            satisfied, same as the PDDL problem's initial state.

    Returns:
        True if the end place is reachable and at least one AND-clause is
        fully covered by init_effects plus effects of transitions reachable
        from init_places.
    """
    node_by_id, in_places, out_places = _index_graph(serialized)
    transitions: Dict[str, Any] = serialized.get("transitions", {})

    transition_ids = [nid for nid, n in node_by_id.items() if n["type"] in TRANS_NODE_TYPES]

    # Reverse index: place -> transitions that read it as an input, so a
    # newly-reached place can directly retry only the transitions it feeds,
    # instead of rescanning every transition in the net.
    transitions_by_input_place: Dict[str, List[str]] = {}
    for tid, inputs in in_places.items():
        for p in inputs:
            transitions_by_input_place.setdefault(p, []).append(tid)

    reachable_places: set[str] = set(init_places)
    fired: set[str] = set()
    matched_pairs: set[tuple[str, str]] = {
        (e["attribute"], e["value"]) for e in (init_effects or [])
    }

    frontier: deque[str] = deque(init_places)

    def _try_fire(tid: str) -> None:
        """Fire tid if not already fired and every input place is reachable
        now; push any newly-reached output place onto frontier so it
        propagates to whatever it feeds, in turn."""
        if tid in fired:
            return
        inputs = in_places.get(tid, [])
        if inputs and not all(p in reachable_places for p in inputs):
            return
        fired.add(tid)
        for out_p in out_places.get(tid, []):
            if out_p not in reachable_places:
                reachable_places.add(out_p)
                frontier.append(out_p)
        t_info = transitions.get(node_by_id[tid].get("label"))
        if t_info:
            for group in t_info.get("effect_groups", []):
                for assignment in group.get("assignments", []):
                    matched_pairs.add((assignment["attribute"], assignment["value"]))

    # Transitions with no input places at all (net sources) fire
    # unconditionally -- try them once up front so their outputs enter the
    # BFS too; everything else only fires once one of its own inputs shows
    # up in the frontier below.
    for tid in transition_ids:
        if not in_places.get(tid):
            _try_fire(tid)

    while frontier:
        place = frontier.popleft()
        for tid in transitions_by_input_place.get(place, []):
            _try_fire(tid)

    end_place = serialized.get("metadata", {}).get("end_place")
    end_reachable = end_place is None or end_place in reachable_places
    if not end_reachable:
        return False

    for clause in goal_sop:
        if not clause:
            continue
        if all((c["attribute"], c["value"]) in matched_pairs for c in clause):
            return True

    return False
