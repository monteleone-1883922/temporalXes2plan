"""
Shared factory functions for all test modules.

Two groups of helpers:
  - EventLog builders (make_event, make_trace, make_log): used by tests for
    temporal_extractor, correlation_miner, and decision_mining.
  - PetriNet builders (build_sequence_net, build_xor_net, build_and_net): used
    by tests for structure_analyzer.

All helpers construct pm4py objects programmatically — no XES files are read.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import PetriNet


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def ts(offset_seconds: int = 0, tz_aware: bool = True) -> datetime:
    """
    Return a datetime anchored at 2024-01-01 08:00:00 UTC, shifted by offset_seconds.

    Args:
        offset_seconds: Number of seconds to add to the base timestamp.
        tz_aware: If True, attach UTC timezone info (timezone-aware datetime).
                  If False, return a naive datetime (no tzinfo).
    """
    base = datetime(2024, 1, 1, 8, 0, 0)
    dt = base + timedelta(seconds=offset_seconds)
    return dt.replace(tzinfo=timezone.utc) if tz_aware else dt


# ---------------------------------------------------------------------------
# EventLog factories
# ---------------------------------------------------------------------------

def make_event(
    name: str,
    lifecycle: Optional[str] = None,
    offset: int = 0,
    tz_aware: bool = True,
    **attrs,
) -> Event:
    """
    Create a pm4py Event with a given activity name, optional lifecycle transition,
    a timestamp computed as base + offset seconds, and arbitrary extra attributes.

    Args:
        name: Value for concept:name (the activity label).
        lifecycle: Value for lifecycle:transition (e.g. "start", "complete").
                   If None the key is omitted, simulating logs without lifecycle info.
        offset: Seconds to add to the base timestamp (2024-01-01 08:00:00).
        tz_aware: Whether the timestamp should be timezone-aware.
        **attrs: Any additional event attributes (e.g. fever="high", crp=3.2).
    """
    e = Event()
    e["concept:name"] = name
    if lifecycle is not None:
        e["lifecycle:transition"] = lifecycle
    e["time:timestamp"] = ts(offset, tz_aware=tz_aware)
    for k, v in attrs.items():
        e[k] = v
    return e


def make_trace(*events: Event, case_id: str = "case_1") -> Trace:
    """
    Assemble a pm4py Trace from individual Event objects.

    Sets trace.attributes["concept:name"] so that pm4py.convert_to_dataframe
    can add a 'case:concept:name' column correctly.

    Args:
        *events: Event objects to append in order.
        case_id: The case identifier stored as a trace attribute.
    """
    t = Trace()
    t.attributes["concept:name"] = case_id
    for e in events:
        t.append(e)
    return t


def make_log(*traces: Trace) -> EventLog:
    """
    Assemble a pm4py EventLog from individual Trace objects.

    Args:
        *traces: Trace objects to include in the log.
    """
    log = EventLog()
    for t in traces:
        log.append(t)
    return log


# ---------------------------------------------------------------------------
# PetriNet factories
# ---------------------------------------------------------------------------

def _transition(name: str, label: Optional[str] = None) -> PetriNet.Transition:
    """
    Create a PetriNet.Transition.

    Args:
        name: Internal identifier (must be unique in the net).
        label: Human-readable label (the activity name). Pass None for a silent
               (tau) transition that carries no observable activity.
    """
    return PetriNet.Transition(name, label)


def _place(name: str) -> PetriNet.Place:
    """Create a PetriNet.Place with the given identifier."""
    return PetriNet.Place(name)


def _arc(source, target) -> PetriNet.Arc:
    """Create a directed PetriNet.Arc from source node to target node."""
    return PetriNet.Arc(source, target)


def build_sequence_net(
    activity_names: List[str],
) -> Tuple[Set[PetriNet.Arc], Set[PetriNet.Place], List[PetriNet.Transition], Dict]:
    """
    Build a linear Petri net connecting activities in the given order:
        p0 -> tA -> p1 -> tB -> p2 -> tC -> p3 ...

    Each transition is labeled with its activity name (as given).
    No silent transitions are created.

    Args:
        activity_names: Ordered list of activity names for the sequence.

    Returns:
        Tuple of (arcs, places, transitions, silent_transitions).
        silent_transitions is an empty dict because this net has no tau transitions.
    """
    transitions = [_transition(f"t_{n}", n) for n in activity_names]
    places = [_place(f"p{i}") for i in range(len(activity_names) + 1)]
    arcs: Set[PetriNet.Arc] = set()
    for i, t in enumerate(transitions):
        arcs.add(_arc(places[i], t))
        arcs.add(_arc(t, places[i + 1]))
    return arcs, set(places), transitions, {}


def build_xor_net(
    source_name: str,
    branch_names: List[str],
) -> Tuple[Set[PetriNet.Arc], Set[PetriNet.Place], List[PetriNet.Transition], Dict]:
    """
    Build a XOR-split net: source fires into a single shared place,
    from which N alternative branch transitions can fire exclusively.

        p_in -> t_source -> p_xor -> t_B
                                  -> t_C

    The shared place (p_xor) has multiple outgoing transitions, which makes it a
    choice / exclusive-or split as recognised by StructureAnalyzer.

    Args:
        source_name: Activity name for the source transition.
        branch_names: Activity names for the alternative branches.

    Returns:
        Tuple of (arcs, places, [t_source, *branch_ts], silent_transitions={}).
    """
    t_src = _transition(f"t_{source_name}", source_name)
    p_in = _place("p_in")
    p_xor = _place("p_xor")
    branch_ts = [_transition(f"t_{n}", n) for n in branch_names]
    branch_out_places = [_place(f"p_out_{n}") for n in branch_names]

    arcs: Set[PetriNet.Arc] = {
        _arc(p_in, t_src),
        _arc(t_src, p_xor),
        *[_arc(p_xor, bt) for bt in branch_ts],
        *[_arc(bt, bp) for bt, bp in zip(branch_ts, branch_out_places)],
    }
    all_places = {p_in, p_xor} | set(branch_out_places)
    return arcs, all_places, [t_src, *branch_ts], {}


def build_and_net(
    source_name: str,
    branch_names: List[str],
) -> Tuple[Set[PetriNet.Arc], Set[PetriNet.Place], List[PetriNet.Transition], Dict]:
    """
    Build an AND-split net: source fires into N separate places, each of which
    leads to a distinct parallel branch transition.

        p_in -> t_source -> p_1 -> t_B -> p_out_B
                         -> p_2 -> t_C -> p_out_C

    Each branch place has exactly one outgoing transition (no choice), so none of
    them qualifies as a decision-point place — StructureAnalyzer will therefore
    classify the source as an AND-split.

    Args:
        source_name: Activity name for the AND-split source transition.
        branch_names: Activity names for the parallel branch transitions.

    Returns:
        Tuple of (arcs, places, [t_source, *branch_ts], silent_transitions={}).
    """
    t_src = _transition(f"t_{source_name}", source_name)
    p_in = _place("p_in")
    branch_places = [_place(f"p_{n}") for n in branch_names]
    branch_ts = [_transition(f"t_{n}", n) for n in branch_names]
    out_places = [_place(f"p_out_{n}") for n in branch_names]

    arcs: Set[PetriNet.Arc] = {
        _arc(p_in, t_src),
        *[_arc(t_src, bp) for bp in branch_places],
        *[_arc(bp, bt) for bp, bt in zip(branch_places, branch_ts)],
        *[_arc(bt, op) for bt, op in zip(branch_ts, out_places)],
    }
    all_places = {p_in} | set(branch_places) | set(out_places)
    return arcs, all_places, [t_src, *branch_ts], {}
