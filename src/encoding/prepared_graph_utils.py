"""Graph indexing helpers for build_domain_from_prepared_info.

Derives lookup structures from a PreparedDomainInput's graph/transitions:
place adjacency by transition label, XOR branches grouped by activity, and
which attributes ever appear negated (needed to know which PDDL predicates
require a negative counterpart).
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from encoding.prepared_input import (
    GraphNode, PreparedDomainInput, PreparedXorBranch, TRANS_NODE_TYPES, PLACE_NODE_TYPES,
)
from models import Guard, PetriNetModel


def _prepared_graph_index(
    prepared: PreparedDomainInput,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build transition-label -> output/input place-label lookups.

    Returns:
        out_places_by_label: transition label -> output place labels.
        in_places_by_label: transition label -> input place labels.
    """
    node_by_id: Dict[str, GraphNode] = {n.id: n for n in prepared.nodes}

    out_places_by_label: Dict[str, List[str]] = {}
    in_places_by_label: Dict[str, List[str]] = {}

    # Pre-seed every transition-like node with an empty adjacency list, so
    # transitions with no output/input places still show up in the lookup.
    for n in prepared.nodes:
        if n.type in TRANS_NODE_TYPES and n.label:
            out_places_by_label.setdefault(n.label, [])
            in_places_by_label.setdefault(n.label, [])

    for edge in prepared.edges:
        src_node = node_by_id.get(edge.source)
        tgt_node = node_by_id.get(edge.target)
        if not src_node or not tgt_node:
            continue

        if src_node.type not in PLACE_NODE_TYPES:
            # transition -> place
            if src_node.label in out_places_by_label:
                out_places_by_label[src_node.label].append(tgt_node.label)
        else:
            # place -> transition
            if tgt_node.label in in_places_by_label:
                in_places_by_label[tgt_node.label].append(src_node.label)

    return out_places_by_label, in_places_by_label


def _prepared_xor_branch_of(
    prepared: PreparedDomainInput,
) -> Dict[str, List[PreparedXorBranch]]:
    """Flatten xor_branches (keyed by place) into activity_name -> branches.

    A transition can be downstream of several independent XOR splits, so it
    may receive more than one PreparedXorBranch.
    """
    result: Dict[str, List[PreparedXorBranch]] = defaultdict(list)
    for branches in prepared.xor_branches.values():
        for branch in branches:
            result[branch.activity_name].append(branch)
    return dict(result)


def _prepared_negated_attributes(prepared: PreparedDomainInput) -> Set[str]:
    """Collect attribute names that appear with Guard.negated=True anywhere.

    Used to decide which PDDL predicates need a negative counterpart
    (e.g. "risk_is_not" alongside "risk_is") — only attributes that are
    actually negated somewhere in a precondition/guard/xor-condition pay
    that cost.
    """
    negated: Set[str] = set()

    def _scan_sop(sop: List[List[Guard]]) -> None:
        for clause in sop:
            for guard in clause:
                if guard.negated:
                    negated.add(guard.attribute)

    for transition in prepared.transitions.values():
        _scan_sop(transition.preconditions)
        for group in transition.effect_groups:
            _scan_sop(group.guard)

    for branches in prepared.xor_branches.values():
        for branch in branches:
            _scan_sop(branch.conditions)

    return negated


def build_petrinet_model_from_prepared(
    prepared: PreparedDomainInput,
    start_place: str,
    end_place: str,
) -> PetriNetModel:
    """Build a real pm4py PetriNetModel from a PreparedDomainInput.

    Inverse of PreparedDomainInput.build_graph_from_petri_net — needed by the
    web GUI's partial-trace replay (replay.trace_replayer.TraceReplayer
    requires a genuine PetriNetModel), since the GUI's only structural source
    is a current.json-derived PreparedDomainInput, never a ParseResult with
    its own PetriNetModel.

    Nodes/edges are keyed by node.id (not label), matching how
    PreparedDomainInput.build_graph_from_petri_net produced them in the first
    place — a place's id is its pm4py place name, a transition's id is its
    pm4py transition name (label is None for silent transitions).
    """
    net = PetriNet("prepared")
    places: Dict[str, PetriNet.Place] = {}
    transitions: Dict[str, PetriNet.Transition] = {}

    for node in prepared.nodes:
        if node.type in PLACE_NODE_TYPES:
            place = PetriNet.Place(node.id)
            places[node.id] = place
            net.places.add(place)
        elif node.type in TRANS_NODE_TYPES:
            label = None if node.type == "silent" else node.label
            trans = PetriNet.Transition(node.id, label)
            transitions[node.id] = trans
            net.transitions.add(trans)

    for edge in prepared.edges:
        src = places.get(edge.source) or transitions.get(edge.source)
        tgt = places.get(edge.target) or transitions.get(edge.target)
        if src is not None and tgt is not None:
            add_arc_from_to(src, tgt, net)

    trans_inputs: Dict[PetriNet.Transition, Set[PetriNet.Place]] = {
        trans: {arc.source for arc in trans.in_arcs} for trans in transitions.values()
    }
    trans_outputs: Dict[PetriNet.Transition, Set[PetriNet.Place]] = {
        trans: {arc.target for arc in trans.out_arcs} for trans in transitions.values()
    }
    place_inputs: Dict[PetriNet.Place, List[PetriNet.Transition]] = {
        place: [arc.source for arc in place.in_arcs] for place in places.values()
    }

    silent_transitions: Dict[PetriNet.Transition, str] = {
        transitions[node.id]: node.label
        for node in prepared.nodes
        if node.type == "silent" and node.label and node.id in transitions
    }

    trans_by_label: Dict[str, PetriNet.Transition] = {
        node.label: transitions[node.id]
        for node in prepared.nodes
        if node.type in TRANS_NODE_TYPES and node.type != "silent"
        and node.label and node.id in transitions
    }
    xor_splits: Dict[PetriNet.Place, List[PetriNet.Transition]] = {}
    for place_name, branches in prepared.xor_branches.items():
        place = places.get(place_name)
        if place is None:
            continue
        branch_transitions = [
            trans_by_label[branch.activity_name]
            for branch in branches
            if branch.activity_name in trans_by_label
        ]
        if branch_transitions:
            xor_splits[place] = branch_transitions

    activities: Set[str] = set(trans_by_label.keys())

    initial_marking = Marking({places[start_place]: 1}) if start_place in places else Marking()
    final_marking = Marking({places[end_place]: 1}) if end_place in places else Marking()

    return PetriNetModel(
        petrinet=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        activities=activities,
        silent_transitions=silent_transitions,
        trans_inputs=trans_inputs,
        trans_outputs=trans_outputs,
        xor_splits=xor_splits,
        place_inputs=place_inputs,
    )
