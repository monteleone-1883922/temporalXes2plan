"""Graph indexing helpers for build_domain_from_prepared_info.

Derives lookup structures from a PreparedDomainInput's graph/transitions:
place adjacency by transition label, XOR branches grouped by activity, and
which attributes ever appear negated (needed to know which PDDL predicates
require a negative counterpart).
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from encoding.prepared_input import (
    GraphNode, PreparedDomainInput, PreparedXorBranch, TRANS_NODE_TYPES, PLACE_NODE_TYPES,
)
from models import Guard


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
