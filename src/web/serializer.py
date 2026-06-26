"""Serialize ParseResult and PetriNetModel into a JSON-friendly dict for the web UI.

The output format is designed for Cytoscape.js consumption (graph nodes/edges)
and for the detail panels (transitions, XOR splits, attribute catalog).
"""

from typing import Any, Dict, List, Optional, Set

import core_utils as utils
from models import (
    AttributeCatalogEntry,
    EffectInfo,
    Guard,
    ParseResult,
    PetriNetModel,
)


def serialize_parse_result(result: ParseResult) -> Dict[str, Any]:
    """Convert a ParseResult into a JSON-serializable dict for the web UI."""
    pnm = result.petri_net_model
    graph = _build_graph(pnm)
    transitions = _build_transitions(result)
    xor_splits = _build_xor_splits(pnm, result)
    catalog = _build_attribute_catalog(result.attribute_catalog)

    return {
        "graph": graph,
        "transitions": transitions,
        "xor_splits": xor_splits,
        "attribute_catalog": catalog,
        "metadata": {
            "start_place": result.start_place,
            "end_place": result.end_place,
        },
    }


# ---------------------------------------------------------------------------
# Graph (nodes + edges)
# ---------------------------------------------------------------------------

def _build_graph(pnm: PetriNetModel) -> Dict[str, Any]:
    """Build Cytoscape-ready node and edge lists from the Petri net."""
    xor_place_names: Set[str] = {p.name for p in pnm.xor_splits}

    and_transition_names: Set[str] = set()
    for t in pnm.petrinet.transitions:
        if not t.label:
            continue
        inputs = pnm.trans_inputs.get(t, set())
        outputs = pnm.trans_outputs.get(t, set())
        if len(inputs) > 1 or len(outputs) > 1:
            and_transition_names.add(utils.sanitize_name(t.label))

    nodes: List[Dict[str, Any]] = []

    for p in pnm.petrinet.places:
        node_type = "xor_split" if p.name in xor_place_names else "place"
        nodes.append({"id": p.name, "type": node_type, "label": p.name})

    for t in pnm.petrinet.transitions:
        t_id = t.name if t.name else str(id(t))
        is_silent = t.label is None
        sanitized = utils.sanitize_name(t.label) if t.label else None

        if is_silent:
            node_type = "silent"
            label = pnm.silent_transitions.get(t, "")
        elif sanitized in and_transition_names:
            node_type = "and_split"
            label = sanitized
        else:
            node_type = "transition"
            label = sanitized

        nodes.append({
            "id": t_id,
            "type": node_type,
            "label": label or "",
            "is_silent": is_silent,
        })

    edges: List[Dict[str, str]] = []
    for arc in pnm.petrinet.arcs:
        src = arc.source
        tgt = arc.target
        src_id = src.name if hasattr(src, "name") else str(id(src))
        tgt_id = tgt.name if hasattr(tgt, "name") else str(id(tgt))
        edges.append({"source": src_id, "target": tgt_id})

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def _guard_to_condition(g: Guard) -> Dict[str, Any]:
    """Convert a Guard dataclass to the simplified UI condition format."""
    predicate = "<>" if g.negated else "="
    value = g.value if g.value is not None else "true"
    return {"attribute": g.attribute, "predicate": predicate, "value": value}


def _guards_to_preconditions(
    guards: Optional[List[List[Guard]]],
) -> List[List[Dict[str, Any]]]:
    """Convert SOP guards (OR of AND) to UI precondition format."""
    if not guards:
        return []
    return [
        [_guard_to_condition(g) for g in and_clause]
        for and_clause in guards
    ]


def _build_effects_lookup(effects: Dict[str, EffectInfo]) -> Dict[str, Any]:
    """Build a nested dict {attr: {value: {probability, guard}}} for UI lookup.

    This is the read-only view of per-(attribute, value) details.  The frontend
    uses it to retrieve probability and guard for each assignment in an effect group.

    Args:
        effects: Per-attribute EffectInfo map from TransitionInfo.

    Returns:
        Nested dict keyed by attribute then value.
    """
    lookup: Dict[str, Any] = {}
    for attr, info in effects.items():
        value_guards_map: Dict[str, List[List[Guard]]] = {}
        if info.value_guards and info.value_guards.guards:
            value_guards_map = info.value_guards.guards

        attr_entry: Dict[str, Any] = {}
        for value, probability in info.value_probabilities.items():
            val_str = str(value)
            guard = _guards_to_preconditions(value_guards_map.get(val_str))
            attr_entry[val_str] = {
                "probability": round(probability, 4),
                "guard": guard,
            }
        lookup[attr] = attr_entry
    return lookup


def _combine_raw_sop(
    sop_a: List[List[Guard]],
    sop_b: List[List[Guard]],
) -> List[List[Guard]]:
    """Combine two raw SOP guard lists via Cartesian product.

    Each pair (AND-clause from sop_a, AND-clause from sop_b) is merged into a
    single AND-clause in the result, preserving the OR structure.  This mirrors
    the logic used by EffectDuplicator when combining guards from independent
    attributes into a single variant.

    Args:
        sop_a: First list of AND-clauses (OR of ANDs).
        sop_b: Second list of AND-clauses (OR of ANDs).

    Returns:
        Cartesian product SOP: every pair (a, b) merged into one AND-clause.
        If either list is empty the other is returned unchanged.
    """
    if not sop_a:
        return sop_b
    if not sop_b:
        return sop_a
    result = {frozenset(clause_a + clause_b) for clause_a in sop_a for clause_b in sop_b}
    return [list(clause) for clause in result]


def _raw_sop_for_effect(
    val_str: str,
    info: EffectInfo,
) -> List[List[Guard]]:
    """Collect the raw SOP guard clauses for one (attribute, value) pair.

    Combines appearance guards (2A) and value guards (2B) via Cartesian product,
    since both conditions must hold for the effect to fire.

    Args:
        val_str: Stringified attribute value.
        info: EffectInfo for the attribute.

    Returns:
        List of AND-clauses (OR of ANDs) in raw Guard form.
    """
    appearance_clauses: List[List[Guard]] = []
    if info.appearance_guards and info.appearance_guards.guards:
        appearance_clauses = info.appearance_guards.guards.get("appears", [])

    value_clauses: List[List[Guard]] = []
    if info.value_guards and info.value_guards.guards:
        value_clauses = info.value_guards.guards.get(val_str, [])

    return _combine_raw_sop(appearance_clauses, value_clauses)


def _build_effect_groups(
    groups: List[List[tuple]],
    effects_info: Dict[str, EffectInfo],
) -> List[Dict[str, Any]]:
    """Serialize effect_groups with per-group guard and probability.

    For each group (a list of (attr, val) pairs that fire together in one
    action variant), computes:

    - guard: Cartesian-product combination of each attribute's appearance and
      value guards, converted to the UI SOP format.
    - probability: product of each attribute's value probability, multiplied
      by presence_probability when the appearance phase is probabilistic
      (appearance_level == 2).

    Args:
        groups: List of groups produced by _extract_effect_groups; each group
            is a sorted list of (attribute, value) tuples.
        effects_info: Per-attribute EffectInfo map from TransitionInfo.effects.

    Returns:
        List of dicts with keys: assignments, guard, probability.
    """
    result: List[Dict[str, Any]] = []

    for group in groups:
        combined_raw: List[List[Guard]] = []
        probability = 1.0

        for attr, val in group:
            val_str = str(val)
            info = effects_info.get(attr)
            if info is None:
                continue

            attr_raw = _raw_sop_for_effect(val_str, info)
            combined_raw = _combine_raw_sop(combined_raw, attr_raw)

            val_prob = info.value_probabilities.get(val, info.value_probabilities.get(val_str, 1.0))
            if info.appearance_level == 2:
                val_prob *= info.presence_probability
            probability *= val_prob

        result.append({
            "assignments": [{"attribute": attr, "value": str(val)} for attr, val in group],
            "guard": _guards_to_preconditions(combined_raw),
            "probability": round(probability, 4),
        })

    return result


def _build_transitions(result: ParseResult) -> Dict[str, Any]:
    """Build the transitions section of the UI JSON."""
    output: Dict[str, Any] = {}

    for act, t_info in result.transitions.items():
        preconditions: List[List[Dict[str, Any]]] = []
        if t_info.attribute_preconditions:
            preconditions = [[_guard_to_condition(g) for g in t_info.attribute_preconditions]]

        effects_lookup = _build_effects_lookup(t_info.effects)
        effect_groups = _build_effect_groups(t_info.effect_groups, t_info.effects)

        duration = None
        if t_info.duration:
            d = t_info.duration
            duration = {
                "effective_min": d.effective_min,
                "effective_max": d.effective_max,
                "source": d.source,
            }
            if d.mean is not None:
                duration["mean"] = round(d.mean, 2)
                duration["std_dev"] = round(d.std_dev, 2)

        meta: Dict[str, Any] = {
            "total_firings": t_info.total_firings,
            "related_effects": [
                sorted(pair) for pair in t_info.related_effects
            ],
            "incompatible_effects": [
                sorted(pair) for pair in t_info.incompatible_effects
            ],
        }
        if t_info.xor_branch:
            meta["xor_branch"] = {
                "probability": round(t_info.xor_branch.probability, 4),
                "total_samples": t_info.xor_branch.total_samples,
                "cascade_level": t_info.xor_branch.cascade_level,
            }

        output[act] = {
            "activity_name": act,
            "input_places": t_info.input_places,
            "preconditions": preconditions,
            "effects": effects_lookup,
            "effect_groups": effect_groups,
            "cost": 0.0,
            "duration": duration,
            "_meta": meta,
        }

    return output


# ---------------------------------------------------------------------------
# XOR splits
# ---------------------------------------------------------------------------

def _build_xor_splits(
    pnm: PetriNetModel, result: ParseResult
) -> Dict[str, Any]:
    """Build XOR split section keyed by place name."""
    output: Dict[str, Any] = {}

    for place, transitions in pnm.xor_splits.items():
        branches: Dict[str, Any] = {}
        for t in transitions:
            if not t.label:
                continue
            act = utils.sanitize_name(t.label)
            t_info = result.transitions.get(act)
            if not t_info or not t_info.xor_branch:
                continue

            xb = t_info.xor_branch
            conditions = _guards_to_preconditions(xb.guards)

            branches[act] = {
                "activity_name": act,
                "conditions": conditions,
                "probability": round(xb.probability, 4),
                "_meta": {
                    "cascade_level": xb.cascade_level,
                    "total_samples": xb.total_samples,
                },
            }

        if branches:
            output[place.name] = {"branches": branches}

    return output


# ---------------------------------------------------------------------------
# Attribute catalog
# ---------------------------------------------------------------------------

def _build_attribute_catalog(
    catalog: Dict[str, AttributeCatalogEntry],
) -> Dict[str, Any]:
    """Serialize the attribute catalog for dropdown population."""
    return {
        attr: {
            "type": entry.attribute_type,
            "possible_values": sorted(str(v) for v in entry.possible_values),
            "bin_boundaries": entry.bin_boundaries or [],
        }
        for attr, entry in catalog.items()
    }
