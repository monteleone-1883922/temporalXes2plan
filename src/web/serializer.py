"""Serialize ParseResult and PetriNetModel into a JSON-friendly dict for the web UI.

The output format is designed for Cytoscape.js consumption (graph nodes/edges)
and for the detail panels (transitions, XOR splits, attribute catalog).
"""

from typing import Any, Dict, List, Optional, Set

import core_utils as utils
from models import (
    AttributeCatalogEntry,
    EffectGuards,
    EffectInfo,
    Guard,
    ParseResult,
    PetriNetModel,
    TransitionInfo,
    XorBranchInfo,
)


def serialize_parse_result(result: ParseResult) -> Dict[str, Any]:
    """Convert a ParseResult into a JSON-serializable dict for the web UI."""
    pnm = result.petri_net_model
    graph = _build_graph(pnm, result)
    transitions = _build_transitions(result)
    xor_splits = _build_xor_splits(pnm, result)
    catalog = _build_attribute_catalog(result.attribute_catalog)

    return {
        "graph": graph,
        "transitions": transitions,
        "xor_splits": xor_splits,
        "artificial_xor_splits": result.artificial_xor_data,
        "attribute_catalog": catalog,
    }


# ---------------------------------------------------------------------------
# Graph (nodes + edges)
# ---------------------------------------------------------------------------

def _build_graph(
    pnm: PetriNetModel, result: ParseResult
) -> Dict[str, Any]:
    """Build Cytoscape-ready node and edge lists from the Petri net."""
    xor_place_names: Set[str] = {p.name for p in pnm.xor_splits}
    artificial_xor_places: Set[str] = result.artificial_xor_places
    variant_to_art_place: Dict[str, str] = result.variant_to_art_place

    and_transition_names: Set[str] = set()
    for t in pnm.petrinet.transitions:
        if not t.label:
            continue
        sanitized = utils.sanitize_name(t.label)
        if sanitized in variant_to_art_place:
            continue
        inputs = pnm.trans_inputs.get(t, set())
        outputs = pnm.trans_outputs.get(t, set())
        if len(inputs) > 1 or len(outputs) > 1:
            and_transition_names.add(sanitized)

    nodes: List[Dict[str, Any]] = []

    for p in pnm.petrinet.places:
        if p.name in artificial_xor_places:
            node_type = "xor_split_artificial"
        elif p.name in xor_place_names:
            node_type = "xor_split"
        else:
            node_type = "place"
        nodes.append({
            "id": p.name,
            "type": node_type,
            "label": p.name,
        })

    for t in pnm.petrinet.transitions:
        t_id = t.name if t.name else str(id(t))
        is_silent = t.label is None
        sanitized = utils.sanitize_name(t.label) if t.label else None

        if is_silent:
            node_type = "silent"
            label = pnm.silent_transitions.get(t, "")
        elif sanitized in variant_to_art_place:
            node_type = "transition"
            label = t.label
        elif sanitized in and_transition_names:
            node_type = "and_split"
            label = sanitized
        else:
            node_type = "transition"
            label = sanitized

        node: Dict[str, Any] = {
            "id": t_id,
            "type": node_type,
            "label": label or "",
            "is_silent": is_silent,
        }
        if sanitized in variant_to_art_place:
            node["art_split_id"] = variant_to_art_place[sanitized]
        nodes.append(node)

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


def _build_effect_entry(attr: str, info: EffectInfo) -> List[Dict[str, Any]]:
    """Build simplified effect entries for one attribute of a transition.

    Flattens the 2A/2B cascade into per-value entries with:
      - preconditions (from value guards if available)
      - attribute, value, probability
      - _meta with cascade levels and stats
    """
    effects: List[Dict[str, Any]] = []

    value_guards_map: Dict[str, List[List[Guard]]] = {}
    if info.value_guards and info.value_guards.guards:
        value_guards_map = info.value_guards.guards

    for value, probability in info.value_probabilities.items():
        val_str = str(value)
        preconditions = _guards_to_preconditions(
            value_guards_map.get(val_str)
        )
        effects.append({
            "attribute": attr,
            "preconditions": preconditions,
            "value": val_str,
            "probability": round(probability, 4),
            "_meta": {
                "presence_probability": round(info.presence_probability, 4),
                "appearance_level": info.appearance_level,
                "value_level": info.value_level,
            },
        })

    return effects


def _build_transitions(result: ParseResult) -> Dict[str, Any]:
    """Build the transitions section of the UI JSON."""
    output: Dict[str, Any] = {}

    for act, t_info in result.transitions.items():
        preconditions: List[List[Dict[str, Any]]] = []
        if t_info.xor_branch and t_info.xor_branch.guards:
            preconditions = _guards_to_preconditions(t_info.xor_branch.guards)

        effects: List[Dict[str, Any]] = []
        for attr, eff_info in t_info.effects.items():
            effects.extend(_build_effect_entry(attr, eff_info))

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
            "effects": effects,
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
        branches: List[Dict[str, Any]] = []
        for t in transitions:
            if not t.label:
                continue
            act = utils.sanitize_name(t.label)
            t_info = result.transitions.get(act)
            if not t_info or not t_info.xor_branch:
                continue

            xb = t_info.xor_branch
            conditions = _guards_to_preconditions(xb.guards)

            branches.append({
                "activity_name": act,
                "conditions": conditions,
                "probability": round(xb.probability, 4),
                "_meta": {
                    "cascade_level": xb.cascade_level,
                    "total_samples": xb.total_samples,
                },
            })

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
        }
        for attr, entry in catalog.items()
    }
