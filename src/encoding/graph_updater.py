"""Inject artificial XOR split nodes into the ParseResult before serialization.

Called after EffectDuplicator to materialise action variants as structural
nodes in the PetriNetModel, so that serialize_parse_result() produces a JSON
that already contains the artificial splits — no post-serialization patching.
"""
import copy
import dataclasses
from typing import Any, Dict, List, Set, Tuple

from pm4py.objects.petri_net.obj import PetriNet

import core_utils as utils
from encoding.action_registry import ActionRegistry
from encoding.pddl_model import PDDLAction, PDDLBaseAction, PDDLCondition, PDDLDurativeAction, PDDLEffect
from models import ParseResult, PetriNetModel

logger = utils.get_logger(__name__)

_EXECUTE_PREFIX = "execute_"


def update_parse_result(
    parse_result: ParseResult,
    registry: ActionRegistry,
) -> ParseResult:
    """Inject artificial XOR split nodes into the PetriNetModel for duplicated actions.

    For each base action execute_T that has N>1 variants in the registry:
    - Adds an artificial Place (__art_T, type xor_split_artificial) to the PetriNet.
    - Adds N variant Transition nodes (T_v0 … T_v{N-1}) to the PetriNet.
    - Re-wires arcs: p_in→T→p_out becomes p_in→__art_T, __art_T→T_vi→p_out.
    - Removes the original single Transition T from the PetriNet.
    - Populates parse_result.artificial_xor_places, .variant_to_art_place,
      and .artificial_xor_data so that serialize_parse_result() can produce
      the correct JSON without any post-serialization patching.

    Args:
        parse_result: ParseResult produced by the parsing pipeline.
        registry: ActionRegistry after EffectDuplicator has run.

    Returns:
        A shallow copy of parse_result with the PetriNetModel mutated in-place
        and the three new ParseResult fields populated.
    """
    duplicated = _find_duplicated(registry)
    if not duplicated:
        return parse_result

    pnm = parse_result.petri_net_model
    petri_net = pnm.petrinet

    artificial_xor_places: Set[str] = set()
    variant_to_art_place: Dict[str, str] = {}
    artificial_xor_data: Dict[str, Any] = {}

    for activity, variants in duplicated.items():
        orig_trans = _find_transition(petri_net, activity)
        if orig_trans is None:
            logger.warning("Transition for activity %s not found, skipping", activity)
            continue

        art_id = f"__art_{activity}"

        in_places = list(pnm.trans_inputs.get(orig_trans, set()))
        out_places = list(pnm.trans_outputs.get(orig_trans, set()))

        art_place = PetriNet.Place(name=art_id)
        petri_net.places.add(art_place)
        artificial_xor_places.add(art_id)

        _remove_transition(petri_net, orig_trans)

        _add_arcs(petri_net, [(p, art_place) for p in in_places])

        common_prec, common_eff = _common_conditions_and_effects(variants)
        branches: List[Dict[str, Any]] = []

        for i, variant in enumerate(variants):
            v_id = f"{activity}_v{i}"
            v_label = f"{activity} (v{i})"

            v_trans = PetriNet.Transition(name=v_id, label=v_label)
            petri_net.transitions.add(v_trans)
            variant_to_art_place[v_id] = art_id

            _add_arcs(petri_net, [(art_place, v_trans)] + [(v_trans, p) for p in out_places])

            dist_prec = _get_preconditions(variant) - common_prec
            dist_eff = _get_effects_set(variant) - common_eff
            conditions = [_conditions_to_ui(dist_prec)] if dist_prec else []
            variant_effects = _effects_to_ui(dist_eff)

            branches.append({
                "variant_name": v_label,
                "activity_name": activity,
                "probability": round(variant.effect_probability, 4),
                "conditions": conditions,
                "variant_effects": variant_effects,
            })

        artificial_xor_data[art_id] = {
            "source_activity": activity,
            "branches": branches,
        }

    updated_pnm = _rebuild_indexes(pnm)

    return dataclasses.replace(
        parse_result,
        petri_net_model=updated_pnm,
        artificial_xor_places=artificial_xor_places,
        variant_to_art_place=variant_to_art_place,
        artificial_xor_data=artificial_xor_data,
    )


def save_original_and_current(
    config_dir: str, data: Dict[str, Any]
) -> tuple[bool, bool]:
    """Atomically write original.json and current.json if they do not exist.

    Args:
        config_dir: Directory for the configuration (e.g. data/sepsis/).
        data: Serialized Petri net dict to persist.

    Returns:
        Tuple (original_written, current_written) — True when the file was
        created, False when it already existed and was skipped.
    """
    import json
    import os
    import tempfile

    os.makedirs(config_dir, exist_ok=True)

    written: list[bool] = []
    for filename in ("original.json", "current.json"):
        path = os.path.join(config_dir, filename)
        if os.path.exists(path):
            logger.info("Skipping %s — already exists", path)
            written.append(False)
            continue
        fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            logger.info("Written %s", path)
            written.append(True)
        except Exception:
            os.unlink(tmp_path)
            raise

    return (written[0], written[1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_duplicated(
    registry: ActionRegistry,
) -> Dict[str, List[PDDLBaseAction]]:
    result: Dict[str, List[PDDLBaseAction]] = {}
    for base_name in registry._order:
        variants = registry.get(base_name)
        if not base_name.startswith(_EXECUTE_PREFIX) or len(variants) <= 1:
            continue
        activity = base_name[len(_EXECUTE_PREFIX):]
        result[activity] = list(variants)
    return result


def _find_transition(
    petri_net: PetriNet, activity: str
) -> PetriNet.Transition | None:
    for t in petri_net.transitions:
        if t.label and utils.sanitize_name(t.label) == activity:
            return t
    return None


def _remove_transition(petri_net: PetriNet, trans: PetriNet.Transition) -> None:
    arcs_to_remove = {a for a in petri_net.arcs if a.source is trans or a.target is trans}
    for arc in arcs_to_remove:
        if hasattr(arc.source, "out_arcs"):
            arc.source.out_arcs.discard(arc)
        if hasattr(arc.target, "in_arcs"):
            arc.target.in_arcs.discard(arc)
        petri_net.arcs.discard(arc)
    petri_net.transitions.discard(trans)


def _add_arcs(
    petri_net: PetriNet,
    connections: List[Tuple[Any, Any]],
) -> None:
    for src, tgt in connections:
        arc = PetriNet.Arc(src, tgt)
        petri_net.arcs.add(arc)
        if hasattr(src, "out_arcs"):
            src.out_arcs.add(arc)
        if hasattr(tgt, "in_arcs"):
            tgt.in_arcs.add(arc)


def _rebuild_indexes(pnm: PetriNetModel) -> PetriNetModel:
    """Rebuild trans_inputs, trans_outputs, xor_splits and place_inputs from arcs."""
    trans_inputs: Dict[PetriNet.Transition, Set[PetriNet.Place]] = {}
    trans_outputs: Dict[PetriNet.Transition, Set[PetriNet.Place]] = {}
    place_inputs: Dict[PetriNet.Place, List[PetriNet.Transition]] = {}

    for arc in pnm.petrinet.arcs:
        if isinstance(arc.source, PetriNet.Place) and isinstance(arc.target, PetriNet.Transition):
            trans_inputs.setdefault(arc.target, set()).add(arc.source)
        elif isinstance(arc.source, PetriNet.Transition) and isinstance(arc.target, PetriNet.Place):
            trans_outputs.setdefault(arc.source, set()).add(arc.target)
            place_inputs.setdefault(arc.target, [])
            if arc.source not in place_inputs[arc.target]:
                place_inputs[arc.target].append(arc.source)

    xor_splits: Dict[PetriNet.Place, List[PetriNet.Transition]] = {}
    for place in pnm.petrinet.places:
        out_transitions = [
            arc.target for arc in pnm.petrinet.arcs
            if arc.source is place and isinstance(arc.target, PetriNet.Transition)
        ]
        if len(out_transitions) > 1:
            xor_splits[place] = out_transitions

    activities = {
        utils.sanitize_name(t.label)
        for t in pnm.petrinet.transitions if t.label
    }

    return PetriNetModel(
        petrinet=pnm.petrinet,
        initial_marking=pnm.initial_marking,
        final_marking=pnm.final_marking,
        activities=activities,
        silent_transitions=pnm.silent_transitions,
        trans_inputs=trans_inputs,
        trans_outputs=trans_outputs,
        xor_splits=xor_splits,
        place_inputs=place_inputs,
    )


def _get_preconditions(action: PDDLBaseAction) -> Set[PDDLCondition]:
    if isinstance(action, PDDLDurativeAction):
        all_conds = (
            action.conditions_at_start
            | action.conditions_over_all
            | action.conditions_at_end
        )
    elif isinstance(action, PDDLAction):
        all_conds = action.preconditions
    else:
        return set()
    return {c for c in all_conds if c.kind != "marked"}


def _get_effects_set(action: PDDLBaseAction) -> frozenset:
    if isinstance(action, PDDLDurativeAction):
        all_eff = action.effects_at_start + action.effects_at_end
    elif isinstance(action, PDDLAction):
        all_eff = action.effects
    else:
        return frozenset()
    return frozenset(e for e in all_eff if e.kind != "marked" and not e.clear)


def _common_conditions_and_effects(
    variants: List[PDDLBaseAction],
) -> Tuple[Set[PDDLCondition], frozenset]:
    prec_sets = [_get_preconditions(v) for v in variants]
    eff_sets = [_get_effects_set(v) for v in variants]
    common_prec = set.intersection(*prec_sets) if prec_sets else set()
    common_eff = frozenset.intersection(*eff_sets) if eff_sets else frozenset()
    return common_prec, common_eff


def _conditions_to_ui(conditions: Set[PDDLCondition]) -> List[Dict[str, Any]]:
    result = []
    for c in sorted(conditions, key=lambda x: (x.attribute, x.kind, x.value or "")):
        if c.kind == "attr_is":
            result.append({"attribute": c.attribute, "predicate": "=", "value": c.value})
        elif c.kind == "attr_is_not":
            result.append({"attribute": c.attribute, "predicate": "<>", "value": c.value})
        elif c.kind == "attr_true":
            result.append({"attribute": c.attribute, "predicate": "=", "value": "true"})
        elif c.kind == "attr_false":
            result.append({"attribute": c.attribute, "predicate": "=", "value": "false"})
    return result


def _effects_to_ui(effects: frozenset) -> List[Dict[str, Any]]:
    result = []
    for e in sorted(effects, key=lambda x: (x.attribute, x.kind, x.value or "")):
        if e.kind == "attr_is":
            result.append({"attribute": e.attribute, "predicate": "=", "value": e.value})
        elif e.kind == "attr_true":
            result.append({"attribute": e.attribute, "predicate": "=", "value": "true"})
    return result
