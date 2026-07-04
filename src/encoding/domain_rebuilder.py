"""Rebuild a PDDLDomain from a serialized current.json dict (edited in the web UI)."""

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from encoding.effect_encoder import boolean_effect, categorical_effect
from encoding.pddl_model import (
    PDDLAction, PDDLBaseAction, PDDLCondition, PDDLDomain,
    PDDLDurativeAction, PDDLEffect, PDDLObject, PDDLPredicate, PDDLType,
)
from models import AttributeCatalogEntry

_PLACE_TYPES = frozenset({"place", "xor_split"})
_TRANS_TYPES = frozenset({"transition", "and_split", "silent"})


class DomainRebuilder:
    """Rebuilds a PDDLDomain from the web UI's current.json dict.

    Each labeled transition may produce N >= 1 actions via Cartesian product of:
      Asse A — OR clauses in transition preconditions
      Asse B — OR clauses in XOR-branch conditions
      Asse C — effect groups (one slot per OR clause in each group's guard)
    Silent (tau) transitions produce a single direct place-transfer action.
    """

    def rebuild(
        self,
        data: Dict[str, Any],
        domain_name: str = "test_process",
        use_durative: bool = False,
        use_costs: bool = False,
        has_deadline: bool = False,
    ) -> PDDLDomain:
        """Rebuild a PDDLDomain from the web UI's current.json.

        Args:
            data: Parsed content of current.json.
            domain_name: Name for the PDDL domain.
            use_durative: Encode durative actions when True and duration data is present.
            use_costs: Include action cost effects and functions section when True.
            has_deadline: Add deadline_exceeded predicate and over-all condition when True.

        Returns:
            A fully populated PDDLDomain.
        """
        graph = data["graph"]
        transitions: Dict[str, Any] = data.get("transitions", {})
        xor_splits: Dict[str, Any] = data.get("xor_splits", {})
        catalog: Dict[str, Any] = data.get("attribute_catalog", {})

        out_places_by_label, in_places_by_label, xor_branch_of = (
            self._build_graph_index(graph, xor_splits)
        )
        negated_attributes = self._find_negated_attributes(transitions, xor_splits)

        types = self._build_types(catalog)
        constants = self._build_constants(graph, catalog)
        predicates = self._build_predicates(catalog, negated_attributes)

        actions: List[PDDLBaseAction] = []
        for act_name, t_info in transitions.items():
            actions.extend(
                self._build_transition_actions(
                    act_name, t_info, out_places_by_label, xor_branch_of,
                    catalog, negated_attributes, use_durative,
                )
            )

        for node in graph["nodes"]:
            if node["type"] == "silent":
                tau = self._build_tau_action(node, in_places_by_label, out_places_by_label)
                if tau:
                    actions.append(tau)

        actions.extend(self._build_place_marking_actions(out_places_by_label))

        requirements = [":strips", ":typing"]
        if any(isinstance(a, PDDLDurativeAction) for a in actions):
            requirements.append(":durative-actions")

        has_costs = use_costs
        if has_costs:
            requirements += [":action-costs", ":numeric-fluents"]

        effective_deadline = has_deadline and any(isinstance(a, PDDLDurativeAction) for a in actions)
        return PDDLDomain(
            name=domain_name,
            requirements=requirements,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions,
            has_costs=has_costs,
            has_deadline=effective_deadline,
        )

    # ------------------------------------------------------------------
    # Phase 1 — Graph indexes
    # ------------------------------------------------------------------

    def _build_graph_index(
        self,
        graph: Dict[str, Any],
        xor_splits: Dict[str, Any],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Tuple[str, Any]]]:
        """Build lookup structures from graph nodes/edges.

        Returns:
            out_places_by_label: transition label → list of output place labels.
            in_places_by_label: transition label → list of input place labels.
            xor_branch_of: activity name → (place_name, branch_dict).
        """
        node_by_id = {n["id"]: n for n in graph["nodes"]}

        out_places_by_label: Dict[str, List[str]] = {}
        in_places_by_label: Dict[str, List[str]] = {}

        for n in graph["nodes"]:
            if n["type"] in _TRANS_TYPES:
                label = n.get("label", "")
                if label:
                    out_places_by_label.setdefault(label, [])
                    in_places_by_label.setdefault(label, [])

        for edge in graph["edges"]:
            src_node = node_by_id.get(edge["source"])
            tgt_node = node_by_id.get(edge["target"])
            if not src_node or not tgt_node:
                continue

            src_label = src_node.get("label", "")
            tgt_label = tgt_node.get("label", "")

            if src_node["type"] not in _PLACE_TYPES:
                # transition → place
                if src_label in out_places_by_label:
                    out_places_by_label[src_label].append(tgt_label)
            else:
                # place → transition
                if tgt_label in in_places_by_label:
                    in_places_by_label[tgt_label].append(src_label)

        xor_branch_of: Dict[str, Tuple[str, Any]] = {}
        for place_name, xor_data in xor_splits.items():
            for act_name, branch in xor_data.get("branches", {}).items():
                xor_branch_of[act_name] = (place_name, branch)

        return out_places_by_label, in_places_by_label, xor_branch_of

    # ------------------------------------------------------------------
    # Phase 2 — Negated attributes
    # ------------------------------------------------------------------

    def _find_negated_attributes(
        self,
        transitions: Dict[str, Any],
        xor_splits: Dict[str, Any],
    ) -> Set[str]:
        """Collect attribute names that appear with predicate '<>' anywhere."""
        negated: Set[str] = set()

        def _scan_sop(sop: Any) -> None:
            for clause in (sop or []):
                for cond in clause:
                    if cond.get("predicate") == "<>":
                        negated.add(cond["attribute"])

        for t_info in transitions.values():
            _scan_sop(t_info.get("preconditions", []))
            for group in t_info.get("effect_groups", []):
                _scan_sop(group.get("guard", []))

        for xor_data in xor_splits.values():
            for branch in xor_data.get("branches", {}).values():
                _scan_sop(branch.get("conditions", []))

        return negated

    # ------------------------------------------------------------------
    # Phase 3 — Types, constants, predicates
    # ------------------------------------------------------------------

    def _build_types(self, catalog: Dict[str, Any]) -> List[PDDLType]:
        types = [
            PDDLType("petri_element"),
            PDDLType("place", parent="petri_element"),
            PDDLType("transition", parent="petri_element"),
        ]
        for attr_name, entry in sorted(catalog.items()):
            if entry.get("type") in ("categorical", "numerical"):
                types.append(PDDLType(f"{attr_name}_val"))
        return types

    def _build_constants(
        self, graph: Dict[str, Any], catalog: Dict[str, Any]
    ) -> List[PDDLObject]:
        constants: List[PDDLObject] = []

        for label in sorted(
            n["label"] for n in graph["nodes"] if n["type"] in _PLACE_TYPES
        ):
            constants.append(PDDLObject(label, "place"))

        for label in sorted(
            n["label"]
            for n in graph["nodes"]
            if n["type"] in _TRANS_TYPES and n.get("label")
        ):
            constants.append(PDDLObject(label, "transition"))

        for attr_name, entry in sorted(catalog.items()):
            if entry.get("type") in ("categorical", "numerical"):
                for val in sorted(str(v) for v in entry.get("possible_values", [])):
                    constants.append(PDDLObject(val, f"{attr_name}_val"))

        return constants

    def _build_predicates(
        self, catalog: Dict[str, Any], negated_attributes: Set[str]
    ) -> List[PDDLPredicate]:
        predicates: List[PDDLPredicate] = [
            PDDLPredicate("marked", [("?x", "petri_element")]),
        ]
        for attr_name, entry in sorted(catalog.items()):
            attr_type = entry.get("type")
            if attr_type in ("categorical", "numerical"):
                predicates.append(
                    PDDLPredicate(f"{attr_name}_is", [("?v", f"{attr_name}_val")])
                )
                if attr_name in negated_attributes:
                    predicates.append(
                        PDDLPredicate(f"{attr_name}_is_not", [("?v", f"{attr_name}_val")])
                    )
            elif attr_type == "boolean":
                predicates.append(PDDLPredicate(f"{attr_name}_true"))
                if attr_name in negated_attributes:
                    predicates.append(PDDLPredicate(f"{attr_name}_false"))
        return predicates

    # ------------------------------------------------------------------
    # Phase 4 — Labeled transition actions (3-axis Cartesian product)
    # ------------------------------------------------------------------

    def _build_transition_actions(
        self,
        act_name: str,
        t_info: Dict[str, Any],
        out_places_by_label: Dict[str, List[str]],
        xor_branch_of: Dict[str, Tuple[str, Any]],
        catalog: Dict[str, Any],
        negated_attributes: Set[str],
        use_durative: bool,
    ) -> List[PDDLBaseAction]:
        in_places: List[str] = t_info.get("input_places", [])
        out_places: List[str] = out_places_by_label.get(act_name, [])

        # Asse A — OR in preconditions
        prec_clauses: List[List] = t_info.get("preconditions") or [[]]

        # Asse B — OR in XOR-branch conditions
        xor_branch = None
        xor_prob = 1.0
        xor_clauses: List[List] = [[]]
        if act_name in xor_branch_of:
            _, xor_branch = xor_branch_of[act_name]
            xor_clauses = xor_branch.get("conditions") or [[]]
            xor_prob = float(xor_branch.get("probability", 1.0))

        # Asse C — one slot per OR clause in each effect group's guard
        effect_groups: List[Dict] = t_info.get("effect_groups", [])
        effect_axis: List[Tuple[Optional[Dict], Optional[List]]] = []
        for group in effect_groups:
            guard = group.get("guard") or []
            if not guard:
                effect_axis.append((group, None))
            else:
                for and_clause in guard:
                    effect_axis.append((group, and_clause))
        if not effect_axis:
            effect_axis = [(None, None)]

        total = len(prec_clauses) * len(xor_clauses) * len(effect_axis)
        n = 0
        actions: List[PDDLBaseAction] = []

        for p_clause in prec_clauses:
            for x_clause in xor_clauses:
                for (eff_group, guard_clause) in effect_axis:
                    action_name = (
                        f"execute_{act_name}" if total == 1
                        else f"execute_{act_name}_v{n}"
                    )
                    n += 1

                    preconds: Set[PDDLCondition] = set()
                    for p in in_places:
                        preconds.add(PDDLCondition.marked(p))
                    for cond in self._conditions_from_clause(p_clause, catalog):
                        preconds.add(cond)
                    for cond in self._conditions_from_clause(x_clause, catalog):
                        preconds.add(cond)
                    if guard_clause:
                        for cond in self._conditions_from_clause(guard_clause, catalog):
                            preconds.add(cond)

                    cost = float(t_info.get("cost", 0.0))
                    if xor_branch is not None and not xor_branch.get("conditions") and xor_prob != 1.0:
                        cost += -math.log(xor_prob)
                    if eff_group is not None and guard_clause is None:
                        eff_prob = float(eff_group.get("probability", 1.0))
                        if 0.0 < eff_prob != 1.0:
                            cost += -math.log(eff_prob)

                    effects: List[PDDLEffect] = []
                    for p in in_places:
                        effects.append(PDDLEffect.unmarking(p))
                    effects.append(PDDLEffect.marking(act_name))
                    if eff_group is not None:
                        effects.extend(
                            self._encode_assignments(
                                eff_group.get("assignments", []),
                                catalog,
                                negated_attributes,
                            )
                        )

                    duration = t_info.get("duration")
                    if use_durative and duration is not None:
                        action: PDDLBaseAction = PDDLDurativeAction(
                            name=action_name,
                            duration_min=float(duration["effective_min"]),
                            duration_max=float(duration["effective_max"]),
                            conditions_at_start=preconds,
                            effects_at_end=effects,
                            base_cost=cost if cost != 0.0 else None,
                        )
                    else:
                        action = PDDLAction(
                            name=action_name,
                            preconditions=preconds,
                            effects=effects,
                            base_cost=cost if cost != 0.0 else None,
                        )
                    actions.append(action)

        return actions

    # ------------------------------------------------------------------
    # Phase 5 — Place-marking actions (one per transition)
    # ------------------------------------------------------------------

    def _build_place_marking_actions(
        self, out_places_by_label: Dict[str, List[str]]
    ) -> List[PDDLAction]:
        """One action per transition: consumes transition marker, marks ALL output places."""
        actions: List[PDDLAction] = []
        for trans_label, out_places in sorted(out_places_by_label.items()):
            if not out_places:
                continue
            effects = [PDDLEffect(kind="marked", attribute=trans_label, clear=True)]
            for p in sorted(out_places):
                effects.append(PDDLEffect.marking(p))
            actions.append(PDDLAction(
                name=f"mark_places_from_{trans_label}",
                preconditions={PDDLCondition.marked(trans_label)},
                effects=effects,
            ))
        return actions

    # ------------------------------------------------------------------
    # Phase 6 — Silent (tau) transition actions
    # ------------------------------------------------------------------

    def _build_tau_action(
        self,
        node: Dict[str, Any],
        in_places_by_label: Dict[str, List[str]],
        out_places_by_label: Dict[str, List[str]],
    ) -> Optional[PDDLAction]:
        label = node.get("label", "")
        if not label:
            return None

        in_places = in_places_by_label.get(label, [])

        preconds: Set[PDDLCondition] = {PDDLCondition.marked(p) for p in in_places}
        effects: List[PDDLEffect] = []
        for p in in_places:
            effects.append(PDDLEffect.unmarking(p))
        effects.append(PDDLEffect.marking(label))

        return PDDLAction(
            name=f"execute_{label}",
            preconditions=preconds,
            effects=effects,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _conditions_from_clause(
        self, clause: Optional[List], catalog: Dict[str, Any]
    ) -> List[PDDLCondition]:
        """Convert one AND-clause from a SOP condition to PDDLCondition objects."""
        if not clause:
            return []
        result: List[PDDLCondition] = []
        for cond in clause:
            attr = cond["attribute"]
            value = str(cond.get("value", ""))
            predicate = cond.get("predicate", "=")
            is_bool = catalog.get(attr, {}).get("type") == "boolean"

            if predicate == "=":
                if is_bool:
                    result.append(
                        PDDLCondition.attr_true(attr) if value == "true"
                        else PDDLCondition.attr_false(attr)
                    )
                else:
                    result.append(PDDLCondition.attr_is(attr, value))
            else:  # "<>"
                if is_bool:
                    result.append(
                        PDDLCondition.attr_false(attr) if value == "true"
                        else PDDLCondition.attr_true(attr)
                    )
                else:
                    result.append(PDDLCondition.attr_is_not(attr, value))
        return result

    def _encode_assignments(
        self,
        assignments: List[Dict[str, str]],
        catalog: Dict[str, Any],
        negated_attributes: Set[str],
    ) -> List[PDDLEffect]:
        """Convert a list of {attribute, value} assignments to PDDLEffect objects."""
        effects: List[PDDLEffect] = []
        for asgn in assignments:
            attr = asgn["attribute"]
            value = asgn["value"]
            entry = catalog.get(attr, {})
            attr_type = entry.get("type", "categorical")
            if attr_type == "boolean":
                effects.extend(boolean_effect(attr, value, negated_attributes))
            else:
                pseudo_entry = AttributeCatalogEntry(
                    attribute_type=attr_type,
                    possible_values=entry.get("possible_values", []),
                )
                effects.extend(categorical_effect(attr, value, pseudo_entry, negated_attributes))
        return effects
